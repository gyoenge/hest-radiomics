from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

from shapely.geometry import Polygon
import pandas as pd
import h5py
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm


from hestradiomics.segment.adapter import iter_polygons
from hestradiomics.segment.io import (
    H5PatchDataset,
    build_sample_paths,
    list_sample_ids_from_patches,
)
from hestradiomics.utils import ensure_uint8_rgb, filter_sample_ids


CLASS_COLOR_MAP = {
    "neoplastic": "#ff4d4d",
    "inflammatory": "#4da6ff",
    "connective": "#00c853",
    "dead": "#ffd54f",
    "epithelial": "#ab47bc",
    "background": "#9e9e9e",
    "unknown": "#00ffff",
}

DEFAULT_OVERLAY_COLOR = "#00ffff"


def _select_patch_indices(
    patch_indices_all: List[int],
    vis_ratio: float = 1.0,
    patch_indices: Optional[List[int]] = None,
) -> List[int]:
    if patch_indices is not None:
        return [int(i) for i in patch_indices]

    if not (0 < vis_ratio <= 1):
        raise ValueError(f"vis_ratio must be in (0, 1], got {vis_ratio}")

    patch_indices_all = [int(i) for i in patch_indices_all]
    total_num_patches = len(patch_indices_all)

    if total_num_patches == 0:
        return []

    num_vis = max(1, math.ceil(total_num_patches * vis_ratio))

    if num_vis >= total_num_patches:
        return patch_indices_all

    selected_positions = np.linspace(
        0,
        total_num_patches - 1,
        num=num_vis,
        dtype=int,
    )

    return [patch_indices_all[i] for i in selected_positions]


def load_cellseg_h5(seg_h5_path):
    rows = []

    with h5py.File(seg_h5_path, "r") as f:
        print("[H5 KEYS]", list(f.keys()))

        def print_tree(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name, obj.shape, obj.dtype)
            else:
                print(name)
        f.visititems(print_tree)

        # 실제 key 이름 확인 필요
        # 예: cells/contours, cells/patch_idx, cells/class_name 등
        contours = f["contours"][:]
        patch_indices = f["patch_idx"][:]

        classes = None
        if "class_name" in f:
            classes = f["class_name"][:]

        for i, contour in enumerate(contours):
            contour = np.asarray(contour)

            if contour.ndim != 2 or contour.shape[0] < 3:
                continue

            geom = Polygon(contour)

            if not geom.is_valid or geom.is_empty:
                continue

            row = {
                "patch_idx": int(patch_indices[i]),
                "geometry": geom,
            }

            if classes is not None:
                cls = classes[i]
                if isinstance(cls, bytes):
                    cls = cls.decode()
                row["class_name"] = cls

            rows.append(row)

    seg_df = pd.DataFrame(rows)

    if seg_df.empty:
        return gpd.GeoDataFrame(
            seg_df,
            geometry=[],
            crs=None,
        ), pd.DataFrame()

    seg_gdf = gpd.GeoDataFrame(
        seg_df,
        geometry="geometry",
        crs=None,
    )

    return seg_gdf, pd.DataFrame()


def save_overlay_png(
    img: np.ndarray,
    gdf: gpd.GeoDataFrame,
    save_path: str,
    title: Optional[str] = None,
    use_class_color: bool = True,
) -> None:
    img = ensure_uint8_rgb(img)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    if use_class_color and len(gdf) > 0 and "class_name" in gdf.columns:
        for class_name, sub_gdf in gdf.groupby("class_name"):
            color = CLASS_COLOR_MAP.get(
                str(class_name).lower(),
                DEFAULT_OVERLAY_COLOR,
            )

            patches = []

            for geom in sub_gdf.geometry:
                for poly in iter_polygons(geom):
                    xy = np.asarray(poly.exterior.coords)

                    if len(xy) >= 3:
                        patches.append(MplPolygon(xy, closed=True))

            if patches:
                ax.add_collection(
                    PatchCollection(
                        patches,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.0,
                    )
                )

    else:
        patches = []

        for geom in gdf.geometry:
            for poly in iter_polygons(geom):
                xy = np.asarray(poly.exterior.coords)

                if len(xy) >= 3:
                    patches.append(MplPolygon(xy, closed=True))

        if patches:
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor="none",
                    edgecolor="lime",
                    linewidth=1.0,
                )
            )

    if use_class_color:
        handles = [
            Line2D([0], [0], color=color, lw=2, label=class_name)
            for class_name, color in CLASS_COLOR_MAP.items()
        ]

        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=8,
        )

    if title:
        ax.set_title(title)

    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )

    plt.close(fig)


def save_overlays_from_cellseg_h5(
    source_h5_path: str,
    seg_h5_path: str,
    overlay_dir: str,
    use_class_color: bool = True,
    vis_ratio: float = 1.0,
    patch_indices: Optional[List[int]] = None,
    overwrite: bool = False,
) -> str:
    os.makedirs(overlay_dir, exist_ok=True)

    seg_gdf, patch_df = load_cellseg_h5(seg_h5_path)

    all_patch_indices = patch_df["patch_idx"].astype(int).tolist()

    selected_patch_indices = _select_patch_indices(
        patch_indices_all=all_patch_indices,
        vis_ratio=vis_ratio,
        patch_indices=patch_indices,
    )

    patch_df = patch_df[
        patch_df["patch_idx"].astype(int).isin(selected_patch_indices)
    ].copy()

    seg_gdf = seg_gdf[
        seg_gdf["patch_idx"].astype(int).isin(selected_patch_indices)
    ].copy()

    dataset = H5PatchDataset(
        h5_path=source_h5_path,
        patch_indices=selected_patch_indices,
    )

    gdf_by_patch = {
        int(patch_idx): sub_gdf.copy()
        for patch_idx, sub_gdf in seg_gdf.groupby("patch_idx")
    }

    for i in tqdm(range(len(dataset)), desc="Saving segment overlays"):
        sample = dataset[i]

        patch_idx = int(sample["patch_idx"])
        barcode = str(sample["barcode"])
        img = sample["image"]

        gdf = gdf_by_patch.get(
            patch_idx,
            gpd.GeoDataFrame(
                {
                    "cell_id_in_patch": [],
                    "class_id": [],
                    "class_name": [],
                },
                geometry=[],
                crs=None,
            ),
        )

        save_path = os.path.join(
            overlay_dir,
            f"{barcode}_idx{patch_idx}.png",
        )

        if os.path.exists(save_path) and not overwrite:
            continue

        save_overlay_png(
            img=img,
            gdf=gdf,
            save_path=save_path,
            title=f"idx={patch_idx}, bc={barcode}, cells={len(gdf)}",
            use_class_color=use_class_color,
        )

    return overlay_dir


def save_overlay_one_sample(
    hest_root: str,
    oncotree: str,
    sample_id: str,
    use_class_color: bool = True,
    vis_ratio: float = 1.0,
    overwrite: bool = False,
    patch_indices: Optional[List[int]] = None,
) -> Optional[str]:
    paths = build_sample_paths(
        hest_root=hest_root,
        oncotree=oncotree,
        sample_id=sample_id,
    )

    patch_h5_path = paths["patch_h5_path"]
    seg_h5_path = paths["seg_h5_path"]
    overlay_dir = paths["overlay_dir"]

    if not os.path.exists(patch_h5_path):
        print(f"[WARN] patch h5 not found: {patch_h5_path}")
        return None

    if not os.path.exists(seg_h5_path):
        print(f"[WARN] segment h5 not found: {seg_h5_path}")
        return None

    if (
        os.path.isdir(overlay_dir)
        and len(os.listdir(overlay_dir)) > 0
        and not overwrite
        and patch_indices is None
        and vis_ratio >= 1.0
    ):
        print(f"[SKIP] overlay exists: {overlay_dir}")
        return overlay_dir

    return save_overlays_from_cellseg_h5(
        source_h5_path=patch_h5_path,
        seg_h5_path=seg_h5_path,
        overlay_dir=overlay_dir,
        use_class_color=use_class_color,
        vis_ratio=vis_ratio,
        patch_indices=patch_indices,
        overwrite=overwrite,
    )


def segment_visualization_from_oncotrees(
    hest_root: str,
    oncotrees: List[str],
    sample_ids: Optional[Tuple[str, ...]] = None,
    use_class_color: bool = True,
    vis_ratio: float = 1.0,
    overwrite: bool = False,
    patch_indices: Optional[List[int]] = None,
) -> List[str]:
    output_dirs: List[str] = []

    for oncotree in oncotrees:
        oncotree_root = os.path.join(hest_root, oncotree)

        if not os.path.isdir(oncotree_root):
            print(f"[SKIP] oncotree root not found: {oncotree_root}")
            continue

        all_sample_ids = list_sample_ids_from_patches(oncotree_root)

        target_sample_ids = filter_sample_ids(
            all_sample_ids=all_sample_ids,
            selected_sample_ids=sample_ids,
        )

        if not target_sample_ids:
            print(f"[SKIP] No selected samples found for {oncotree}")
            continue

        print("=" * 80)
        print(f"[ONCOTREE] {oncotree}")
        print(f"[ROOT] {oncotree_root}")
        print(f"[NUM SAMPLES] {len(target_sample_ids)} / {len(all_sample_ids)}")
        print(f"[VIS RATIO] {vis_ratio}")
        print(f"[USE CLASS COLOR] {use_class_color}")
        print(f"[OVERWRITE] {overwrite}")
        print("=" * 80)

        for sample_id in target_sample_ids:
            print(f"\n[SAMPLE] {sample_id}")

            overlay_dir = save_overlay_one_sample(
                hest_root=hest_root,
                oncotree=oncotree,
                sample_id=sample_id,
                use_class_color=use_class_color,
                vis_ratio=vis_ratio,
                overwrite=overwrite,
                patch_indices=patch_indices,
            )

            if overlay_dir is not None:
                output_dirs.append(overlay_dir)

    return output_dirs
    