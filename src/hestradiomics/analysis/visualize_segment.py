from __future__ import annotations

import os
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm

from hestradiomics.config import CellSegmentConfig, DownloadConfig
from hestradiomics.segment.adapter import iter_polygons
from hestradiomics.segment.io import (
    H5PatchDataset, load_cellseg_h5, 
    list_sample_ids_from_patches, build_sample_paths, 
)
from hestradiomics.utils import ensure_uint8_rgb, filter_sample_ids

def save_overlay_png(
    img: np.ndarray,
    gdf: gpd.GeoDataFrame,
    save_path: str,
    title: Optional[str] = None,
    use_class_color: bool = True,
) -> None:
    img = ensure_uint8_rgb(img)

    class_color_map = {
        "neoplastic": "#ff4d4d",
        "inflammatory": "#4da6ff",
        "connective": "#00c853",
        "dead": "#ffd54f",
        "epithelial": "#ab47bc",
        "background": "#9e9e9e",
        "unknown": "#00ffff",
    }

    default_color = "#00ffff"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    if use_class_color and len(gdf) > 0:
        for class_name, sub_gdf in gdf.groupby("class_name"):
            color = class_color_map.get(
                str(class_name).lower(),
                default_color,
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
            for class_name, color in class_color_map.items()
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
    patch_indices: Optional[List[int]] = None,
) -> str:
    os.makedirs(overlay_dir, exist_ok=True)

    seg_gdf, patch_df = load_cellseg_h5(seg_h5_path)

    if patch_indices is not None:
        patch_df = patch_df[
            patch_df["patch_idx"].isin(patch_indices)
        ].copy()

        seg_gdf = seg_gdf[
            seg_gdf["patch_idx"].isin(patch_indices)
        ].copy()

    dataset = H5PatchDataset(
        h5_path=source_h5_path,
        patch_indices=patch_df["patch_idx"].astype(int).tolist(),
    )

    gdf_by_patch = {
        int(patch_idx): sub_gdf.copy()
        for patch_idx, sub_gdf in seg_gdf.groupby("patch_idx")
    }

    for i in tqdm(range(len(dataset)), desc="Saving overlays"):
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
    overwrite: bool = False,
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
    ):
        print(f"[SKIP] overlay exists: {overlay_dir}")
        return overlay_dir

    return save_overlays_from_cellseg_h5(
        source_h5_path=patch_h5_path,
        seg_h5_path=seg_h5_path,
        overlay_dir=overlay_dir,
        use_class_color=use_class_color,
    )



def save_overlays_all_oncotrees(
    hest_root: str,
    oncotrees: List[str],
    sample_ids: Optional[Tuple[str, ...]] = None,
    use_class_color: bool = True,
    overwrite: bool = False,
) -> List[str]:
    output_dirs = []

    for oncotree in oncotrees:
        oncotree_root = os.path.join(
            hest_root,
            oncotree,
        )

        all_sample_ids = list_sample_ids_from_patches(
            oncotree_root
        )

        target_sample_ids = filter_sample_ids(
            all_sample_ids=all_sample_ids,
            selected_sample_ids=sample_ids,
        )

        for sample_id in target_sample_ids:
            overlay_dir = save_overlay_one_sample(
                hest_root=hest_root,
                oncotree=oncotree,
                sample_id=sample_id,
                use_class_color=use_class_color,
                overwrite=overwrite,
            )

            if overlay_dir is not None:
                output_dirs.append(overlay_dir)

    return output_dirs


def save_overlays_all_oncotrees_from_config(
    download_cfg: DownloadConfig,
    cellseg_cfg: CellSegmentConfig,
    sample_ids: Optional[Tuple[str, ...]] = None,
) -> List[str]:
    return save_overlays_all_oncotrees(
        hest_root=str(download_cfg.download_dir),
        oncotrees=list(download_cfg.oncotrees),
        sample_ids=sample_ids,
        use_class_color=cellseg_cfg.use_class_color,
        overwrite=cellseg_cfg.overwrite_overlay,
    )
