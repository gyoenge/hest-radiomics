from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hestradiomics.config import CellSegmentConfig, DownloadConfig
from hestradiomics.segment.adapter import CellViTInferenceAdapter
from hestradiomics.segment.io import (
    H5PatchDataset,
    collate_patches,
    save_cellseg_h5,
)
from hestradiomics.segment.visualize import save_overlays_from_cellseg_h5
from hestradiomics.utils import filter_sample_ids


def _gdf_to_cell_rows(
    gdf: gpd.GeoDataFrame,
    patch_idx: int,
    barcode: str,
) -> List[Dict[str, Any]]:
    rows = []

    if len(gdf) == 0:
        return rows

    gdf = gdf.copy()

    if "cell_id_in_patch" not in gdf.columns:
        gdf["cell_id_in_patch"] = list(range(1, len(gdf) + 1))

    for _, row in gdf.iterrows():
        rows.append(
            {
                "patch_idx": int(patch_idx),
                "barcode": barcode,
                "cell_id_in_patch": int(row["cell_id_in_patch"]),
                "class_id": (
                    int(row["class_id"])
                    if "class_id" in gdf.columns and pd.notna(row["class_id"])
                    else -1
                ),
                "class_name": (
                    str(row["class_name"])
                    if "class_name" in gdf.columns and pd.notna(row["class_name"])
                    else "unknown"
                ),
                "geometry": row.geometry,
            }
        )

    return rows


def segment_h5_patches_with_cellvit(
    h5_path: str,
    seg_h5_path: str,
    model_path: str,
    runtime_dir: str,
    summary_json_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    patch_indices: Optional[List[int]] = None,
    device: str = "cuda:0",
    predictor: Optional[CellViTInferenceAdapter] = None,
) -> str:
    os.makedirs(os.path.dirname(seg_h5_path), exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)

    dataset = H5PatchDataset(
        h5_path=h5_path,
        patch_indices=patch_indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patches,
        pin_memory=torch.cuda.is_available(),
    )

    if predictor is None:
        predictor = CellViTInferenceAdapter(
            model_path=model_path,
            model_name=os.path.basename(model_path),
            output_dir=os.path.dirname(seg_h5_path),
            runtime_dir=runtime_dir,
            device=device,
        )

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    pbar = tqdm(
        total=len(dataset),
        desc="Segmenting patches",
    )

    for batch in loader:
        images = batch["images"]
        patch_idxs = batch["patch_idx"]
        barcodes = batch["barcodes"]
        coords = batch["coords"]

        gdfs = predictor.predict_batch_to_gdfs(images)

        for gdf, patch_idx, barcode, coord in zip(
            gdfs,
            patch_idxs,
            barcodes,
            coords,
        ):
            rows = _gdf_to_cell_rows(
                gdf=gdf,
                patch_idx=patch_idx,
                barcode=barcode,
            )

            all_rows.extend(rows)

            summary_rows.append(
                {
                    "patch_idx": int(patch_idx),
                    "barcode": barcode,
                    "coord_raw": (
                        None
                        if coord is None
                        else np.asarray(coord).tolist()
                    ),
                    "n_cells": int(len(gdf)),
                }
            )

            pbar.update(1)

    pbar.close()

    save_cellseg_h5(
        rows=all_rows,
        summary_rows=summary_rows,
        save_path=seg_h5_path,
    )

    summary = {
        "h5_path": h5_path,
        "segment_dir": os.path.dirname(seg_h5_path),
        "runtime_dir": runtime_dir,
        "num_patches": len(dataset),
        "num_polygons": int(len(all_rows)),
        "seg_h5_path": seg_h5_path,
    }

    if summary_json_path is not None:
        os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

    return seg_h5_path


def list_sample_ids_from_patches(
    oncotree_root: str,
) -> List[str]:
    patches_dir = os.path.join(
        oncotree_root,
        "patches",
    )

    if not os.path.isdir(patches_dir):
        print(f"[WARN] patches dir not found: {patches_dir}")
        return []

    return [
        os.path.splitext(filename)[0]
        for filename in sorted(os.listdir(patches_dir))
        if filename.endswith(".h5")
    ]


def build_sample_paths(
    hest_root: str,
    oncotree: str,
    sample_id: str,
) -> Dict[str, str]:
    data_root = os.path.join(
        hest_root,
        oncotree,
    )

    segment_dir = os.path.join(
        data_root,
        "segment",
    )

    segment_vis_dir = os.path.join(
        data_root,
        "segment_vis",
    )

    return {
        "data_root": data_root,
        "patch_h5_path": os.path.join(data_root, "patches", f"{sample_id}.h5"),
        "segment_dir": segment_dir,
        "segment_vis_dir": segment_vis_dir,
        "seg_h5_path": os.path.join(segment_dir, f"{sample_id}.h5"),
        "summary_json_path": os.path.join(segment_dir, f"{sample_id}.summary.json"),
        "runtime_dir": os.path.join(segment_dir, "_cellvit_runtime"),
        "overlay_dir": os.path.join(segment_vis_dir, sample_id),
    }


def segment_one_sample(
    hest_root: str,
    oncotree: str,
    sample_id: str,
    model_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Optional[str]:
    paths = build_sample_paths(
        hest_root=hest_root,
        oncotree=oncotree,
        sample_id=sample_id,
    )

    patch_h5_path = paths["patch_h5_path"]
    seg_h5_path = paths["seg_h5_path"]

    if not os.path.exists(patch_h5_path):
        print(f"[WARN] patch h5 not found: {patch_h5_path}")
        return None

    if os.path.exists(seg_h5_path) and not overwrite:
        print(f"[SKIP] segment exists: {seg_h5_path}")
        return seg_h5_path

    return segment_h5_patches_with_cellvit(
        h5_path=patch_h5_path,
        seg_h5_path=seg_h5_path,
        model_path=model_path,
        runtime_dir=paths["runtime_dir"],
        summary_json_path=paths["summary_json_path"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


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


def segment_all_oncotrees(
    hest_root: str,
    oncotrees: List[str],
    model_path: str,
    sample_ids: Optional[Tuple[str, ...]] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> List[str]:
    output_paths = []

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
            seg_h5_path = segment_one_sample(
                hest_root=hest_root,
                oncotree=oncotree,
                sample_id=sample_id,
                model_path=model_path,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                overwrite=overwrite,
            )

            if seg_h5_path is not None:
                output_paths.append(seg_h5_path)

    return output_paths


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


def segment_all_oncotrees_from_config(
    download_cfg: DownloadConfig,
    cellseg_cfg: CellSegmentConfig,
    sample_ids: Optional[Tuple[str, ...]] = None,
) -> List[str]:
    return segment_all_oncotrees(
        hest_root=str(download_cfg.download_dir),
        oncotrees=list(download_cfg.oncotrees),
        model_path=str(cellseg_cfg.model_path),
        sample_ids=sample_ids,
        batch_size=cellseg_cfg.batch_size,
        num_workers=cellseg_cfg.num_workers,
        device=cellseg_cfg.device,
        overwrite=cellseg_cfg.overwrite_segment,
    )


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
