from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import geopandas as gpd
import h5py
import numpy as np
from tqdm import tqdm

from hestradiomics.extract.constants import (
    CELL_CLASS_COLUMN,
    EXTRACTOR_DEFAULT_LABEL,
    PATCH_IDX_COLUMN,
)
from hestradiomics.utils import (
    build_threshold_mask,
    filter_sample_ids,
    get_barcodes_key,
    get_coords_key,
    get_img_key,
    load_cellseg_dataframe,
    load_patch_data,
    normalize_class_name,
    rasterize_geometries_to_mask,
)

VIS_DIR_COLOR = "color"
VIS_DIR_GRAY = "gray"
VIS_DIR_MASKED_GRAY_THRESHOLD = "masked_gray_threshold"
VIS_DIR_MASKED_COLOR_THRESHOLD = "masked_color_threshold"
VIS_DIR_MASKED_GRAY_CELLSEG = "masked_gray_cellseg"
VIS_DIR_MASKED_COLOR_CELLSEG = "masked_color_cellseg"

MASK_FOREGROUND_THRESHOLD = 0


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    if img.size > 0 and img.max() <= 1.0:
        img = img * 255.0

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _save_png(path: Path, img: np.ndarray, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    img = _ensure_uint8(img)

    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def _to_gray(color_or_gray: np.ndarray) -> np.ndarray:
    img = _ensure_uint8(color_or_gray)

    if img.ndim == 2:
        return img

    if img.ndim == 3 and img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    raise ValueError(f"Unsupported image shape for gray conversion: {img.shape}")


def _apply_mask_to_gray(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = _ensure_uint8(gray)
    return np.where(mask > MASK_FOREGROUND_THRESHOLD, gray, 0).astype(np.uint8)


def _apply_mask_to_color(color: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = _ensure_uint8(color)

    mask_bool = mask > MASK_FOREGROUND_THRESHOLD

    if mask_bool.ndim == 2 and color.ndim == 3:
        mask_bool = mask_bool[..., None]

    out = np.zeros_like(color)
    out[mask_bool.repeat(color.shape[-1], axis=-1) if color.ndim == 3 else mask_bool] = (
        color[mask_bool.repeat(color.shape[-1], axis=-1) if color.ndim == 3 else mask_bool]
    )

    return out


def _select_patch_indices(
    total_num_patches: int,
    vis_ratio: float = 1.0,
    patch_indices: Optional[List[int]] = None,
) -> List[int]:
    if patch_indices is not None:
        return [int(i) for i in patch_indices]

    if not (0 < vis_ratio <= 1):
        raise ValueError(f"vis_ratio must be in (0, 1], got {vis_ratio}")

    num_vis = max(1, math.ceil(total_num_patches * vis_ratio))

    if num_vis >= total_num_patches:
        return list(range(total_num_patches))

    return np.linspace(
        0,
        total_num_patches - 1,
        num=num_vis,
        dtype=int,
    ).tolist()


def get_patch_cellseg(
    cellseg_df: gpd.GeoDataFrame,
    patch_idx: int,
) -> gpd.GeoDataFrame:
    patch_cellseg = cellseg_df[
        cellseg_df[PATCH_IDX_COLUMN] == patch_idx
    ].copy()

    patch_cellseg = patch_cellseg[
        patch_cellseg.geometry.notnull()
    ].copy()

    if len(patch_cellseg) > 0 and CELL_CLASS_COLUMN in patch_cellseg.columns:
        patch_cellseg[CELL_CLASS_COLUMN] = (
            patch_cellseg[CELL_CLASS_COLUMN]
            .map(normalize_class_name)
        )

    return patch_cellseg


def save_patch_visualization_variants(
    patch,
    output_dir: str | Path,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    cellseg_df: Optional[gpd.GeoDataFrame] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    filename = f"{patch.base_filename}.png"

    color = _ensure_uint8(patch.color_patch)
    gray = _to_gray(patch.gray_patch)

    _save_png(output_dir / VIS_DIR_COLOR / filename, color, overwrite=overwrite)
    _save_png(output_dir / VIS_DIR_GRAY / filename, gray, overwrite=overwrite)

    threshold_mask = build_threshold_mask(
        gray,
        label=label,
    )

    _save_png(
        output_dir / VIS_DIR_MASKED_GRAY_THRESHOLD / filename,
        _apply_mask_to_gray(gray, threshold_mask),
        overwrite=overwrite,
    )

    _save_png(
        output_dir / VIS_DIR_MASKED_COLOR_THRESHOLD / filename,
        _apply_mask_to_color(color, threshold_mask),
        overwrite=overwrite,
    )

    has_cellseg = False

    if cellseg_df is not None:
        patch_cellseg = get_patch_cellseg(
            cellseg_df=cellseg_df,
            patch_idx=patch.patch_idx,
        )

        if len(patch_cellseg) > 0:
            has_cellseg = True

            cellseg_mask = rasterize_geometries_to_mask(
                patch_cellseg.geometry.tolist(),
                image_shape=gray.shape,
                label=label,
            )

            _save_png(
                output_dir / VIS_DIR_MASKED_GRAY_CELLSEG / filename,
                _apply_mask_to_gray(gray, cellseg_mask),
                overwrite=overwrite,
            )

            _save_png(
                output_dir / VIS_DIR_MASKED_COLOR_CELLSEG / filename,
                _apply_mask_to_color(color, cellseg_mask),
                overwrite=overwrite,
            )

    return {
        "patch_idx": int(patch.patch_idx),
        "barcode": patch.barcode,
        "has_cellseg": has_cellseg,
    }


def save_patch_visualizations_from_h5(
    h5_path: str | Path,
    output_dir: str | Path,
    sample_id: Optional[str] = None,
    cellseg_path: Optional[str | Path] = None,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    vis_ratio: float = 1.0,
    overwrite: bool = False,
    patch_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    sample_id = sample_id or h5_path.stem

    cellseg_df = None
    if cellseg_path is not None:
        cellseg_path = Path(cellseg_path)
        if cellseg_path.exists():
            cellseg_df = load_cellseg_dataframe(str(cellseg_path))
        else:
            print(f"[WARN] cellseg file not found: {cellseg_path}")

    rows: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as f:
        img_key = get_img_key(f)
        coords_key = get_coords_key(f)
        barcodes_key = get_barcodes_key(f)

        total_num_patches = len(f[img_key])
        indices = _select_patch_indices(
            total_num_patches=total_num_patches,
            vis_ratio=vis_ratio,
            patch_indices=patch_indices,
        )

        for i in tqdm(indices, desc=f"[Visualizing patches] {sample_id}"):
            patch = load_patch_data(
                f=f,
                img_key=img_key,
                coords_key=coords_key,
                barcodes_key=barcodes_key,
                patch_idx=i,
            )

            row = save_patch_visualization_variants(
                patch=patch,
                output_dir=output_dir / sample_id,
                label=label,
                cellseg_df=cellseg_df,
                overwrite=overwrite,
            )

            rows.append(row)

    return rows


def patch_visualization_from_oncotrees(
    oncotrees: List[str],
    download_dir: str | Path,
    sample_ids: Optional[tuple[str, ...]] = None,
    patch_dirname: str = "patches",
    cellseg_dirname: str = "segment",
    patch_vis_dirname: str = "patches_vis",
    label: int = EXTRACTOR_DEFAULT_LABEL,
    vis_ratio: float = 1.0,
    overwrite_visualization: bool = False,
) -> None:
    download_dir = Path(download_dir)

    for oncotree in oncotrees:
        oncotree_root = download_dir / oncotree

        patch_dir = oncotree_root / patch_dirname
        cellseg_dir = oncotree_root / cellseg_dirname
        patch_vis_root = oncotree_root / patch_vis_dirname

        if not patch_dir.exists():
            print(f"[SKIP] Patch dir not found: {patch_dir}")
            continue

        h5_paths = sorted(patch_dir.glob("*.h5"))

        if not h5_paths:
            print(f"[SKIP] No h5 files found: {patch_dir}")
            continue

        all_sample_ids = [path.stem for path in h5_paths]

        target_sample_ids = filter_sample_ids(
            all_sample_ids=all_sample_ids,
            selected_sample_ids=sample_ids,
        )

        if not target_sample_ids:
            print(f"[SKIP] No selected samples found for {oncotree}")
            continue

        target_sample_id_set = set(target_sample_ids)

        h5_paths = [
            path for path in h5_paths
            if path.stem in target_sample_id_set
        ]

        print("=" * 80)
        print(f"[ONCOTREE] {oncotree}")
        print(f"[PATCH DIR] {patch_dir}")
        print(f"[CELLSEG DIR] {cellseg_dir}")
        print(f"[OUTPUT DIR] {patch_vis_root}")
        print(f"[NUM SAMPLES] {len(h5_paths)} / {len(all_sample_ids)}")
        print(f"[VIS RATIO] {vis_ratio}")
        print(f"[OVERWRITE] {overwrite_visualization}")
        print("=" * 80)

        for h5_path in h5_paths:
            sample_id = h5_path.stem
            cellseg_path = cellseg_dir / f"{sample_id}.h5"

            print(f"\n[SAMPLE] {sample_id}")

            save_patch_visualizations_from_h5(
                h5_path=h5_path,
                output_dir=patch_vis_root,
                sample_id=sample_id,
                cellseg_path=(
                    cellseg_path
                    if cellseg_path.exists()
                    else None
                ),
                label=label,
                vis_ratio=vis_ratio,
                overwrite=overwrite_visualization,
            )
    