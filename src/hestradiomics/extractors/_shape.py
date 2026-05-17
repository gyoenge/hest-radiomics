from __future__ import annotations

from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import SimpleITK as sitk

from hestradiomics.extractors.constants import *

from hestradiomics.utils import (
    build_local_polygon_mask,
    crop_patch_by_bbox,
    align_local_mask_to_crop,
    normalize_class_name,
    strip_shape2d_prefix,
)

from hestradiomics.extractors.builders import (
    build_shape2d_extractor,
)


# ------------------------------------------------------------------------------
# Radiomics execution helpers
# ------------------------------------------------------------------------------
# This module contains utilities for:
#
#   1. Per-cell Shape2D extraction
#   2. Morphology feature aggregation
#   3. First-order statistical summarization
#
# Main workflow:
#
#   cell polygon
#       ↓
#   local binary mask
#       ↓
#   Shape2D radiomics extraction
#       ↓
#   per-cell morphology vectors
#       ↓
#   patch-level aggregation statistics
#
# ------------------------------------------------------------------------------


def _execute_firstorder_aggregation(
    values: List[float],
) -> Dict[str, float]:
    """
    Safely compute first-order statistics for a morphology feature vector.

    This function is intentionally implemented manually instead of using
    PyRadiomics first-order features because:
        - morphology vectors may be very short
        - degenerate distributions can produce RuntimeWarnings
        - manual aggregation is more stable and lightweight

    Example input:
        [12.3, 14.1, 13.8, 11.9]

    Example output:
        {
            "mean": ...,
            "variance": ...,
            "entropy": ...,
            ...
        }

    Args:
        values:
            List of numeric morphology values.

    Returns:
        Dictionary of aggregated statistics.
    """

    # Convert safely to numeric
    s = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    # Remove inf / NaN values
    s = (
        s.replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # Handle empty input safely
    if len(s) == 0:
        return {}

    arr = s.to_numpy(dtype=np.float64)
    n = len(arr)

    # --------------------------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------------------------
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    range_val = float(max_val - min_val)

    var_val = float(np.var(arr, ddof=0))
    std_val = float(np.std(arr, ddof=0))

    # --------------------------------------------------------------------------
    # Percentile statistics
    # --------------------------------------------------------------------------
    p10_val = float(np.percentile(arr, 10))
    p25_val = float(np.percentile(arr, 25))

    p75_val = float(np.percentile(arr, 75))
    p90_val = float(np.percentile(arr, 90))

    iqr_val = float(p75_val - p25_val)

    # --------------------------------------------------------------------------
    # Distribution shape statistics
    # --------------------------------------------------------------------------
    if std_val == 0.0:

        # Constant distribution
        skew_val = 0.0
        kurt_val = 0.0

    else:

        # Standardized values
        z = (arr - mean_val) / std_val

        # Third standardized moment
        skew_val = float(np.mean(z ** 3))

        # Fourth standardized moment
        kurt_val = float(np.mean(z ** 4) - 3.0)

    # --------------------------------------------------------------------------
    # Entropy estimation
    # --------------------------------------------------------------------------
    if n <= 1 or min_val == max_val:

        # Entropy undefined for constant distributions
        entropy_val = 0.0

    else:

        # Histogram-based entropy approximation
        hist, _ = np.histogram(
            arr,
            bins=min(16, max(2, n)),
        )

        prob = hist.astype(np.float64)

        # Remove zero-probability bins
        prob = prob[prob > 0]

        # Normalize histogram
        prob = prob / prob.sum()

        # Shannon entropy
        entropy_val = float(
            -np.sum(prob * np.log2(prob))
        )

    # --------------------------------------------------------------------------
    # Additional statistics
    # --------------------------------------------------------------------------
    mad_val = float(
        np.mean(np.abs(arr - mean_val))
    )

    energy_val = float(
        np.sum(arr ** 2)
    )

    rms_val = float(
        np.sqrt(np.mean(arr ** 2))
    )

    # --------------------------------------------------------------------------
    # Statistic dictionary
    # --------------------------------------------------------------------------
    stat_map = {
        "mean": mean_val,
        "median": median_val,

        "minimum": min_val,
        "maximum": max_val,
        "range": range_val,

        "variance": var_val,
        "standarddeviation": std_val,

        "p10": p10_val,
        "p25": p25_val,
        "p75": p75_val,
        "p90": p90_val,

        "iqr": iqr_val,

        "meanabsolutedeviation": mad_val,

        "skewness": skew_val,
        "kurtosis": kurt_val,

        "entropy": entropy_val,

        "energy": energy_val,
        "rootmeansquared": rms_val,
    }

    # Return only requested statistics
    return {
        k: stat_map[k]
        for k in MORPH_AGG_STAT_KEYS
    }


# ------------------------------------------------------------------------------
# Morphology features
# ------------------------------------------------------------------------------

def extract_single_cell_shape_features(
    gray_patch: np.ndarray,
    geom,
    shape_extractor,
    label: int = EXTRACTOR_DEFAULT_LABEL,
) -> Dict[str, float]:
    """
    Extract Shape2D morphology features for a single cell polygon.

    Workflow:
        1. Build local polygon mask
        2. Crop image patch around polygon
        3. Align mask to cropped image
        4. Run Shape2D PyRadiomics extraction

    Args:
        gray_patch:
            Grayscale patch image.

        geom:
            Cell polygon geometry.

        shape_extractor:
            Shape2D radiomics extractor.

        label:
            Foreground mask label.

    Returns:
        Dictionary of Shape2D features.
    """

    # Build binary mask around local polygon region
    mask_local, bbox_xyxy = build_local_polygon_mask(
        geom,
        label=label,
        margin=1,
    )

    # Crop image around polygon bounding box
    gray_crop = crop_patch_by_bbox(
        gray_patch,
        bbox_xyxy,
    )

    # Align mask to cropped image coordinates
    mask_crop = align_local_mask_to_crop(
        mask_local,
        bbox_xyxy,
        gray_patch.shape,
    )

    # --------------------------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------------------------
    if gray_crop.size == 0 or mask_crop.size == 0:
        return {}

    if gray_crop.shape != mask_crop.shape:
        return {}

    # Shape2D requires minimum foreground area
    if np.count_nonzero(mask_crop > 0) < 3:
        return {}

    # Convert numpy arrays to SimpleITK images
    image_sitk = sitk.GetImageFromArray(gray_crop)
    mask_sitk = sitk.GetImageFromArray(mask_crop)

    # Execute Shape2D feature extraction
    result = shape_extractor.execute(
        image_sitk,
        mask_sitk,
    )

    out = {}

    # Keep only Shape2D features
    for k, v in result.items():

        if "shape2d" not in str(k).lower():
            continue

        try:
            out[str(k)] = float(v)

        except Exception:
            continue

    return out


def extract_morphology_aggregates(
    gray_patch: np.ndarray,
    patch_cellseg: gpd.GeoDataFrame,
    label: int = EXTRACTOR_DEFAULT_LABEL,
    shape_extractor=None,
) -> Dict[str, float]:
    """
    Extract patch-level morphology aggregation features.

    Pipeline:
        1. Extract Shape2D features for each cell
        2. Build per-cell morphology table
        3. Aggregate each morphology feature using
           first-order statistics

    Example:
        Cell areas:
            [100, 110, 95, 130]

        Aggregated features:
            morph_area_mean
            morph_area_std
            morph_area_entropy
            ...

    Args:
        gray_patch:
            Grayscale patch image.

        patch_cellseg:
            Cell segmentation dataframe.

        label:
            Foreground label value.

        shape_extractor:
            Optional cached Shape2D extractor.

    Returns:
        Dictionary of aggregated morphology features.
    """

    out = {}

    # --------------------------------------------------------------------------
    # Handle empty segmentation safely
    # --------------------------------------------------------------------------
    if patch_cellseg is None or len(patch_cellseg) == 0:
        return out

    patch_cellseg = patch_cellseg.copy()

    patch_cellseg = patch_cellseg[
        patch_cellseg.geometry.notnull()
    ]

    if len(patch_cellseg) == 0:
        return out

    # Lazily initialize Shape2D extractor
    if shape_extractor is None:
        shape_extractor = build_shape2d_extractor(
            label=label
        )

    # --------------------------------------------------------------------------
    # Per-cell feature extraction
    # --------------------------------------------------------------------------
    per_cell_rows = []

    for _, r in patch_cellseg.iterrows():

        geom = r.geometry

        class_name = normalize_class_name(
            r.get(
                CELL_CLASS_COLUMN,
                UNKNOWN_CELL_CLASS,
            )
        )

        try:

            feats = extract_single_cell_shape_features(
                gray_patch=gray_patch,
                geom=geom,
                shape_extractor=shape_extractor,
                label=label,
            )

            # Skip invalid cells
            if not feats:
                continue

            # Store normalized cell class
            feats[CELL_CLASS_COLUMN] = class_name

            per_cell_rows.append(feats)

        except Exception:
            continue

    # No valid cells extracted
    if not per_cell_rows:
        return out

    # --------------------------------------------------------------------------
    # Build per-cell morphology dataframe
    # --------------------------------------------------------------------------
    cell_df = pd.DataFrame(per_cell_rows)

    morph_cols = [
        c
        for c in cell_df.columns
        if c != CELL_CLASS_COLUMN
    ]

    # --------------------------------------------------------------------------
    # Aggregate morphology features
    # --------------------------------------------------------------------------
    for col in morph_cols:

        vals = (
            pd.to_numeric(
                cell_df[col],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )

        # Compute first-order aggregation
        agg = _execute_firstorder_aggregation(vals)

        # Remove Shape2D prefix for cleaner naming
        base_name = strip_shape2d_prefix(col)

        # Store aggregated statistics
        for stat_name, stat_val in agg.items():

            out[
                f"morph_{base_name}_{stat_name.lower()}"
            ] = stat_val

    # --------------------------------------------------------------------------
    # Optional class-specific aggregation
    # --------------------------------------------------------------------------
    # Example:
    #     morph_neoplastic_area_mean
    #     morph_epithelial_area_std
    #
    # Currently disabled for feature dimensionality control.
    # --------------------------------------------------------------------------
    # for class_name, sub in cell_df.groupby(CELL_CLASS_COLUMN):
    #
    #     safe_class = normalize_class_name(class_name)
    #
    #     for col in morph_cols:
    #
    #         vals = (
    #             pd.to_numeric(
    #                 sub[col],
    #                 errors="coerce",
    #             )
    #             .dropna()
    #             .tolist()
    #         )
    #
    #         agg = _execute_firstorder_aggregation(vals)
    #
    #         base_name = strip_shape2d_prefix(col)
    #
    #         for stat_name, stat_val in agg.items():
    #
    #             out[
    #                 f"morph_{safe_class}_{base_name}_{stat_name.lower()}"
    #             ] = stat_val

    return out
