from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hestradiomics.extractors.constants import *


# ------------------------------------------------------------------------------
# Post-processing helpers
# ------------------------------------------------------------------------------
# This module provides utilities for:
#
#   1. Identifying valid radiomics/morphology feature columns
#   2. Clipping outliers using quantiles
#   3. Z-score normalization
#   4. Min-max scaling
#   5. Building processed feature tables
#
# Typical processing pipeline:
#
#   raw features
#       ↓
#   quantile clipping
#       ↓
#   z-score normalization
#       ↓
#   min-max rescaling
#       ↓
#   processed feature dataframe
#
# ------------------------------------------------------------------------------


def is_processed_feature_column(col: str) -> bool:
    """
    Determine whether a dataframe column is a valid
    radiomics or morphology feature column.

    Supported feature groups:
        - morphology features
        - patch-level radiomics
        - cellseg-level radiomics

    Logic:
        1. Morphology columns are accepted immediately
        2. Patch/cellseg columns must additionally contain
           a valid radiomics image prefix

    Args:
        col:
            Column name.

    Returns:
        True if column should be processed.
    """

    col_lower = col.lower()

    # Morphology features are always valid
    if col_lower.startswith(MORPH_FEATURE_PREFIX):
        return True

    # Ignore non-radiomics columns
    if not (
        col_lower.startswith(PATCH_FEATURE_PREFIX)
        or col_lower.startswith(CELLSEG_FEATURE_PREFIX)
    ):
        return False

    # Extract remainder after prefix
    remainder = (
        col_lower.split("_", 1)[1]
        if "_" in col_lower
        else ""
    )

    # Check whether image filter prefix exists
    return any(
        token in remainder
        for token in RADIOMICS_IMAGE_PREFIXES
    )


def get_radiomics_feature_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    Retrieve all processable radiomics/morphology columns.

    Args:
        df:
            Input dataframe.

    Returns:
        List of feature column names.
    """

    return [
        col
        for col in df.columns
        if is_processed_feature_column(col)
    ]


def clip_feature_series(
    s: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Clip feature values using lower/upper quantiles.

    Purpose:
        Reduce the impact of extreme outliers before normalization.

    Processing:
        1. Convert values to numeric
        2. Remove NaNs for quantile computation
        3. Compute lower/upper bounds
        4. Clip values into range

    Args:
        s:
            Input feature series.

        lower_q:
            Lower quantile threshold.

        upper_q:
            Upper quantile threshold.

    Returns:
        clipped:
            Clipped feature series.

        stats:
            Dictionary containing clipping statistics.
    """

    # Convert values safely to numeric
    s_num = pd.to_numeric(
        s,
        errors="coerce",
    )

    # Remove NaNs for statistics
    valid = s_num.dropna()

    # Handle fully invalid column
    if valid.empty:
        return s_num, {
            LOWER_BOUND_COLUMN: np.nan,
            UPPER_BOUND_COLUMN: np.nan,
            MEAN_COLUMN: np.nan,
            STD_COLUMN: np.nan,
            MIN_AFTER_CLIP_COLUMN: np.nan,
            MAX_AFTER_CLIP_COLUMN: np.nan,
        }

    # Compute clipping bounds
    lower_bound = valid.quantile(lower_q)
    upper_bound = valid.quantile(upper_q)

    # Clip feature values
    clipped = s_num.clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    # Return clipped values + summary statistics
    return clipped, {
        LOWER_BOUND_COLUMN: float(lower_bound),
        UPPER_BOUND_COLUMN: float(upper_bound),

        MEAN_COLUMN:
            float(clipped.mean())
            if not pd.isna(clipped.mean())
            else np.nan,

        STD_COLUMN:
            float(clipped.std(ddof=0))
            if not pd.isna(clipped.std(ddof=0))
            else np.nan,

        MIN_AFTER_CLIP_COLUMN:
            float(clipped.min())
            if not pd.isna(clipped.min())
            else np.nan,

        MAX_AFTER_CLIP_COLUMN:
            float(clipped.max())
            if not pd.isna(clipped.max())
            else np.nan,
    }


def z_normalize_series(
    s: pd.Series,
) -> pd.Series:
    """
    Apply z-score normalization.

    Formula:
        z = (x - mean) / std

    If standard deviation is zero,
    returns a zero-filled series.

    Args:
        s:
            Input feature series.

    Returns:
        Z-normalized series.
    """

    s_num = pd.to_numeric(
        s,
        errors="coerce",
    )

    mean = s_num.mean()
    std = s_num.std(ddof=0)

    # Handle constant features safely
    if pd.isna(std) or std == 0:
        return pd.Series(
            np.zeros(len(s_num)),
            index=s_num.index,
            dtype=float,
        )

    return (s_num - mean) / std


def minmax_rescale_series(
    s: pd.Series,
) -> pd.Series:
    """
    Apply min-max scaling into [0, 1].

    Formula:
        scaled = (x - min) / (max - min)

    If feature is constant,
    returns a zero-filled series.

    Args:
        s:
            Input feature series.

    Returns:
        Min-max scaled series.
    """

    s_num = pd.to_numeric(
        s,
        errors="coerce",
    )

    min_val = s_num.min()
    max_val = s_num.max()

    # Handle constant features safely
    if (
        pd.isna(min_val)
        or pd.isna(max_val)
        or max_val == min_val
    ):
        return pd.Series(
            np.zeros(len(s_num)),
            index=s_num.index,
            dtype=float,
        )

    return (s_num - min_val) / (max_val - min_val)


def build_processed_feature_df(
    df: pd.DataFrame,
    status_col: str = STATUS_COLUMN,
    ok_status: str = STATUS_OK,
    lower_q: float = DEFAULT_CLIP_LOWER_Q,
    upper_q: float = DEFAULT_CLIP_UPPER_Q,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a fully processed radiomics feature dataframe.

    Processing pipeline for each feature:
        raw feature
            ↓
        quantile clipping
            ↓
        z-score normalization
            ↓
        min-max scaling

    Only rows with status == ok_status are processed.
    Invalid rows are filled with NaN.

    Args:
        df:
            Raw feature dataframe.

        status_col:
            Status column name.

        ok_status:
            Status value considered valid.

        lower_q:
            Lower clipping quantile.

        upper_q:
            Upper clipping quantile.

    Returns:
        processed_df:
            Processed feature dataframe.

        stats_df:
            Per-feature preprocessing statistics.
    """

    # Copy original dataframe
    processed_df = df.copy()

    # Detect processable feature columns
    feature_cols = get_radiomics_feature_columns(df)

    # Fail if no features were detected
    if not feature_cols:

        sample_cols = df.columns[:30].tolist()

        raise ValueError(
            f"No radiomics/morphology feature columns "
            f"found to process. "
            f"Sample columns: {sample_cols}"
        )

    stats_rows = []

    # Select only valid rows
    ok_mask = processed_df[status_col] == ok_status

    # --------------------------------------------------------------------------
    # Process each feature independently
    # --------------------------------------------------------------------------
    for col in feature_cols:

        # Convert feature values safely to numeric
        original_series = pd.to_numeric(
            processed_df.loc[ok_mask, col],
            errors="coerce",
        )

        # --------------------------------------------------------------
        # Quantile clipping
        # --------------------------------------------------------------
        clipped, clip_stats = clip_feature_series(
            original_series,
            lower_q=lower_q,
            upper_q=upper_q,
        )

        # --------------------------------------------------------------
        # Z-score normalization
        # --------------------------------------------------------------
        z_norm = z_normalize_series(clipped)

        # --------------------------------------------------------------
        # Min-max scaling
        # --------------------------------------------------------------
        scaled = minmax_rescale_series(z_norm)

        # Store processed values
        processed_df.loc[ok_mask, col] = (
            scaled.astype(float)
        )

        # Invalid rows become NaN
        processed_df.loc[~ok_mask, col] = np.nan

        # --------------------------------------------------------------
        # Store preprocessing statistics
        # --------------------------------------------------------------
        stats_rows.append(
            {
                FEATURE_COLUMN: col,

                LOWER_Q_COLUMN: lower_q,
                UPPER_Q_COLUMN: upper_q,

                **clip_stats,

                Z_MEAN_COLUMN:
                    float(z_norm.mean())
                    if len(z_norm.dropna())
                    else np.nan,

                Z_STD_COLUMN:
                    float(z_norm.std(ddof=0))
                    if len(z_norm.dropna())
                    else np.nan,

                SCALED_MIN_COLUMN:
                    float(scaled.min())
                    if len(scaled.dropna())
                    else np.nan,

                SCALED_MAX_COLUMN:
                    float(scaled.max())
                    if len(scaled.dropna())
                    else np.nan,
            }
        )

    # Build statistics dataframe
    stats_df = pd.DataFrame(stats_rows)

    return processed_df, stats_df

