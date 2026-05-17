from __future__ import annotations

from typing import Dict

import geopandas as gpd

from hestradiomics.extractors.constants import *


# ------------------------------------------------------------------------------
# Cell-type distribution features
# ------------------------------------------------------------------------------
# This module computes simple cell composition statistics
# from cell segmentation annotations.
#
# Extracted statistics include:
#
#   1. Total cell count
#   2. Per-class cell counts
#   3. Per-class cell ratios
#
# Example output:
#
#   dist_total_count = 120
#   dist_count_neoplastic = 60
#   dist_ratio_neoplastic = 0.50
#   dist_count_inflammatory = 20
#   dist_ratio_inflammatory = 0.17
#
# These features summarize the cellular composition
# of each image patch.
#
# ------------------------------------------------------------------------------


def extract_cell_type_distribution(
    patch_cellseg: gpd.GeoDataFrame,
) -> Dict[str, float]:
    """
    Extract cell-type distribution statistics
    from a patch-level cell segmentation dataframe.

    Processing steps:
        1. Normalize cell class labels
        2. Count cells per class
        3. Compute class ratios
        4. Return standardized feature dictionary

    Supported outputs:
        - total cell count
        - per-class count
        - per-class ratio

    Args:
        patch_cellseg:
            Patch-level cell segmentation dataframe.

    Returns:
        Dictionary containing distribution statistics.
    """

    out = {}

    # --------------------------------------------------------------------------
    # Handle empty segmentation safely
    # --------------------------------------------------------------------------
    if patch_cellseg is None or len(patch_cellseg) == 0:

        # Total cell count
        out[DIST_TOTAL_COUNT_KEY] = 0.0

        # Initialize all known classes with zero counts/ratios
        for cls in KNOWN_CELL_CLASSES:

            out[f"{DIST_COUNT_PREFIX}{cls}"] = 0.0

            out[f"{DIST_RATIO_PREFIX}{cls}"] = 0.0

        return out

    # --------------------------------------------------------------------------
    # Normalize cell class labels
    # --------------------------------------------------------------------------
    # Example:
    #     "Neoplastic Cell" → "neoplastic_cell"
    # --------------------------------------------------------------------------
    patch_cellseg = patch_cellseg.copy()

    patch_cellseg[CELL_CLASS_COLUMN] = (
        patch_cellseg[CELL_CLASS_COLUMN]

        # Replace missing labels
        .fillna(UNKNOWN_CELL_CLASS)

        # Convert to string safely
        .astype(str)

        # Remove leading/trailing spaces
        .str.strip()

        # Lowercase normalization
        .str.lower()

        # Replace spaces with underscores
        .str.replace(" ", "_", regex=False)
    )

    # --------------------------------------------------------------------------
    # Compute cell counts
    # --------------------------------------------------------------------------
    counts = (
        patch_cellseg[CELL_CLASS_COLUMN]
        .value_counts()
        .to_dict()
    )

    total = int(len(patch_cellseg))

    # Store total cell count
    out[DIST_TOTAL_COUNT_KEY] = float(total)

    # --------------------------------------------------------------------------
    # Compute per-class statistics
    # --------------------------------------------------------------------------
    # Only predefined known classes are used.
    #
    # Example:
    #     dist_count_neoplastic
    #     dist_ratio_neoplastic
    # --------------------------------------------------------------------------
    class_names = list(KNOWN_CELL_CLASSES)

    for cls in class_names:

        # Retrieve count safely
        cnt = int(counts.get(cls, 0))

        # Compute ratio safely
        ratio = (
            float(cnt / total)
            if total > 0
            else 0.0
        )

        # Store count feature
        out[f"{DIST_COUNT_PREFIX}{cls}"] = float(cnt)

        # Store ratio feature
        out[f"{DIST_RATIO_PREFIX}{cls}"] = ratio

    return out

