from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity, wkt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


# ------------------------------------------------------------------------------
# geometry / mask helpers
# ------------------------------------------------------------------------------

def iter_polygons(geom):
    if geom is None:
        return

    if hasattr(geom, "is_empty") and geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield geom
        return

    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
        return

    if isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from iter_polygons(g)
        return


def rasterize_geometries_to_mask(
    geometries,
    image_shape: Tuple[int, int],
    label: int = 255,
) -> np.ndarray:
    """
    Rasterize shapely polygons into a uint8 mask.

    Args:
        geometries:
            Iterable of shapely geometries.

        image_shape:
            Target image shape as (H, W).

        label:
            Foreground label value.

    Returns:
        Binary/label mask as uint8 array.
    """

    h, w = image_shape

    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)

    for geom in geometries:
        for poly in iter_polygons(geom):
            ext = [
                (float(x), float(y))
                for x, y in poly.exterior.coords
            ]

            if len(ext) >= 3:
                draw.polygon(ext, fill=label)

            for interior in poly.interiors:
                hole = [
                    (float(x), float(y))
                    for x, y in interior.coords
                ]

                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)

    return np.array(canvas, dtype=np.uint8)


def build_threshold_mask(
    gray_patch: np.ndarray,
    label: int = 255,
) -> np.ndarray:
    """
    Build a simple foreground mask using grayscale intensity thresholding.
    """

    mask_patch = (
        (gray_patch > 30)
        & (gray_patch < 220)
    ).astype(np.uint8)

    return (mask_patch * label).astype(np.uint8)


def build_full_patch_mask(
    gray_patch: np.ndarray,
    label: int = 255,
) -> np.ndarray:
    """
    Build a full foreground mask covering the entire patch.
    """

    return np.full(
        gray_patch.shape,
        label,
        dtype=np.uint8,
    )


# ------------------------------------------------------------------------------
# cell segmentation loaders
# ------------------------------------------------------------------------------

def _decode_scalar(v) -> str:
    """
    Decode scalar values loaded from HDF5/string-like sources.
    """

    if isinstance(v, np.ndarray) and v.shape:
        v = v[0]

    if isinstance(v, bytes):
        return v.decode("utf-8")

    return str(v)


def _normalize_cellseg_dataframe(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Normalize cell segmentation dataframe columns.

    Required columns:
        - patch_idx
        - geometry

    Standardized columns:
        - class_name
    """

    if "patch_idx" not in gdf.columns:
        raise ValueError("cellseg dataframe must contain 'patch_idx'")

    if "geometry" not in gdf.columns:
        raise ValueError("cellseg dataframe must contain 'geometry'")

    if "class_name" not in gdf.columns:
        if "class_id" in gdf.columns:
            gdf["class_name"] = gdf["class_id"].astype(str)
        else:
            gdf["class_name"] = "unknown"

    gdf["class_name"] = (
        gdf["class_name"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return gdf


def load_cellseg_h5_as_gdf(
    cellseg_path: str | Path,
) -> gpd.GeoDataFrame:
    """
    Load CellViT segmentation results saved as HDF5.

    Expected H5 structure:
        cells/
          patch_idx
          barcode
          cell_id_in_patch
          class_id
          class_name
          geometry_wkt
    """

    rows = []

    with h5py.File(cellseg_path, "r") as f:
        if "cells" not in f:
            raise ValueError(f"Invalid cellseg h5: missing 'cells' group: {cellseg_path}")

        cells = f["cells"]

        required_keys = [
            "patch_idx",
            "barcode",
            "cell_id_in_patch",
            "class_id",
            "class_name",
            "geometry_wkt",
        ]

        for key in required_keys:
            if key not in cells:
                raise ValueError(
                    f"Invalid cellseg h5: missing cells/{key}: {cellseg_path}"
                )

        n_cells = len(cells["patch_idx"])

        for i in range(n_cells):
            geom_wkt = _decode_scalar(cells["geometry_wkt"][i])

            rows.append(
                {
                    "patch_idx": int(cells["patch_idx"][i]),
                    "barcode": _decode_scalar(cells["barcode"][i]),
                    "cell_id_in_patch": int(cells["cell_id_in_patch"][i]),
                    "class_id": int(cells["class_id"][i]),
                    "class_name": _decode_scalar(cells["class_name"][i]),
                    "geometry": wkt.loads(geom_wkt),
                }
            )

    if not rows:
        gdf = gpd.GeoDataFrame(
            {
                "patch_idx": [],
                "barcode": [],
                "cell_id_in_patch": [],
                "class_id": [],
                "class_name": [],
            },
            geometry=[],
            crs=None,
        )
    else:
        gdf = gpd.GeoDataFrame(
            rows,
            geometry="geometry",
            crs=None,
        )

    return _normalize_cellseg_dataframe(gdf)


def load_cellseg_dataframe(
    cellseg_path: Optional[str | Path],
) -> Optional[gpd.GeoDataFrame]:
    """
    Load cell segmentation dataframe from parquet or h5.

    Supported formats:
        - .parquet
        - .h5
        - .hdf5

    Returns:
        GeoDataFrame or None.
    """

    if not cellseg_path:
        return None

    cellseg_path = Path(cellseg_path)

    if not cellseg_path.exists():
        raise FileNotFoundError(f"cellseg file not found: {cellseg_path}")

    suffix = cellseg_path.suffix.lower()

    if suffix == ".parquet":
        gdf = gpd.read_parquet(cellseg_path)
        return _normalize_cellseg_dataframe(gdf)

    if suffix in {".h5", ".hdf5"}:
        return load_cellseg_h5_as_gdf(cellseg_path)

    raise ValueError(
        f"Unsupported cellseg file format: {cellseg_path}. "
        f"Expected one of: .parquet, .h5, .hdf5"
    )


# ------------------------------------------------------------------------------
# local polygon mask helpers
# ------------------------------------------------------------------------------

def build_local_polygon_mask(
    geom,
    label: int = 255,
    margin: int = 1,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Build local binary mask using polygon bounding box.

    Returns:
        mask_local:
            Local uint8 mask.

        bbox_xyxy:
            Bounding box as (min_x, min_y, max_x, max_y).
    """

    polys = list(iter_polygons(geom))

    if len(polys) == 0:
        raise ValueError("No valid polygon found")

    min_x, min_y, max_x, max_y = geom.bounds

    min_x = int(math.floor(min_x)) - margin
    min_y = int(math.floor(min_y)) - margin
    max_x = int(math.ceil(max_x)) + margin
    max_y = int(math.ceil(max_y)) + margin

    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)

    shifted_geoms = []

    for poly in polys:
        shifted_geoms.append(
            affinity.translate(
                poly,
                xoff=-min_x,
                yoff=-min_y,
            )
        )

    mask_local = rasterize_geometries_to_mask(
        shifted_geoms,
        image_shape=(height, width),
        label=label,
    )

    return mask_local, (min_x, min_y, max_x, max_y)


def crop_patch_by_bbox(
    gray_patch: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Crop grayscale patch by xyxy bounding box.
    """

    min_x, min_y, max_x, max_y = bbox_xyxy
    h, w = gray_patch.shape

    x0 = max(0, min_x)
    y0 = max(0, min_y)
    x1 = min(w - 1, max_x)
    y1 = min(h - 1, max_y)

    return gray_patch[
        y0:y1 + 1,
        x0:x1 + 1,
    ]


def align_local_mask_to_crop(
    mask_local: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    gray_patch_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Align a local polygon mask to the actual cropped patch region.

    This is needed when the polygon bounding box extends outside
    the patch boundary.
    """

    min_x, min_y, max_x, max_y = bbox_xyxy
    h, w = gray_patch_shape

    x_start = 0 if min_x >= 0 else -min_x
    y_start = 0 if min_y >= 0 else -min_y

    x_end = (
        mask_local.shape[1]
        if max_x < w
        else mask_local.shape[1] - (max_x - (w - 1))
    )

    y_end = (
        mask_local.shape[0]
        if max_y < h
        else mask_local.shape[0] - (max_y - (h - 1))
    )

    return mask_local[
        y_start:y_end,
        x_start:x_end,
    ]