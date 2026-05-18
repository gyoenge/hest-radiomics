from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely import wkt
from torch.utils.data import Dataset

from hestradiomics.utils import ensure_uint8_rgb


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape:
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class H5PatchDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        patch_indices: Optional[List[int]] = None,
        img_key_candidates: Tuple[str, ...] = ("img", "imgs", "images"),
        barcode_key_candidates: Tuple[str, ...] = ("barcode", "barcodes"),
    ):
        self.h5_path = h5_path
        self.img_key_candidates = img_key_candidates
        self.barcode_key_candidates = barcode_key_candidates

        with h5py.File(self.h5_path, "r") as f:
            self.img_key = self._find_key(f, self.img_key_candidates)
            self.barcode_key = self._find_key(f, self.barcode_key_candidates)
            self.has_coords = "coords" in f

            n = f[self.img_key].shape[0]
            self.patch_indices = list(range(n)) if patch_indices is None else patch_indices

    @staticmethod
    def _find_key(
        f: h5py.File,
        candidates: Iterable[str],
    ) -> str:
        for key in candidates:
            if key in f:
                return key

        raise KeyError(f"None of keys found: {candidates}")

    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(
        self,
        i: int,
    ) -> Dict[str, Any]:
        patch_idx = self.patch_indices[i]

        with h5py.File(self.h5_path, "r") as f:
            image = f[self.img_key][patch_idx]
            barcode_raw = f[self.barcode_key][patch_idx]
            coord = f["coords"][patch_idx] if self.has_coords else None

        return {
            "patch_idx": int(patch_idx),
            "barcode": _decode_scalar(barcode_raw),
            "coord": None if coord is None else np.asarray(coord),
            "image": ensure_uint8_rgb(image),
        }


def collate_patches(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "patch_idx": [item["patch_idx"] for item in batch],
        "barcodes": [item["barcode"] for item in batch],
        "coords": [item["coord"] for item in batch],
    }


def save_cellseg_h5(
    rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    save_path: str,
) -> str:
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(save_path, "w") as f:
        cells_grp = f.create_group("cells")
        patches_grp = f.create_group("patches")

        if rows:
            cells_grp.create_dataset(
                "patch_idx",
                data=np.asarray([row["patch_idx"] for row in rows], dtype=np.int64),
                compression="gzip",
            )
            cells_grp.create_dataset(
                "barcode",
                data=np.asarray([row["barcode"] for row in rows], dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )
            cells_grp.create_dataset(
                "cell_id_in_patch",
                data=np.asarray([row["cell_id_in_patch"] for row in rows], dtype=np.int64),
                compression="gzip",
            )
            cells_grp.create_dataset(
                "class_id",
                data=np.asarray([row["class_id"] for row in rows], dtype=np.int64),
                compression="gzip",
            )
            cells_grp.create_dataset(
                "class_name",
                data=np.asarray([row["class_name"] for row in rows], dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )
            cells_grp.create_dataset(
                "geometry_wkt",
                data=np.asarray([row["geometry"].wkt for row in rows], dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )
        else:
            cells_grp.create_dataset("patch_idx", data=np.asarray([], dtype=np.int64))
            cells_grp.create_dataset("barcode", data=np.asarray([], dtype=object), dtype=str_dtype)
            cells_grp.create_dataset("cell_id_in_patch", data=np.asarray([], dtype=np.int64))
            cells_grp.create_dataset("class_id", data=np.asarray([], dtype=np.int64))
            cells_grp.create_dataset("class_name", data=np.asarray([], dtype=object), dtype=str_dtype)
            cells_grp.create_dataset("geometry_wkt", data=np.asarray([], dtype=object), dtype=str_dtype)

        patches_grp.create_dataset(
            "patch_idx",
            data=np.asarray([row["patch_idx"] for row in summary_rows], dtype=np.int64),
            compression="gzip",
        )
        patches_grp.create_dataset(
            "barcode",
            data=np.asarray([row["barcode"] for row in summary_rows], dtype=object),
            dtype=str_dtype,
            compression="gzip",
        )
        patches_grp.create_dataset(
            "n_cells",
            data=np.asarray([row["n_cells"] for row in summary_rows], dtype=np.int64),
            compression="gzip",
        )

        if summary_rows and "coord_raw" in summary_rows[0]:
            coord_json = [
                json.dumps(row["coord_raw"], ensure_ascii=False)
                for row in summary_rows
            ]

            patches_grp.create_dataset(
                "coord_raw_json",
                data=np.asarray(coord_json, dtype=object),
                dtype=str_dtype,
                compression="gzip",
            )

    return save_path


def load_cellseg_h5(
    seg_h5_path: str,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    with h5py.File(seg_h5_path, "r") as f:
        cells = f["cells"]
        patches = f["patches"]

        cell_rows = []

        for i in range(len(cells["patch_idx"])):
            cell_rows.append(
                {
                    "patch_idx": int(cells["patch_idx"][i]),
                    "barcode": _decode_scalar(cells["barcode"][i]),
                    "cell_id_in_patch": int(cells["cell_id_in_patch"][i]),
                    "class_id": int(cells["class_id"][i]),
                    "class_name": _decode_scalar(cells["class_name"][i]),
                    "geometry": wkt.loads(_decode_scalar(cells["geometry_wkt"][i])),
                }
            )

        patch_rows = []

        for i in range(len(patches["patch_idx"])):
            row = {
                "patch_idx": int(patches["patch_idx"][i]),
                "barcode": _decode_scalar(patches["barcode"][i]),
                "n_cells": int(patches["n_cells"][i]),
            }

            if "coord_raw_json" in patches:
                row["coord_raw"] = json.loads(
                    _decode_scalar(patches["coord_raw_json"][i])
                )

            patch_rows.append(row)

    if cell_rows:
        seg_gdf = gpd.GeoDataFrame(
            cell_rows,
            geometry="geometry",
            crs=None,
        )
    else:
        seg_gdf = gpd.GeoDataFrame(
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

    return seg_gdf, pd.DataFrame(patch_rows)
