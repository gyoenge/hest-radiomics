from __future__ import annotations

import inspect
import os
from typing import Any, List, Optional
import warnings

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


MODEL_SRC_MAP = {
    "CellViT-256-x20.pth": "1w99U4sxDQgOSuiHMyvS_NYBiz6ozolN2",
    "CellViT-256-x40.pth": "1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q",
    "CellViT-SAM-H-x40.pth": "1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV",
    "CellViT-SAM-H-x20.pth": "1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30",
}


def infer_cellvit_model_type(
    model_name: str,
) -> str:
    name = model_name.upper()

    if "SAM" in name:
        return "SAM"
    if "HIPT" in name:
        return "HIPT"
    if "CELLVIT" in name or "256" in name:
        return "CellViT"

    raise ValueError(f"Cannot infer CellViT model type from {model_name}")


def resolve_device(device: str = "cuda:0") -> str:
    if device is None:
        device = "cuda:0"

    device = str(device)

    if device == "cuda":
        device = "cuda:0"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        try:
            gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        except ValueError:
            gpu_id = 0

        if gpu_id >= torch.cuda.device_count():
            print(
                f"[WARN] Requested {device}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) available. "
                "Falling back to cuda:0."
            )
            return "cuda:0"

        return f"cuda:{gpu_id}"

    return device


def verify_or_download_model(
    model_path: str,
    model_name: str,
) -> None:
    if os.path.exists(model_path):
        print(f"[INFO] Found model at {model_path}")
        return

    if model_name not in MODEL_SRC_MAP:
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Known models: {list(MODEL_SRC_MAP.keys())}"
        )

    try:
        import gdown
    except Exception as e:
        raise ImportError("gdown is required to auto-download model weights.") from e

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    gdown.download(
        id=MODEL_SRC_MAP[model_name],
        output=model_path,
        quiet=False,
    )


def polygon_from_mask(
    mask: np.ndarray,
) -> List[Polygon]:
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: List[Polygon] = []

    for contour in contours:
        if len(contour) < 3:
            continue

        coords = contour[:, 0, :]
        poly = Polygon(coords)

        if poly.is_valid and poly.area > 0:
            polygons.append(poly)

    return polygons


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


class CellViTInferenceAdapter:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        output_dir: str,
        runtime_dir: Optional[str] = None,
        device: str = "cuda:0",
    ):
        self._patch_ray_init()

        from cellvit.detect_cells import CellViTInference, SystemConfiguration

        self.model_path = model_path
        self.model_name = infer_cellvit_model_type(model_name)
        self.output_dir = output_dir
        self.device = resolve_device(device)

        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )

        self._tmp_root = runtime_dir or os.path.join(
            output_dir,
            "_cellvit_runtime",
        )

        os.makedirs(self._tmp_root, exist_ok=True)

        self.runner = self._build_runner(
            CellViTInference,
            SystemConfiguration,
        )

    def _patch_ray_init(self) -> None:
        import ray

        if getattr(ray, "_patched_by_h5radiomics", False):
            return

        original_init = ray.init

        def patched_init(*args, **kwargs):
            kwargs.setdefault("ignore_reinit_error", True)
            kwargs.setdefault("include_dashboard", False)
            kwargs.setdefault("_memory", 32 * 1024**3)
            kwargs.setdefault("object_store_memory", 8 * 1024**3)

            return original_init(*args, **kwargs)

        ray.init = patched_init
        ray._patched_by_h5radiomics = True

    def _build_system_configuration(
        self,
        SystemConfiguration,
    ):
        sig = inspect.signature(SystemConfiguration)
        kwargs = {}

        for name in sig.parameters:
            if name == "device":
                kwargs[name] = self.device
            elif name in ("gpu", "gpu_id", "gpu_ids"):
                gpu_id = (
                    int(self.device.split(":")[-1])
                    if self.device.startswith("cuda:")
                    else 0
                )
                kwargs[name] = [gpu_id] if name == "gpu_ids" else gpu_id
            elif name in ("mixed_precision", "amp", "enforce_mixed_precision"):
                kwargs[name] = False
            elif name == "batch_size":
                kwargs[name] = 1
            elif name == "num_workers":
                kwargs[name] = 0
            elif name == "seed":
                kwargs[name] = 42
            elif name == "verbose":
                kwargs[name] = False

        try:
            return SystemConfiguration(**kwargs)
        except Exception:
            return SystemConfiguration()

    def _build_runner(
        self,
        CellViTInference,
        SystemConfiguration,
    ):
        system_configuration = self._build_system_configuration(
            SystemConfiguration
        )

        sig = inspect.signature(CellViTInference)
        kwargs = {}

        for name in sig.parameters:
            if name == "model_path":
                kwargs[name] = self.model_path
            elif name == "model_name":
                kwargs[name] = self.model_name
            elif name == "outdir":
                kwargs[name] = self._tmp_root
            elif name == "system_configuration":
                kwargs[name] = system_configuration
            elif name == "device":
                kwargs[name] = self.device

        return CellViTInference(**kwargs)

    def _try_call_method(
        self,
        method_names: List[str],
        *args,
        **kwargs,
    ):
        last_error = None

        for name in method_names:
            fn = getattr(self.runner, name, None)

            if fn is None:
                continue

            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e

        if last_error is not None:
            raise last_error

        return None

    def _predict_batch_raw(
        self,
        images: List[np.ndarray],
    ):
        result = self._try_call_method(
            ["predict_batch", "process_batch", "infer_batch", "run_batch"],
            images,
        )

        if result is not None:
            return result

        pil_images = [
            Image.fromarray(img)
            for img in images
        ]

        return self._try_call_method(
            ["predict_batch", "process_batch", "infer_batch", "run_batch"],
            pil_images,
        )

    def _predict_single_raw(
        self,
        image: np.ndarray,
    ):
        import cv2

        pil = Image.fromarray(image)
        x = self.runner.inference_transforms(pil).unsqueeze(0)
        x = x.to(torch.device(self.device))

        with torch.no_grad():
            out = self.runner.model(x)

        out = self.runner.apply_softmax_reorder(out)
        type_map = self._extract_type_map_from_output(out)

        if isinstance(out, dict) and "instance_map" in out:
            inst_map = out["instance_map"][0]

            if torch.is_tensor(inst_map):
                inst_map = inst_map.detach().cpu().numpy()

            return {
                "instance_map": np.asarray(inst_map, dtype=np.int32),
                "type_map": type_map,
            }

        if isinstance(out, dict) and "nuclei_binary_map" in out:
            bin_map = out["nuclei_binary_map"][0]
            arr = (
                bin_map.detach().cpu().numpy()
                if torch.is_tensor(bin_map)
                else np.asarray(bin_map)
            )

            if arr.ndim == 3:
                if arr.shape[0] in (2, 3, 4):
                    pred = np.argmax(arr, axis=0)
                elif arr.shape[-1] in (2, 3, 4):
                    pred = np.argmax(arr, axis=-1)
                else:
                    raise RuntimeError(f"Unexpected nuclei_binary_map shape: {arr.shape}")
            elif arr.ndim == 2:
                pred = arr
            else:
                raise RuntimeError(f"Unexpected nuclei_binary_map ndim: {arr.ndim}")

            pred = np.squeeze(np.asarray(pred))

            if pred.ndim != 2:
                raise RuntimeError(f"pred must be 2D, got shape={pred.shape}")

            foreground = np.ascontiguousarray((pred > 0).astype(np.uint8))
            _, labels = cv2.connectedComponents(foreground)

            return {
                "instance_map": np.asarray(labels, dtype=np.int32),
                "type_map": type_map,
            }

        raise RuntimeError(
            f"Unknown output format: "
            f"{list(out.keys()) if isinstance(out, dict) else type(out)}"
        )

    def _extract_gdf_from_any(
        self,
        raw: Any,
    ) -> Optional[gpd.GeoDataFrame]:
        if raw is None:
            return None

        if isinstance(raw, gpd.GeoDataFrame):
            return raw

        if (
            isinstance(raw, str)
            and raw.endswith((".geojson", ".json"))
            and os.path.exists(raw)
        ):
            return gpd.read_file(raw)

        if isinstance(raw, dict):
            for key in ["gdf", "geojson_gdf", "cells_gdf"]:
                if key in raw and isinstance(raw[key], gpd.GeoDataFrame):
                    return raw[key]

            for key in [
                "geojson_path",
                "cells_geojson",
                "cells_path",
                "output_geojson",
            ]:
                if key in raw and isinstance(raw[key], str) and os.path.exists(raw[key]):
                    return gpd.read_file(raw[key])

            for key in ["geometry", "geometries"]:
                if key in raw and isinstance(raw[key], list):
                    return gpd.GeoDataFrame(
                        {"cell_id_in_patch": list(range(len(raw[key])))},
                        geometry=raw[key],
                        crs=None,
                    )

        return None

    def _extract_instance_map_from_any(
        self,
        raw: Any,
    ) -> Optional[np.ndarray]:
        if raw is None:
            return None

        if torch.is_tensor(raw):
            raw = raw.detach().cpu().numpy()

        if isinstance(raw, np.ndarray):
            raw = np.asarray(raw)

            if raw.ndim == 2:
                return raw.astype(np.int32)

            if raw.ndim == 3 and 1 in raw.shape:
                raw = np.squeeze(raw)

                if raw.ndim == 2:
                    return raw.astype(np.int32)

        if isinstance(raw, dict):
            for key in [
                "instance_map",
                "instance_maps",
                "inst_map",
                "inst_maps",
                "nuclei_instance_map",
                "nuclei_map",
            ]:
                if key not in raw:
                    continue

                value = raw[key]

                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()

                value = np.asarray(value)

                if value.ndim == 2:
                    return value.astype(np.int32)
                if value.ndim == 3 and value.shape[0] == 1:
                    return value[0].astype(np.int32)
                if value.ndim == 3 and value.shape[-1] == 1:
                    return value[..., 0].astype(np.int32)
                if value.ndim == 3:
                    return value.astype(np.int32)

        return None

    def _extract_type_map_from_output(
        self,
        out: Any,
    ) -> Optional[np.ndarray]:
        if not isinstance(out, dict) or "nuclei_type_map" not in out:
            return None

        type_map = out["nuclei_type_map"][0]

        if torch.is_tensor(type_map):
            type_map = type_map.detach().cpu().numpy()
        else:
            type_map = np.asarray(type_map)

        if type_map.ndim == 3:
            if type_map.shape[0] <= 10:
                type_map = np.argmax(type_map, axis=0)
            elif type_map.shape[-1] <= 10:
                type_map = np.argmax(type_map, axis=-1)
            else:
                raise RuntimeError(f"Unexpected nuclei_type_map shape: {type_map.shape}")
        elif type_map.ndim != 2:
            raise RuntimeError(f"Unexpected nuclei_type_map ndim: {type_map.ndim}")

        return type_map.astype(np.int32)

    def _instance_majority_type(
        self,
        inst_mask: np.ndarray,
        type_map: Optional[np.ndarray],
    ) -> int:
        if type_map is None:
            return -1

        values = type_map[inst_mask > 0]
        values = values[values > 0]

        if values.size == 0:
            return -1

        counts = np.bincount(values.astype(np.int32))

        return int(np.argmax(counts))

    def _raw_to_gdf(
        self,
        raw: Any,
    ) -> gpd.GeoDataFrame:
        gdf = self._extract_gdf_from_any(raw)

        if gdf is not None:
            if "geometry" not in gdf.columns:
                gdf = gpd.GeoDataFrame(
                    gdf,
                    geometry="geometry",
                    crs=None,
                )

            return gdf

        if isinstance(raw, dict):
            inst_map = self._extract_instance_map_from_any(raw)
            type_map = raw.get("type_map", None)
        else:
            inst_map = self._extract_instance_map_from_any(raw)
            type_map = None

        if inst_map is None:
            raise RuntimeError(f"Unsupported CellViT output type: {type(raw)}")

        class_name_map = {
            0: "background",
            1: "neoplastic",
            2: "inflammatory",
            3: "connective",
            4: "dead",
            5: "epithelial",
        }

        rows = []
        instance_ids = np.unique(inst_map)
        instance_ids = instance_ids[instance_ids > 0]

        for inst_id in instance_ids:
            mask = inst_map == inst_id
            class_id = self._instance_majority_type(mask.astype(np.uint8), type_map)
            class_name = class_name_map.get(class_id, f"class_{class_id}")

            for poly in polygon_from_mask(mask):
                rows.append(
                    {
                        "cell_id_in_patch": int(inst_id),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "geometry": poly,
                    }
                )

        if rows:
            return gpd.GeoDataFrame(
                rows,
                geometry="geometry",
                crs=None,
            )

        return gpd.GeoDataFrame(
            {
                "cell_id_in_patch": [],
                "class_id": [],
                "class_name": [],
            },
            geometry=[],
            crs=None,
        )

    def predict_batch_to_gdfs(
        self,
        images: List[np.ndarray],
    ) -> List[gpd.GeoDataFrame]:
        raw_batch = self._predict_batch_raw(images)

        if raw_batch is not None:
            if isinstance(raw_batch, list):
                return [self._raw_to_gdf(item) for item in raw_batch]

            if isinstance(raw_batch, tuple):
                return [self._raw_to_gdf(item) for item in list(raw_batch)]

            if isinstance(raw_batch, dict):
                inst = self._extract_instance_map_from_any(raw_batch)

                if inst is not None and inst.ndim == 3:
                    return [self._raw_to_gdf(mask) for mask in inst]

        return [
            self._raw_to_gdf(self._predict_single_raw(image))
            for image in images
        ]
