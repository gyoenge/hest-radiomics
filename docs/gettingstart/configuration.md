# Configuration

HEST-Radiomics uses a centralized dataclass-based configuration system.

The main configuration object is defined as:

```python
CONFIG = PipelineConfig()
```

This object contains all settings required for dataset download, cell segmentation, radiomics extraction, and statistics generation.

---

## Overview

The configuration is organized into several sub-configurations:

```text
PipelineConfig
├── RunConfig
├── DownloadConfig
├── CellSegmentConfig
├── RadiomicsConfig
└── StatisticsConfig
```

Each configuration class controls a specific part of the pipeline.

---

## RunConfig

`RunConfig` controls which pipeline steps are executed.

```python
@dataclass(frozen=True)
class RunConfig:
    run_hest_download: bool = True
    run_segment: bool = True
    run_overlay: bool = True
    run_radiomics_extraction: bool = True
    run_statistics: bool = True

    sample_ids: Optional[tuple[str, ...]] = None
```

### Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `run_hest_download` | `bool` | `True` | Whether to download the HEST base dataset |
| `run_segment` | `bool` | `True` | Whether to run cell segmentation |
| `run_overlay` | `bool` | `True` | Whether to generate segmentation overlay visualizations |
| `run_radiomics_extraction` | `bool` | `True` | Whether to extract radiomics features |
| `run_statistics` | `bool` | `True` | Whether to generate statistics and visualization outputs |
| `sample_ids` | `Optional[tuple[str, ...]]` | `None` | Target sample IDs to process |

If `sample_ids` is `None`, all available samples are processed.

Example:

```python
sample_ids = None
```

To process only selected samples:

```python
sample_ids = ("NCBI681", "NCBI682")
```

---

## DownloadConfig

`DownloadConfig` controls HEST dataset download settings.

```python
@dataclass(frozen=True)
class DownloadConfig:
    root: Path = Path("./data").resolve()
    subroot: str = "hest"

    oncotrees: tuple[str] = field(default_factory=lambda: [
        "IDC",
    ])

    required_dirs: tuple[str] = field(default_factory=lambda: [
        "patches",
        "st",
    ])

    optional_dirs: tuple[str] = field(default_factory=lambda: [
        "metadata",
        "patches_vis",
        "thumbnails",
        "spatial_plots",
    ])

    technologies: tuple[str] = field(default_factory=lambda: [
        "Xenium",
        "Visium",
    ])
```

### Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `root` | `Path` | `./data` | Root directory for downloaded data |
| `subroot` | `str` | `hest` | Subdirectory name under the root path |
| `oncotrees` | `tuple[str]` | `("IDC",)` | Target cancer types or OncoTree codes |
| `required_dirs` | `tuple[str]` | `("patches", "st")` | Required dataset directories |
| `optional_dirs` | `tuple[str]` | metadata-related dirs | Optional dataset directories |
| `technologies` | `tuple[str]` | `("Xenium", "Visium")` | Target spatial transcriptomics technologies |

The final download directory is defined by:

```python
download_dir = root / subroot
```

Example:

```text
data/hest/
```

---

## CellSegmentConfig

`CellSegmentConfig` controls cell segmentation and overlay generation.

```python
@dataclass(frozen=True)
class CellSegmentConfig:
    model_name: str = "CellViT-SAM-H-x20.pth"
    model_root: Path = Path("./models")

    device: str = "cuda:0"
    batch_size: int = 8
    num_workers: int = 0

    use_class_color: bool = True

    overwrite_segment: bool = False
    overwrite_overlay: bool = False
```

### Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `model_name` | `str` | `CellViT-SAM-H-x20.pth` | CellViT model checkpoint filename |
| `model_root` | `Path` | `./models` | Directory containing model checkpoints |
| `device` | `str` | `cuda:0` | Device used for segmentation |
| `batch_size` | `int` | `8` | Segmentation batch size |
| `num_workers` | `int` | `0` | Number of dataloader workers |
| `use_class_color` | `bool` | `True` | Whether to use class-specific colors in overlays |
| `overwrite_segment` | `bool` | `False` | Whether to overwrite existing segmentation files |
| `overwrite_overlay` | `bool` | `False` | Whether to overwrite existing overlay files |

The model path is defined as:

```python
model_path = model_root / model_name
```

Example:

```text
models/CellViT-SAM-H-x20.pth
```

---

## RadiomicsConfig

`RadiomicsConfig` controls radiomics feature extraction.

```python
@dataclass(frozen=True)
class RadiomicsConfig:
    mask_source: str = "cellseg"

    output_dirname: str = "radiomics"
    segment_dirname: str = "segment"
    patch_dirname: str = "patches"

    num_workers: int = 16

    overwrite: bool = False
    save_patches: bool = False
```

### Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `mask_source` | `str` | `cellseg` | Source used to generate extraction masks |
| `output_dirname` | `str` | `radiomics` | Output directory for radiomics features |
| `segment_dirname` | `str` | `segment` | Directory containing segmentation files |
| `patch_dirname` | `str` | `patches` | Directory containing patch image files |
| `num_workers` | `int` | `16` | Number of workers for parallel extraction |
| `overwrite` | `bool` | `False` | Whether to overwrite existing radiomics outputs |
| `save_patches` | `bool` | `False` | Whether to save intermediate patch images |

### Mask Source

`mask_source` determines how masks are generated during radiomics extraction.

Supported values:

| Value | Description |
|---|---|
| `threshold` | Generate masks from image thresholding |
| `cellseg` | Generate masks from cell segmentation polygons |

Use `cellseg` when segmentation files are available and cell-aware extraction is required.

---

## StatisticsConfig

`StatisticsConfig` controls statistics and visualization outputs.

```python
@dataclass(frozen=True)
class StatisticsConfig:
    output_dirname: str = "statistics"
    overwrite: bool = True
```

### Options

| Option | Type | Default | Description |
|---|---:|---:|---|
| `output_dirname` | `str` | `statistics` | Output directory for statistics results |
| `overwrite` | `bool` | `True` | Whether to overwrite existing statistics outputs |

---

## PipelineConfig

`PipelineConfig` combines all sub-configurations into a single configuration object.

```python
@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    cellseg: CellSegmentConfig = field(default_factory=CellSegmentConfig)
    radiomics: RadiomicsConfig = field(default_factory=RadiomicsConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    @property
    def sample_ids(self) -> Optional[tuple[str, ...]]:
        return self.run.sample_ids
```

The global configuration object is:

```python
CONFIG = PipelineConfig()
```

---

## Example: Full Pipeline

The default configuration runs the full pipeline:

```python
CONFIG = PipelineConfig()
```

Equivalent behavior:

```python
RunConfig(
    run_hest_download=True,
    run_segment=True,
    run_overlay=True,
    run_radiomics_extraction=True,
    run_statistics=True,
    sample_ids=None,
)
```

This processes all available samples.

---

## Example: Run Selected Samples Only

To process only selected samples:

```python
@dataclass(frozen=True)
class RunConfig:
    sample_ids: Optional[tuple[str, ...]] = ("NCBI681", "NCBI682")
```

This limits the pipeline to:

```text
NCBI681
NCBI682
```

---

## Example: Skip Dataset Download

If the dataset has already been downloaded:

```python
@dataclass(frozen=True)
class RunConfig:
    run_hest_download: bool = False
    run_segment: bool = True
    run_overlay: bool = True
    run_radiomics_extraction: bool = True
    run_statistics: bool = True
```

---

## Example: Run Only Radiomics Extraction

To run only radiomics extraction:

```python
@dataclass(frozen=True)
class RunConfig:
    run_hest_download: bool = False
    run_segment: bool = False
    run_overlay: bool = False
    run_radiomics_extraction: bool = True
    run_statistics: bool = False
```

This assumes that patch files and segmentation files already exist.

---

## Example: CPU Execution

To run segmentation on CPU:

```python
@dataclass(frozen=True)
class CellSegmentConfig:
    device: str = "cpu"
```

```{note}
CPU execution may be significantly slower than GPU execution.
```

---

## Example: Change Target Cancer Types

To download and process multiple OncoTree groups:

```python
@dataclass(frozen=True)
class DownloadConfig:
    oncotrees: tuple[str] = field(default_factory=lambda: [
        "IDC",
        "SKCM",
        "LUAD",
        "PAAD",
        "COAD",
    ])
```

---

## Notes

- The configuration classes are frozen dataclasses.
- To change a value, edit the configuration source file before running the pipeline.
- `sample_ids=None` means all samples are processed.
- Existing outputs are skipped unless the relevant `overwrite` option is set to `True`.
- Segmentation-based radiomics extraction requires valid files under the `segment/` directory.
- The default configuration is intended for running the full pipeline on IDC samples.

