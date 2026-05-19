## Segment Output Format

Cell segmentation outputs are stored as patch-level parquet tables under:

```text
segment/
├── IDC/
│   ├── TENX95.parquet
│   ├── TENX99.parquet
│   └── ...
```

Each file contains cell-level segmentation results extracted from pathology image patches using CellViT-based nucleus segmentation.

---

### Parquet Schema

| Column             | Type              | Description                                                     |
| ------------------ | ----------------- | --------------------------------------------------------------- |
| `patch_idx`        | `int`             | Index of the patch within the sample                            |
| `barcode`          | `str`             | Spatial transcriptomics (ST) barcode corresponding to the patch |
| `cell_id_in_patch` | `int`             | Unique nucleus identifier inside the patch                      |
| `class_id`         | `int`             | Cell class ID predicted by CellViT                              |
| `class_name`       | `str`             | Human-readable cell type name                                   |
| `geometry`         | `polygon`         | Polygon geometry representing the segmented nucleus boundary    |
| `coord_raw`        | `tuple[int, int]` | Raw spatial coordinate of the ST spot                           |
| `n_cells`          | `int`             | Total number of segmented cells within the patch                |

---

### Notes

* Each row corresponds to a single segmented nucleus.
* `geometry` stores polygon coordinates that can be rasterized into binary masks for downstream analysis.
* `n_cells` is repeated for all rows belonging to the same patch to simplify aggregation and filtering.
* The segmentation outputs are designed for:

  * cell-aware radiomics extraction,
  * morphology analysis,
  * spatial neighborhood analysis,
  * and pathology visualization pipelines.

---

### Example Workflow

Typical downstream usage:

1. Load patch-level segmentation parquet
2. Filter nuclei by `class_name`
3. Rasterize `geometry` into ROI masks
4. Extract:

   * shape features,
   * intensity features,
   * texture features,
   * cell composition statistics

5. Align extracted features with spatial transcriptomics spots using `barcode` and `coord_raw`
