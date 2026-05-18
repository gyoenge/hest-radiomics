# Dataset

The project expects the HEST dataset to follow this structure:

```text
data/
└── hest/
    └── IDC/
        ├── patches/
        ├── st/
        ├── cellseg/
        └── radiomics/
```

### Required inputs
- Patches

    Patch images are stored as .h5 files.

    ```text
    data/hest/IDC/patches/{sample_id}.h5
    ```

- Spatial transcriptomics

    ST data are stored as .h5ad files.

    ```text
    data/hest/IDC/st/{sample_id}.h5ad
    ```

- Cell segmentation

    Cell segmentation results are stored as .h5 files.

    ```text
    data/hest/IDC/cellseg/{sample_id}.h5
    ```

