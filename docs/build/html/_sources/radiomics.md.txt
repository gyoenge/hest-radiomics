# Radiomics Extraction

Radiomics features are extracted from image patches and ROI masks.

## Input

- Patch image `.h5`
- Cell segmentation `.h5`
- Optional spatial transcriptomics `.h5ad`

## Output

Radiomics features are saved under:

```text
data/hest/IDC/radiomics/
```

## Run extraction
```bash 
python src/hestradiomics/_extract.py
```

## Feature groups

The current extractor supports:

    First-order statistics
    GLCM
    GLRLM
    GLSZM
    GLDM
    NGTDM
    Shape features

