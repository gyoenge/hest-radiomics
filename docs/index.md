# HEST Radiomics Documentation

Welcome to the documentation for **HEST-Radiomics**, a radiomics-based handcrafted feature extraction pipeline for spatial transcriptomics and whole-slide imaging (WSI) analysis.

HEST-Radiomics is built on the publicly available HEST spatial transcriptomics dataset and uses the PyRadiomics library for handcrafted feature extraction.

```{image} _static/hest_radiomics_overview.png
:width: 90%
:align: center
:alt: HEST-Radiomics overview
```

This documentation describes the basic usage of the HEST-Radiomics pipeline.

<!-- The project provides:

- HEST dataset handling utilities
- Patch-level radiomics extraction
- Cell-aware segmentation processing
- Spatial transcriptomics integration
- HDF5 / H5AD-based workflows
- Multi-processing extraction pipelines -->


---

## Contents

```{toctree}
:maxdepth: 1
:caption: Getting Started

gettingstart/installation
gettingstart/quickstart
gettingstart/stepbystep
gettingstart/configuration
```

```{toctree}
:maxdepth: 1
:caption: Output Description

outputs/overview
outputs/segment
outputs/radiomics
```

<!-- 
```{toctree}
:maxdepth: 1
:caption: API Reference

api/extractors
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Additional Resources

github
``` -->

<!-- ---

## Project Structure

```text
hestradiomics/
├── extractors/
├── segment/
├── radiomics/
├── datasets/
├── utils/
└── io/
``` -->

<!-- ---

## Main Features

### Patch-Level Radiomics

Extract handcrafted radiomics features from spatial transcriptomics patches using PyRadiomics.

### Cell-Aware Segmentation

Polygon-based cell segmentation storage with rasterized mask generation during runtime.

### HEST Dataset Integration

Built-in utilities for loading and processing HEST spatial transcriptomics datasets.

### Spatial Transcriptomics Support

Compatible with H5AD-based transcriptomics workflows and Visium-style datasets.

### Efficient Storage

Compact polygon-based HDF5 segmentation representation for large-scale WSI processing.

 -->
