# Installation

## 1. Create conda environment

Firstly, create and activate the conda environment:

```bash
conda create -n hestradiomics python=3.10
conda activate hestradiomics
```

> `cellvit` currently requires Python >= 3.10.

---

## 2. Install PyTorch

Install `torch` and `torchvision` matching your CUDA environment.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install compatible NumPy version:

```bash
pip install "numpy<2.0.0,>=1.24"
```

> Some dependencies are not yet fully compatible with NumPy 2.x.

---

## 3. Install hest-radiomics

Inside the project root:

```bash
git clone https://github.com/gyoenge/hest-radiomics.git
cd hest-radiomics/
pip install -e . --no-build-isolation
```

---

## 4. Install CellViT dependencies 

The `segment` engine additionally requires `cellvit`.

### Install OpenSlide

```bash
conda install -c conda-forge openslide
pip install openslide-python openslide-bin
```

### Install CellViT

```bash
pip install cellvit
```

---

## 5. Verify installation

```bash
python -c "import hestradiomics; print('ok')"
```

---

## 6. Recommended environment

- Python 3.10
- CUDA 12.1
- PyTorch >= 2.1
- Linux (Ubuntu / Rocky Linux / RHEL recommended)

---

## 7. Troubleshooting

### CUDA mismatch

If you encounter:

```text
The detected CUDA version mismatches the version that was used to compile PyTorch
```

Make sure:
- CUDA runtime matches the installed PyTorch build
- `torch.version.cuda` matches your environment

Check:

```bash
python -c "import torch; print(torch.version.cuda)"
```

### OpenSlide import error

If:

```text
ImportError: libopenslide.so
```

Reinstall:

```bash
conda install -c conda-forge openslide
```

### NumPy compatibility issues

If packages fail with NumPy 2.x:

```bash
pip install "numpy<2.0.0"
```

