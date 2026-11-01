# 2.5D / 2D Feature Extraction with Pre-trained ResNets

This repository contains minimal scripts for extracting deep features from 3D medical
volumes (CHAOS MRI dataset) and 2D slices (OASIS dataset) using pre-trained
ImageNet ResNet backbones. Features are extracted from the **`maxpool` layer** of
the network (i.e. `conv1 → bn1 → relu → maxpool`, before `layer1`) and saved as
`.pkl` arrays ready to be used as inputs to downstream segmentation / registration
models.

Two extraction regimes are provided:

| Regime | Scripts | Input |
|---|---|---|
| **Tri-planar 2.5D** (3D volumes, sliced along one axis with 3-channel windows) | `Res18_PreNorm_Dim1.py`, `Res18_PreNorm_Dim2.py`, `Res18_PreNorm_Dim3.py` | 3D `.nii.gz` (CHAOS / L2R Task-1 MR) |
| **Single-slice 2D** (one axial slice, channels duplicated) | `ExtractFeatures_OASIS_2D_resnet18_32x32.py`, `ExtractFeatures_OASIS_2D_resnet152_32x32.py` | 2D `.nii.gz` (OASIS Affine data) |

---

## Contents

Res18_PreNorm_Dim1.py # tri-planar, slice along axis 0

Res18_PreNorm_Dim2.py # tri-planar, slice along axis 1

Res18_PreNorm_Dim3.py # tri-planar, slice along axis 2

ExtractFeatures_OASIS_2D_resnet18_32x32.py # 2D, ResNet-18

ExtractFeatures_OASIS_2D_resnet152_32x32.py # 2D, ResNet-152

## Background: the 2.5D tri-planar scheme

A 3D volume is flattened into a sequence of 2D slices by picking **groups of 3
consecutive slices** along one anatomical axis and stacking them as the RGB
channels of a 2D image. The pre-trained 2D ResNet then processes each
"3-channel pseudo-image" independently, and the resulting feature maps are
re-assembled back into a 3D tensor.

For a CHAOS volume of shape `(192, 160, 192)` the three scripts produce:

| Script | Sliced axis | Input per slice | # slices | Output tensor shape (C, D1, D2, D3) |
|---|---|---|---|---|
| `Dim1.py` | axis 0 | `(3, 160, 192)` → `(160, 192, 3)` | 64 | `(64, 64, 32, 32)` |
| `Dim2.py` | axis 1 | `(192, 3, 192)` → `(192, 192, 3)` | 54 | `(64, 32, 54, 32)` |
| `Dim3.py` | axis 2 | `(192, 160, 3)` | 64 | `(64, 32, 32, 64)` |

- **D1/D2/D3** correspond to the original volume axes.
- Each 2D feature map is `(64, 32, 32)` for ResNet-18 at input resolution
  `128 × 128` (because `maxpool` downsamples by 4: `128/4 = 32`).

For OASIS (`160 × 192` slice, single channel), channels are duplicated to make
an RGB image (`np.repeat`), and the network outputs a single `(64, 32, 32)`
feature map per subject.


## Preprocessing
For each 2D slice (or 3-slice window) the following transform is applied:

```python

T.Compose([
    T.Resize((128, 128), interpolation=T.InterpolationMode.BICUBIC),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```
# 2D single-slice extraction (OASIS)

```bash
python ExtractFeatures_OASIS_2D_resnet18_32x32.py
python ExtractFeatures_OASIS_2D_resnet152_32x32.py
```

# 3D tri-planar extraction (CHAOS)

```bash
python Res18_PreNorm_Dim1.py
python Res18_PreNorm_Dim2.py
python Res18_PreNorm_Dim3.py

```
### Switching backbones (ResNet-18 → ResNet-152)

The tri-planar scripts (`Res18_PreNorm_Dim{1,2,3}.py`) ship with a **ResNet-18**
backbone. Adapting them to **ResNet-152** is deliberately trivial — the `FeatureExtractor` wrapper, the
tri-planar slicing logic, the transforms, and the pickle I/O are all backbone
agnostic. In practice you only need to change one line:

**Swap the model constructor**:

   ```python
   # model = models.resnet18(weights='IMAGENET1K_V1').to(device).eval()
   model = models.resnet152(weights='DEFAULT').to(device).eval()
    ```