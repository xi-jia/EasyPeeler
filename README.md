# Easy Peeler: Towards Efficient Modular Medical Image Registration


This repository is intended to provide the essential implementation of Easy Peeler and resources for reproducing the main experiments reported in the paper. 


## Overview

**Easy Peeler** is a modular framework for medical image registration that decouples visual representation learning from deformation estimation. Unlike conventional learning-based registration approaches that jointly optimize image representations and registration networks, Easy Peeler leverages **frozen pretrained visual encoders** together with a **lightweight trainable transition–decoder** to estimate spatial transformations between medical images.

Beyond introducing an efficient registration framework, our work systematically investigates the suitability of pretrained visual representations for dense spatial correspondence. We evaluate **47 feature representations extracted from 14 pretrained visual encoders across 10 biomedical imaging datasets**, covering diverse imaging modalities, anatomical structures, and registration settings.

Our findings **challenge several common assumptions** about the relationship between foundation-model capabilities and medical image registration performance.

## Framework

Easy Peeler consists of two principal components:

### 1. Frozen Visual Encoder

A pretrained visual encoder extracts feature representations from the moving and fixed images. All encoder parameters remain frozen throughout registration training. (All image features can be pre-extracted.)

### 2. Trainable Transition–Decoder

A lightweight transition–decoder processes the extracted moving and fixed image features to estimate the spatial deformation required for image registration.

By separating feature extraction from deformation estimation, Easy Peeler provides a flexible framework for investigating how different pretrained representations contribute to registration accuracy and computational efficiency.

We note that **both the 2D and 3D network architecture of Easy Peeler are provided in `./Architecture/`.**


## Evaluated Pretrained Visual Encoders

We investigate **14 pretrained encoder architectures**, producing a total of **47 candidate feature representations**.

| Encoder family | Evaluated variants | Number of feature representations |
|---|---|---:|
| ResNet | ResNet18, ResNet152 | 2 |
| FCN | FCN-ResNet101 | 5 |
| DeepLabv3 | DeepLabv3-ResNet101 | 4 |
| DINOv2 | Small, Base | 10 |
| DINOv3 | Small, Base | 10 |
| SAM | ViT-Base | 6 |
| MedSAM | ViT-Base | 6 |
| SAM2 | Tiny, Small, Base Plus, Large | 4 |
| **Total** | **14 encoders** | **47** |

The evaluated encoders span convolutional neural networks, vision transformers, self-supervised visual representations, and segmentation-oriented foundation models.

We consider both general-purpose and medically specialized pretrained models to investigate the influence of architecture, representation depth, model scale, and pretraining domain on registration performance.

### Feature Extraction

Easy Peeler builds upon publicly released pretrained visual encoders, including ResNet, FCN, DeepLabv3, DINOv2, DINOv3, SAM, MedSAM, and SAM2.

Input images are resized according to the requirements of each pretrained encoder. Intermediate features are extracted and reshaped to a common spatial resolution of **32 × 32**, while preserving the architecture-dependent channel dimensionality.

For encoders that support intermediate feature extraction, we investigate representations from different network depths to assess how feature hierarchy influences registration performance.

All pretrained encoders remain frozen, and no additional encoder training or fine-tuning is performed.

**An example feature extraction code using ResNet-18 and ResNet-152 in both 2D and 3D are provided in `./FeatureExtraction/`.**

---
### Volumetric Feature Extraction

For three-dimensional registration, Easy Peeler adopts a **tri-planar 2.5D feature-extraction scheme**.

Features are extracted from non-overlapping three-slice blocks along the sagittal, coronal, or axial orientation. When the number of slices is not divisible by three, the final block is constructed using the last three slices.

This approach allows pretrained 2D encoders to be applied to volumetric biomedical images without requiring volumetric encoder retraining.

## Experimental Evaluation

We evaluate Easy Peeler across **10 biomedical image registration datasets**, including **seven three-dimensional registration benchmarks**.

Our experiments investigate:

- **Registration accuracy:** Comparison with established learning-based registration baselines.
- **Computational efficiency:** Reduction in task-specific training computation through frozen feature extraction.
- **Representation suitability:** Systematic evaluation of 47 pretrained feature representations.
- **Model architecture and scale:** Influence of encoder families, model sizes, and generations.
- **Pretraining domain:** Comparison between general-purpose and medical-domain pretrained representations.
- **Feature hierarchy:** Influence of intermediate encoder representations on registration performance.
- **Volumetric orientation:** Effect of sagittal, coronal, and axial feature extraction.
- **Feature information:** Relationship between image–feature mutual information and registration accuracy.

Together, these experiments provide a systematic investigation of how pretrained visual representations can support efficient and accurate medical image registration.
## Datasets and Training Configurations

### Datasets

We evaluated **Easy Peeler** on ten deformable medical and biomedical image registration tasks constructed from publicly available datasets. These benchmarks cover diverse anatomical structures, imaging modalities, and registration scenarios, including **intra- and inter-subject registration, mono- and cross-modality registration, and both 2D and 3D imaging**.

| Dataset / Task | Imaging modality | Anatomical region / Application |
|---|---|---|
| OASIS-1 | T1-weighted MRI | Brain |
| ACDC | Cine MRI | Cardiac |
| Abdomen-CT | CT | Abdomen |
| BCV-CT | CT | Abdomen |
| CHAOS-MR | T2-SPIR MRI | Abdomen |
| MindBoggle-101 | T1-weighted MRI | Brain |
| Lung-CT | CT | Lung |
| BCV–CHAOS | CT–MRI | Cross-modality abdominal registration |
| Abdomen-CT–MR | CT–MRI | Cross-modality abdominal registration |
| Fluo-N2DH-GOWT1 | Fluorescence microscopy | GFP–GOWT1 mouse embryonic stem cells |

For population-level registration experiments, datasets were partitioned into **training, validation, and test sets before constructing registration pairs**, ensuring separation between the corresponding data subsets.

The **Abdomen-CT–MR** task was evaluated using intra-subject cross-modality registration through instance-level optimization. The registration model was initialized with weights learned from the inter-subject **BCV–CHAOS** task.

### Dataset-Specific Training Configurations

The following table summarizes the training configurations used in our experiments. All models employed diffusion regularization, with dataset-specific regularization weights \lambda.

| Dataset | C_g | Similarity metric | \lambda | Epochs | Trained Weights |
|---|---:|---|---:|---:|---:|
| Fluo-N2DH-GOWT1 | 36 | MSE | Model-specific¹ | 500 | [Link](https://drive.google.com/drive/folders/159_x3zVNYWoy9e056T8sZNAt-zWVFtuB?usp=drive_link) |
| ACDC | 32 | MSE | 0.01 | 500 | [Link](https://drive.google.com/drive/folders/1bBoTkp5bsj_6joQpsOOwHBTSdwVSscd3?usp=drive_link) |
| OASIS-1 | 48 | MSE | 0.005 | 10² | [Link](https://drive.google.com/drive/folders/16q742DeROdkYg8wDtx1bU0FGeX_E-1F_) |
| MindBoggle-101 | 32 | MSE | 0.05 | 500 | [Link](https://drive.google.com/drive/folders/1Zbg-vsYWWHtuW9zV1qyCR_8cuFlIEmYi?usp=drive_link) |
| Lung-CT | 32 | MSE | 0.001 | 500³ | [Link](https://drive.google.com/drive/folders/1GaUNjM4hseievRyYPjOC7WLVWlhnpEJ7?usp=sharing) |
| Abdomen-CT | 32 | LNCC | 1.0 | 500 | [Link](https://drive.google.com/drive/folders/167Jz2e4uIzcBA_idUW8uubxW8t2AaCYy?usp=sharing) |
| BCV-CT | 32 | LNCC | 1.0 | 500 | [Link](https://drive.google.com/drive/folders/1PUX0uz7kdxXjUg4TIrJvp0SB54zhkhEn?usp=sharing) |
| CHAOS-MR | 32 | MSE | 0.05 | 500 | [Link](https://drive.google.com/drive/folders/1AoKMHMs-qJMmOdbWbDlUY4TGRa0Y0S8i?usp=sharing) |
| BCV–CHAOS | 32 | MI | 0.5 | 500 | [Link](https://drive.google.com/drive/folders/166QK_pf_knibq74Bq8yhXxHdm6Xl5ZTQ?usp=sharing) |
| Abdomen-CT–MR | 32 | MI + MIND | 0.105 | 500 | [Link](https://drive.google.com/drive/folders/11Dc3m_KAT1xe0GZH3kfpYTkqs7n6zLuj?usp=sharing) |

Here, C_g denotes the channel dimensionality of the feature representations \(g^{\mathrm{ins}}_M\), \(g^{\mathrm{str}}_M\), \(g^{\mathrm{ins}}_F\), and \(g^{\mathrm{str}}_F\).

**Additional training details:**

1. **Fluo-N2DH-GOWT1:** The diffusion regularization weight was individually tuned for each model from `{0.001, 0.0005, 0.0001, 0.00005, 0.00001}` based on validation performance.
2. **OASIS-1:** Training was limited to 10 epochs due to the large number of available training pairs.
3. **Lung-CT:** Registration employed a stationary velocity field (SVF) diffeomorphic formulation with scaling-and-squaring integration.
4. **Abdomen-CT–MR:** The similarity objective combined mutual information (MI) and modality-independent neighbourhood descriptors (MIND):

   \[
   \mathcal{L}_{S} = \mathcal{L}_{MI} + 0.05 \times \mathcal{L}_{MIND}
   \]

   The relative weights were determined through empirical validation to balance global statistical alignment and local feature matching.

### Dataset Availability

The registration benchmarks were constructed from publicly available biomedical imaging datasets. Users should obtain the datasets from their original sources and follow the corresponding preprocessing and experimental configuration instructions to reproduce the reported results.


