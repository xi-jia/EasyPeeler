# EasyPeeler — OASIS 2D Reproducibility Guide

This README provides dataset-specific instructions for reproducing the **OASIS 2D** experiments in the EasyPeeler study. It covers classical registration with ANTs SyN, learning-based comparison methods, baseline EasyPeeler variants, and the additional requirements for foundation-feature variants.

## Contents

- [Dataset preparation](#dataset-preparation)
- [Experimental configuration](#experimental-configuration)
- [Classical registration: ANTs SyN](#classical-registration-ants-syn)
- [Comparison methods](#comparison-methods)
- [EasyPeeler variants](#easypeeler-variants)
- [Foundation-feature variants](#foundation-feature-variants)
- [Reproducibility notes](#reproducibility-notes)

## Dataset preparation

1. Download **OASIS 2D** using the [Neurite OASIS dataset instructions](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md).
2. Extract the dataset into a local directory, for example `Dataset_OASIS`.
3. Update `--datapath` in all commands to point to your extracted dataset.

The commands below retain the original absolute path `--datapath /Dataset_OASIS/`. This refers to a directory at the filesystem root; replace it with the actual path on your system.

## Experimental configuration

The following settings are common to the learning-based models. All command-line argument names and values are preserved from the original README.

| Parameter | Value |
| --- | --- |
| Random seeds (`--global_seed`) | `0`, `1`, `2` |
| Learning rate (`--lr`) | `1e-4` |
| Smoothness weight (`--smth_labda`) | `0.005` |
| L2 setting (`--using_l2`) | `1` |
| Training-set selection (`--trainingset`) | `4` |
| Iterations (`--iteration`) | `804001` |
| Checkpoint argument (`--checkpoint`) | `2010` |
| Batch size (`--bs`) | `1` |

Model-specific arguments:

| Method | Model-specific flags |
| --- | --- |
| VoxelMorph | `--vm_model 2` |
| TransMorph | `--diff_model 0` |
| LKUNet | `--diff 0 --start_channel 16` |
| Baseline EasyPeeler | `--diff 0 --start_channel 48` |
| Baseline Pyramid EasyPeeler | `--diff 0 --start_channel 48` |

**How to run:** Execute each code block starting in the repository root. Each block changes to the correct method directory. Training and evaluation commands are shown for all three seeds.

## Classical registration: ANTs SyN

Run the following single-scale and pyramid registration configurations.

```bash
cd ./ANTsSyN/

# Single-scale SyN
python SyN_Test.py --global_seed 0 --reg_iterations 1280 --syn_metric meansquares --datapath /Dataset_OASIS/

# Pyramid SyN
python SyN_Test.py --global_seed 0 --reg_iterations 320,160,80 --syn_metric LNCC9 --syn_metric meansquares  --datapath /Dataset_OASIS/
```

## Comparison methods

The comparison methods are **VoxelMorph**, **TransMorph**, and **LKUNet**. For each method, inspect the model summary, train with three seeds, and evaluate on the test set.

### VoxelMorph

**Model summary** — run `summary.py` to inspect available model statistics (such as parameter counts and multiply–add operations).

```bash
cd ./VoxelMorph/
python summary.py
```

**Training**

```bash
cd ./VoxelMorph/
python train.py --vm_model 2 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 0 --datapath /Dataset_OASIS/
python train.py --vm_model 2 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 1 --datapath /Dataset_OASIS/
python train.py --vm_model 2 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 2 --datapath /Dataset_OASIS/
```

**Test-set evaluation**

```bash
cd ./VoxelMorph/
python test_nearest.py --vm_model 2 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 0 --datapath /Dataset_OASIS/
python test_nearest.py --vm_model 2 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 1 --datapath /Dataset_OASIS/
python test_nearest.py --vm_model 2 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 2 --datapath /Dataset_OASIS/
```

### TransMorph

**Model summary** — run `summary.py` to inspect available model statistics (such as parameter counts and multiply–add operations).

```bash
cd ./TransMorph/
python summary.py
```

**Training**

```bash
cd ./TransMorph/
python train.py --diff_model 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 0 --datapath /Dataset_OASIS/
python train.py --diff_model 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 1 --datapath /Dataset_OASIS/
python train.py --diff_model 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 2 --datapath /Dataset_OASIS/
```

**Test-set evaluation**

```bash
cd ./TransMorph/
python test_nearest.py --diff_model 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 0 --datapath /Dataset_OASIS/
python test_nearest.py --diff_model 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 1 --datapath /Dataset_OASIS/
python test_nearest.py --diff_model 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --global_seed 2 --datapath /Dataset_OASIS/
```

### LKUNet

**Model summary** — run `summary.py` to inspect available model statistics (such as parameter counts and multiply–add operations).

```bash
cd ./LKUNet/
python summary.py
```

**Training**

```bash
cd ./LKUNet/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 16 --global_seed 0 --datapath /Dataset_OASIS/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 16 --global_seed 1 --datapath /Dataset_OASIS/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 16 --global_seed 2 --datapath /Dataset_OASIS/
```

**Test-set evaluation**

```bash
cd ./LKUNet/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 16 --global_seed 0 --datapath /Dataset_OASIS/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 16 --global_seed 1 --datapath /Dataset_OASIS/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 16 --global_seed 2 --datapath /Dataset_OASIS/
```

## EasyPeeler variants

The OASIS 2D experiments include **Baseline EasyPeeler** and **Baseline Pyramid EasyPeeler**.

### Baseline EasyPeeler

**Training**

```bash
cd ./Baseline_EP/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 0 --datapath /Dataset_OASIS/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 1 --datapath /Dataset_OASIS/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 2 --datapath /Dataset_OASIS/
```

**Test-set evaluation**

```bash
cd ./Baseline_EP/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 0 --datapath /Dataset_OASIS/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 1 --datapath /Dataset_OASIS/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 2 --datapath /Dataset_OASIS/
```

### Baseline Pyramid EasyPeeler

**Training**

```bash
cd ./Baseline_Py_EP/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 0 --datapath /Dataset_OASIS/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 1 --datapath /Dataset_OASIS/
python train.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 2 --datapath /Dataset_OASIS/
```

**Test-set evaluation**

```bash
cd ./Baseline_Py_EP/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 0 --datapath /Dataset_OASIS/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 1 --datapath /Dataset_OASIS/
python test_nearest.py --diff 0 --using_l2 1 --smth_labda 0.005 --lr 1e-4  --trainingset 4 --iteration 804001 --checkpoint 2010 --bs 1 --start_channel 48 --global_seed 2 --datapath /Dataset_OASIS/
```

## Foundation-feature variants

**EasyPeeler with foundation features**, including the pyramid variant, follows the corresponding EasyPeeler training and testing structure but requires **feature extraction before the registration network forward pass**.

**An example feature extraction code using ResNet-18 and ResNet-152 in both 2D and 3D are provided in `./FeatureExtraction/`.**

Also refer to the **“Frozen Encoding Path”** section of the accompanying manuscript for the methodological description. 