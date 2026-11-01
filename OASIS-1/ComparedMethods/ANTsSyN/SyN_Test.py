import os
import glob
import sys
from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import torch
import torch.nn as nn
from Functions import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
import random
import ants

from surface_distance import compute_surface_distances, compute_robust_hausdorff, compute_average_surface_distance
from skimage.metrics import structural_similarity as ssim_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_mutual_information as similarity_nmi

import nibabel as nib


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def parse_int_tuple(s):
    """Parse a comma-separated string into a tuple of integers."""
    try:
        return tuple(int(x.strip()) for x in s.replace('(', '').replace(')', '').split(','))
    except ValueError as exc:
        raise ArgumentTypeError(
            "Expected comma-separated integers (e.g., 100,50,25)."
        ) from exc


parser = ArgumentParser()
parser.add_argument(
    "--global_seed",
    type=int,
    dest="global_seed",
    default=2024,
    help="global_seed",
)
parser.add_argument(
    "--bs",
    type=int,
    dest="bs",
    default=1,
    help="batch_size",
)
parser.add_argument(
    '--reg_iterations',
    type=parse_int_tuple,
    required=True,
    help='Comma-separated registration iteration counts (e.g., 100,50,25).',
)
parser.add_argument(
    '--syn_metric',
    type=str,
    default='meansquares',
    choices=['LNCC9', 'mattes', 'meansquares', 'demons'],
    help='Similarity metric for SyN registration (default: meansquares).',
)
parser.add_argument(
    '--datapath',
    type=str,
    default='./',
)
opt = parser.parse_args()

global_seed = opt.global_seed
bs = opt.bs
reg_iterations = opt.reg_iterations
syn_metric = opt.syn_metric
datapath = opt.datapath


def jacobian_determinant_vxm(disp):
    """Compute the Jacobian determinant of a 2D or 3D displacement field."""
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    import pystrum.pynd.ndutils as nd

    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    J = np.gradient(disp + grid)

    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    dfdx = J[0]
    dfdy = J[1]
    return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def dice(pred1, truth1, truth2):
    dice_k = []
    mask4_value1 = np.unique(truth2)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_k.append(intersection / (np.sum(pred) + np.sum(truth)))
    return dice_k


def hd95_assd(pred1, truth1, truth2=None):
    hd95_k = []
    assd_k = []
    mask4_value1 = np.unique(truth2)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        hd, ad = compute_metrics_for_label(
            1,
            pred,
            truth,
            voxel_spacing=(1.0, 1.0),
            percentile=95,
        )
        hd95_k.append(hd)
        assd_k.append(ad)
    return hd95_k, assd_k


def compute_metrics_for_label(
    label,
    pred_mask,
    gt_mask,
    voxel_spacing=(1.0, 1.0),
    percentile=95,
):
    """Compute HD95 and ASSD for a given label.

    Returns (0.0, 0.0) when both masks are empty and (NaN, NaN) when
    exactly one mask is empty.
    """
    pred_binary = pred_mask == label
    gt_binary = gt_mask == label

    if np.count_nonzero(pred_binary) == 0 and np.count_nonzero(gt_binary) == 0:
        print(np.count_nonzero(pred_binary), np.count_nonzero(gt_binary))
        return 0.0, 0.0
    elif np.count_nonzero(pred_binary) == 0 or np.count_nonzero(gt_binary) == 0:
        print(np.count_nonzero(pred_binary), np.count_nonzero(gt_binary))
        return np.nan, np.nan

    surface_distances = compute_surface_distances(
        gt_binary,
        pred_binary,
        voxel_spacing,
    )
    hd95 = compute_robust_hausdorff(surface_distances, percentile)

    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    assd = np.concatenate([distances_gt_to_pred, distances_pred_to_gt]).mean()
    return hd95, assd


def test(csv_name):
    bs = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    Dices = []
    HD95s = []
    ASSDs = []
    RMSEs = []
    SSIMs = []
    NMIs = []
    NegJ = []
    SDJ = []
    test_set = ValidationDataset(datapath, img_file='test_list.txt')
    test_generator = Data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    for mov_name, fix_name, mov_img, fix_img, mov_lab, fix_lab in test_generator:
        with torch.no_grad():
            mov_img = mov_img.squeeze().detach().cpu().numpy()
            fix_img = fix_img.squeeze().detach().cpu().numpy()
            mov_lab = mov_lab.squeeze().detach().cpu().numpy()
            fix_lab = fix_lab.squeeze().detach().cpu().numpy()

            x = ants.from_numpy(mov_img)
            y = ants.from_numpy(fix_img)
            x_ants = ants.from_numpy(mov_lab.astype(np.float32))
            y_ants = ants.from_numpy(fix_lab.astype(np.float32))
            if syn_metric == 'LNCC9':
                reg12 = ants.registration(
                    y,
                    x,
                    'SyNOnly',
                    reg_iterations=reg_iterations,
                    multivariate_extras=[["CC", y, x, 1, 4]],
                )
            else:
                reg12 = ants.registration(
                    y,
                    x,
                    'SyNOnly',
                    reg_iterations=reg_iterations,
                    syn_metric=syn_metric,
                )

            def_seg = ants.apply_transforms(
                fixed=y_ants,
                moving=x_ants,
                transformlist=reg12['fwdtransforms'],
                interpolator='nearestNeighbor',
            )
            def_x = ants.apply_transforms(
                fixed=y,
                moving=x,
                transformlist=reg12['fwdtransforms'],
                interpolator='linear',
            )
            flow = np.array(
                nib_load(reg12['fwdtransforms'][0]),
                dtype='float32',
                order='C',
            )
            flow = flow[:, :, :, 0, :].transpose(3, 0, 1, 2)
            def_x = def_x.numpy()
            def_seg = def_seg.numpy()

            warped_mov_img = def_x
            warped_mov_lab = def_seg

            rmse = np.sqrt(mean_squared_error(warped_mov_img, fix_img))
            ssim = ssim_similarity(
                warped_mov_img,
                fix_img,
                data_range=fix_img.max() - fix_img.min(),
            )
            nmi = similarity_nmi(warped_mov_img, fix_img)

            RMSEs.append(rmse)
            SSIMs.append(ssim)
            NMIs.append(nmi)

            hh, ww = flow.shape[1:3]
            if True:
                dice_bs_list = dice(
                    warped_mov_lab.copy(),
                    fix_lab.copy(),
                    mov_lab.copy(),
                )
                dice_bs = np.mean(dice_bs_list)
                hd95_bs_list, assd_bs_list = hd95_assd(
                    warped_mov_lab.copy(),
                    fix_lab.copy(),
                    mov_lab.copy(),
                )
                hd95_bs = np.mean(hd95_bs_list)
                assd_bs = np.mean(assd_bs_list)

                Dices.append(dice_bs)
                HD95s.append(hd95_bs)
                ASSDs.append(assd_bs)
                jac_det = jacobian_determinant_vxm(flow[:, :, :, 0])
                negJ = np.sum(jac_det <= 0) / hh / ww * 100
                NegJ.append(negJ)
                sd_jac_det = jac_det.std()
                SDJ.append(sd_jac_det)
                f = open(csv_name, 'a')
                with f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            mov_name[0],
                            fix_name[0],
                            dice_bs,
                            hd95_bs,
                            assd_bs,
                            rmse,
                            ssim,
                            nmi,
                            negJ,
                            sd_jac_det,
                        ]
                    )

    print('Dice: {:.4f} ± {:.4f}'.format(np.mean(Dices), np.std(Dices)))
    print('HD95s: {:.4f} ± {:.4f}'.format(np.mean(HD95s), np.std(HD95s)))
    print('ASSDs: {:.4f} ± {:.4f}'.format(np.mean(ASSDs), np.std(ASSDs)))
    print('RMSEs: {:.4f} ± {:.4f}'.format(np.mean(RMSEs), np.std(RMSEs)))
    print('SSIMs: {:.4f} ± {:.4f}'.format(np.mean(SSIMs), np.std(SSIMs)))
    print('NMIs: {:.4f} ± {:.4f}'.format(np.mean(NMIs), np.std(NMIs)))
    print(
        'NegJ 100% {:.4f} ± {:.4f} {}'.format(
            np.mean(NegJ),
            np.std(NegJ),
            np.mean(NegJ),
        )
    )
    print('SDJ: {:.4f} ± {:.4f}'.format(np.mean(SDJ), np.std(SDJ)))


if __name__ == '__main__':
    import time

    csv_name = (
        f'Test_SyNOnly_Seed_{global_seed}_Metric_{syn_metric}_'
        f'Iterations_{reg_iterations}.csv'
    )
    print('----------------------------------------------------------------------------------------')
    print(csv_name)
    assert os.path.exists(csv_name) == 0
    f = open(csv_name, 'w')
    with f:
        fnames = [
            'Mov',
            'Fix',
            'Dice_Avg',
            'HD95_Avg',
            'ASSD_Avg',
            'RMSE_Avg',
            'SSIM_Avg',
            'NMI_Avg',
            'Neg_J',
            'Std_J',
        ]
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    start = time.perf_counter()
    test(csv_name)
    end = time.perf_counter()
    print(f"Runtime: {end - start:.6f} seconds")
