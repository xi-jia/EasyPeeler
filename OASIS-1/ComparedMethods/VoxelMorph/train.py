import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv
import random

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = ArgumentParser()

parser.add_argument(
    "--global_seed", type=int, dest="global_seed", default=0, help="global_seed"
)

parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int, dest="bs", default=10, help="batch_size")
parser.add_argument(
    "--iteration",
    type=int,
    dest="iteration",
    default=320001,
    help="number of total iterations",
)

parser.add_argument(
    "--smth_labda",
    type=float,
    dest="smth_labda",
    default=0.02,
    help="labda loss: suggested range 0.1 to 10",
)
parser.add_argument(
    "--checkpoint",
    type=int,
    dest="checkpoint",
    default=4000,
    help="frequency of saving models",
)
parser.add_argument(
    "--vm_model", type=int, dest="vm_model", default=1, help="number of start VMnels"
)
parser.add_argument(
    "--datapath",
    type=str,
    dest="datapath",
    default="/Dataset_OASIS/",
    help="data path for training images",
)
parser.add_argument(
    "--trainingset",
    type=int,
    dest="trainingset",
    default=4,
    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images",
)
parser.add_argument(
    "--using_l2", type=int, dest="using_l2", default=1, help="using l2 or not"
)
opt = parser.parse_args()


global_seed = opt.global_seed
lr = opt.lr
bs = opt.bs
iteration = opt.iteration
vm_model = opt.vm_model
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
trainingset = opt.trainingset
using_l2 = opt.using_l2


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(global_seed)
setup_seed(seed=global_seed, cuda_deterministic=True)


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
    return np.mean(dice_k)


def save_checkpoint(state, save_dir, save_filename, max_model_num=11):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + "*"))

    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + "*"))


def train():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if vm_model == 1:
        model = VxmDense_1(inshape=(128, 128)).to(device)
    elif vm_model == 2:
        model = VxmDense_2(inshape=(128, 128)).to(device)

    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC(win=9)

    loss_smooth = smoothloss

    transform = SpatialTransform().to(device)
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((3, iteration))
    train_set = TrainDataset(
        datapath, img_file="train_list.txt", trainingset=trainingset
    )
    training_generator = Data.DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_set = ValidationDataset(opt.datapath, img_file="val_list.txt")
    test_generator = Data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    model_dir = "./Seed_{}_L2ss_{}_VM_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Png/".format(
        global_seed, using_l2, vm_model, smooth, trainingset, lr, bs
    )
    model_dir_pth1 = "./Seed_{}_L2ss_{}_VM_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth/".format(
        global_seed, using_l2, vm_model, smooth, trainingset, lr, bs
    )

    csv_name = "Seed_{}_L2ss_{}_VM_{}_Smth_{}_Set_{}_LR_{}_BZ_{}.csv".format(
        global_seed, using_l2, vm_model, smooth, trainingset, lr, bs
    )
    if os.path.exists(csv_name):
        assert 0 == 1
    f = open(csv_name, "w")
    with f:
        fnames = ["Index", "Dice"]
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(model_dir_pth1):
        os.mkdir(model_dir_pth1)

    step = 1

    while step <= iteration:
        for mov_img, fix_img in training_generator:

            fix_img = fix_img.to(device).float()

            mov_img = mov_img.to(device).float()

            f_xy = model(mov_img, fix_img)

            __, warped_mov = transform(mov_img, f_xy.permute(0, 2, 3, 1))

            loss1 = loss_similarity(fix_img, warped_mov)
            loss5 = loss_smooth(f_xy)

            loss = loss1 + smooth * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:, step] = np.array([loss.item(), loss1.item(), loss5.item()])
            sys.stdout.write(
                "\r"
                + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" -smo "{3:.4f}" '.format(
                    step, loss.item(), loss1.item(), loss5.item()
                )
            )
            sys.stdout.flush()

            if (step % n_checkpoint == 0) or (step == 1):
                with torch.no_grad():
                    Dices_Validation = []
                    for (
                        __,
                        __,
                        vmov_img,
                        vfix_img,
                        vmov_lab,
                        vfix_lab,
                    ) in test_generator:
                        model.eval()
                        V_xy = model(
                            vmov_img.float().to(device), vfix_img.float().to(device)
                        )
                        grid, warped_vmov_lab = transform(
                            vmov_lab.float().to(device),
                            V_xy.permute(0, 2, 3, 1),
                            mod="nearest",
                        )
                        dice_bs = dice(
                            warped_vmov_lab[0, ...].data.cpu().numpy().copy(),
                            vfix_lab[0, ...].data.cpu().numpy().copy(),
                            vmov_lab[0, ...].data.cpu().numpy().copy(),
                        )
                        Dices_Validation.append(dice_bs)
                    modelname = "DiceVal_{:.4f}_Step_{:09d}.pth".format(
                        np.mean(Dices_Validation), step
                    )
                    csv_dice = np.mean(Dices_Validation)

                    save_checkpoint(model.state_dict(), model_dir_pth1, modelname)

                    np.save(model_dir + "Loss.npy", lossall)
                    f = open(csv_name, "a")
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])

            step += 1

            if step > iteration:
                break
        print("one epoch pass")
        np.save(model_dir + "/Loss.npy", lossall)
        torch.save(
            {
                "step": step,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_dir + "/latest.pth.tar",
        )
        torch.save(model.state_dict(), model_dir + "/latest_model.pth.tar")

train()
