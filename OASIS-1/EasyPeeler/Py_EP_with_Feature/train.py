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

parser = ArgumentParser()

# Reproducibility
parser.add_argument("--global_seed", type=int,
                    dest="global_seed", default=2024, help="global_seed")

# Model
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")

# Optimizer
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000,
                    help="frequency of saving models")

#Input
parser.add_argument("--feat", type=int, dest="feat", default=0, help="feature to use")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    #default='/export/local/',
                    default='/Datasets/Dataset_OASIS/OASIS_AffineData/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")

# Loss
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.02,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--diff", type=int,
                    dest="diff",
                    default=0)
opt = parser.parse_args()

print(opt)
# Reproducibility
global_seed = opt.global_seed
# Model
start_channel = opt.start_channel
# Optimizer
lr = opt.lr
bs = opt.bs
iteration = opt.iteration
n_checkpoint = opt.checkpoint
#Input
feat = opt.feat
datapath = opt.datapath
trainingset = opt.trainingset
# Loss
diff = opt.diff
using_l2 = opt.using_l2
smooth = opt.smth_labda
####################################################################################
##################################Reproducibility###################################
####################################################################################

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
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
    dice_k=[]
    mask4_value1 = np.unique(truth2)
    mask4_value2 = np.unique(truth1)
    # print(len(mask4_value2))
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_k.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_k)

def save_checkpoint(state, save_dir, save_filename, max_model_num=11):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    assert feat >= 1 and feat <= 6 
    feat_number = 768
    if feat==6:
        feat_number = 256
    model = UNet(start_channel,feat_number).to(device)
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
    train_set = TrainDataset(datapath,img_file='train_list.txt',trainingset = trainingset, feat_to_use = feat)
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_set = ValidationDataset(opt.datapath,img_file='val_list.txt', feat_to_use = feat)
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    model_dir = './Diff_{}_Seed_{}_Feat_{}_L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Png/'.format(diff, global_seed, feat, using_l2, start_channel, smooth, trainingset, lr,bs)
    model_dir_pth = './Diff_{}_Seed_{}_Feat_{}_L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth/'.format(diff, global_seed, feat, using_l2, start_channel, smooth, trainingset, lr,bs)
    csv_name = 'Diff_{}_Seed_{}_Feat_{}_L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}.csv'.format(diff, global_seed, feat, using_l2, start_channel, smooth, trainingset, lr,bs)
    if os.path.exists(csv_name):
        assert 0==1
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(model_dir_pth):
        os.mkdir(model_dir_pth)
    
    step = 1

    while step <= iteration:
        for mov_feat, fix_feat, mov_img, fix_img in training_generator:

            mov_feat = mov_feat.to(device).float()
            fix_feat = fix_feat.to(device).float()
            mov_img = mov_img.to(device).float()
            fix_img = fix_img.to(device).float()
            
            f_xy = model(mov_feat, fix_feat, mov_img, fix_img)#, mov_80, fix_80)
            if diff == 1:
                Df_xy = diff_transform(f_xy)
            else:
                Df_xy = f_xy
            __, warped_mov_img = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
           
            loss1 = loss_similarity(fix_img, warped_mov_img) # GT shall be 1st Param
            loss5 = loss_smooth(f_xy)
            
            loss = loss1 + smooth * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lossall[:,step] = np.array([loss.item(),loss1.item(),loss5.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" -smo "{3:.4f}" '.format(step, loss.item(),loss1.item(),loss5.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):# and (step >= 0.5 * iteration):
                with torch.no_grad():
                    Dices_Validation = []
                    for __, __, vmov_feat, vfix_feat, vmov_img, vfix_img, vmov_lab, vfix_lab in test_generator:
                        model.eval()
                        vmov_img = vmov_img.float().to(device)
                        vfix_img = vfix_img.float().to(device)
                        vmov_feat = vmov_feat.float().to(device)
                        vfix_feat = vfix_feat.float().to(device)

                        V_xy = model(vmov_feat, vfix_feat, vmov_img, vfix_img)#, vmov_80, vfix_80)
                        # DV_xy = diff_transform(V_xy)
                        DV_xy = V_xy
                        
                        grid, warped_vmov_lab = transform(vmov_lab.float().to(device), DV_xy.permute(0, 2, 3, 1), mod = 'nearest')
                        dice_bs = dice(warped_vmov_lab[0,...].data.cpu().numpy().copy(),vfix_lab[0,...].data.cpu().numpy().copy(),vmov_lab[0,...].data.cpu().numpy().copy())
                        Dices_Validation.append(dice_bs)
                    csv_dice = np.mean(Dices_Validation)
                    modelname = 'DiceVal_{:.4f}_Step_{:09d}.pth'.format(csv_dice, step)
                    save_checkpoint(model.state_dict(), model_dir_pth, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
        np.save(model_dir + '/Loss.npy', lossall)
        torch.save({
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_dir + '/latest.pth.tar')
        torch.save(model.state_dict(), model_dir + '/latest_model.pth.tar')

train()
