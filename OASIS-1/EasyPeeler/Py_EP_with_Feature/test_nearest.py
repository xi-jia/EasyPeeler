import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import ValidationDataset
import torch.utils.data as Data
import csv
from natsort import natsorted
import random
from surface_distance import compute_surface_distances, compute_robust_hausdorff, compute_average_surface_distance
from skimage.metrics import structural_similarity as ssim_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_mutual_information as similarity_nmi

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

####################################################################################
##################################Reproducibility###################################
####################################################################################
def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    # disp = disp.transpose(1, 2, 3, 0)
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    
    import pystrum.pynd.ndutils as nd
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def dice(pred1, truth1, truth2):
    dice_k=[]
    mask4_value1 = np.unique(truth2)
    mask4_value2 = np.unique(truth1)
    # print(len(mask4_value2))
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    # mask_value4 = [0,1,2,3]
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_k.append(intersection / (np.sum(pred) + np.sum(truth)))
    return dice_k
    
def hd95_assd(pred1, truth1, truth2=None):
    hd95_k=[]
    assd_k=[]
    mask4_value1 = np.unique(truth2)
    mask4_value2 = np.unique(truth1)
    # print(len(mask4_value2))
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        hd, ad = compute_metrics_for_label(1, pred, truth, voxel_spacing=(1.0, 1.0), percentile=95)
        hd95_k.append(hd)
        assd_k.append(ad)
    return hd95_k, assd_k

def compute_metrics_for_label(label, pred_mask, gt_mask, voxel_spacing=(1.0, 1.0), percentile=95):
    """
    Compute HD, HD95, and ASSD for a given label.

    Args:
        label (int): The label for which to compute the metrics.
        pred_mask (np.ndarray): The predicted mask (with labels).
        gt_mask (np.ndarray): The ground truth mask (with labels).
        voxel_spacing (tuple): Voxel spacing in mm (default assumes isotropic 1mm).
        percentile (int): The percentile for robust Hausdorff (default 95).

    Returns:
        hd (float): Hausdorff Distance.
        hd95 (float): 95th Percentile Hausdorff Distance.
        assd (float): Average Symmetric Surface Distance.
    """
    # Create binary masks for the label of interest.
    pred_binary = (pred_mask == label)#.astype(np.uint8)
    gt_binary = (gt_mask == label)#.astype(np.uint8)
    
    # Check if both masks have at least one positive voxel.
    if np.count_nonzero(pred_binary) == 0 and np.count_nonzero(gt_binary) == 0:
        # Both are empty: define metrics as 0 (or choose as appropriate).
        # print(np.count_nonzero(pred_binary), np.count_nonzero(gt_binary))
        # print(dice(pred_mask, gt_mask, gt_mask))
        return 0.0, 0.0
    elif np.count_nonzero(pred_binary) == 0 or np.count_nonzero(gt_binary) == 0:
        # One mask is empty and the other is not: metrics are undefined.
        # print('nan   ', np.count_nonzero(pred_binary), np.count_nonzero(gt_binary))
        # print(dice(pred_mask, gt_mask, gt_mask))
        return np.nan, np.nan
        
    # Compute the surface distances between the two binary masks.
    surface_distances = compute_surface_distances(gt_binary, pred_binary, voxel_spacing)
    
    # Hausdorff Distance: maximum of all surface distances.
    # distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    # distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    # hd = np.concatenate([distances_gt_to_pred,distances_pred_to_gt,]).max()
    # print(hd)
    
    # 95th Percentile Hausdorff Distance.
    hd95 = compute_robust_hausdorff(surface_distances, percentile)
    # print(hd95)
    
    # Average Symmetric Surface Distance.
    # assd = compute_average_surface_distance(surface_distances)
    # print(assd, sum(assd)/len(assd))
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    assd =  np.concatenate([distances_gt_to_pred,distances_pred_to_gt,]).mean()
    # print(assd)
    return hd95, assd
def test(model_dir, csv_name):
    bs = 1
    use_cuda = torch.cuda.is_available()
    # torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if use_cuda else "cpu")
    assert feat >= 1 and feat <= 6 
    feat_number = 768
    if feat==6:
        feat_number = 256
    model = UNet(start_channel,feat_number).to(device)

    model_idx = -1
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location=device)#['state_dict']
    model.load_state_dict(best_model)
    
    transform = SpatialTransform().to(device)
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)
    #model_lambda = model.ic_block.labda.data.cpu().numpy()
    #model_odr = model.ic_block.odr.data.cpu().numpy()
    model.eval()
    transform.eval()
    diff_transform.eval()
    
    Dices=[]
    HD95s=[]
    ASSDs=[]
    RMSEs=[]
    SSIMs=[]
    NMIs =[]
    NegJ=[]
    SDJ=[]
    
    test_set = ValidationDataset(opt.datapath,img_file='test_list.txt', feat_to_use = feat)
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=2)
    for mov_name, fix_name, mov_feat, fix_feat, mov_img, fix_img, mov_lab, fix_lab in test_generator:
        with torch.no_grad():
            model.eval()
            mov_img = mov_img.float().to(device)
            fix_img = fix_img.float().to(device)
            mov_feat = mov_feat.float().to(device)
            fix_feat = fix_feat.float().to(device)
            V_xy  = model(mov_feat, fix_feat, mov_img, fix_img)
            if diff == 1:
                Df_xy = diff_transform(V_xy)
            else:
                Df_xy = V_xy

            __, warped_mov_lab = transform(mov_lab.float().to(device), Df_xy.permute(0, 2, 3, 1), mod = 'nearest')
            __, warped_mov_img = transform(mov_img.float().to(device), Df_xy.permute(0, 2, 3, 1), mod = 'bilinear')
                
            rmse = np.sqrt(mean_squared_error(warped_mov_img.data.cpu().squeeze().numpy(), fix_img.data.cpu().squeeze().numpy()))
            ssim = ssim_similarity(warped_mov_img.data.cpu().squeeze().numpy(), fix_img.data.cpu().squeeze().numpy(), data_range=fix_img.data.cpu().squeeze().numpy().max() - fix_img.data.cpu().squeeze().numpy().min())
            nmi = similarity_nmi(warped_mov_img.data.cpu().squeeze().numpy(), fix_img.data.cpu().squeeze().numpy())
            NMIs.append(nmi)
            
            RMSEs.append(rmse)
            SSIMs.append(ssim)
            
            D_V_xy = Df_xy
            hh, ww = D_V_xy.shape[-2:]
            D_V_xy = D_V_xy.detach().cpu().numpy()
            D_V_xy[:,0,:,:] = D_V_xy[:,0,:,:] * (hh - 1) / 2
            D_V_xy[:,1,:,:] = D_V_xy[:,1,:,:] * (ww - 1) / 2
            
            for bs_index in range(bs):
                dice_bs_list1 = dice(mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy(),mov_lab[bs_index,...].data.cpu().numpy().copy())
                dice_bs1 = np.mean(dice_bs_list1)
                # print(len(dice_bs_list1), 'was:,   ', dice_bs1, ':,   ', dice_bs_list1)
                dice_bs_list = dice(warped_mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy(),mov_lab[bs_index,...].data.cpu().numpy().copy())
                dice_bs = np.mean(dice_bs_list)
                # print(len(dice_bs_list), 'now:,   ', dice_bs, ':,   ', dice_bs_list)
                hd95_bs_list, assd_bs_list = hd95_assd(warped_mov_lab[bs_index,...].data.cpu().squeeze().numpy().copy(),fix_lab[bs_index,...].data.cpu().squeeze().numpy().copy(),mov_lab[bs_index,...].data.cpu().squeeze().numpy().copy())
                hd95_bs = np.mean(hd95_bs_list)
                assd_bs = np.mean(assd_bs_list)

                # print(f'{len(dice_bs_list)} labels {fix_lab.shape}')
                Dices.append(dice_bs)
                HD95s.append(hd95_bs)
                ASSDs.append(assd_bs)
                
                jac_det = jacobian_determinant_vxm(D_V_xy[0, :, :, :])
                negJ = np.sum(jac_det <= 0) / hh / ww * 100
                NegJ.append(negJ)
                # if np.min(jac_det)<0:
                    # print(np.min(jac_det))
                    # sdJ = (jac_det - np.min(jac_det) + 1).clip(0.000000001, 1000000000)
                # else:
                    # sdJ = (jac_det + 1).clip(0.000000001, 1000000000)
                sd_jac_det = jac_det.std()#np.log(sdJ).std()
                SDJ.append(sd_jac_det)
                f = open(csv_name, 'a')
                with f:
                    writer = csv.writer(f)
                    writer.writerow([mov_name[0], fix_name[0], dice_bs, hd95_bs, assd_bs, rmse, ssim, nmi, negJ,  sd_jac_det])

    print('Dice: {:.4f} ± {:.4f}'.format(np.mean(Dices), np.std(Dices)))
    print('HD95s: {:.4f} ± {:.4f}'.format(np.mean(HD95s), np.std(HD95s)))
    print('ASSDs: {:.4f} ± {:.4f}'.format(np.mean(ASSDs), np.std(ASSDs)))
    print('RMSEs: {:.4f} ± {:.4f}'.format(np.mean(RMSEs), np.std(RMSEs)))
    print('SSIMs: {:.4f} ± {:.4f}'.format(np.mean(SSIMs), np.std(SSIMs)))
    print('NMIs: {:.4f} ± {:.4f}'.format(np.mean(NMIs), np.std(NMIs)))
    print('NegJ 100% {:.4f} ± {:.4f} {}'.format(np.mean(NegJ), np.std(NegJ), np.mean(NegJ)))
    print('SDJ: {:.4f} ± {:.4f}'.format(np.mean(SDJ), np.std(SDJ)))




if __name__ == '__main__':
    csv_name = 'Infer_Diff_{}_Seed_{}_Feat_{}_L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}.csv'.format(diff, global_seed, feat, using_l2, start_channel, smooth, trainingset, lr,bs)
    assert os.path.exists(csv_name)==0
    f = open(csv_name, 'w')
    with f:
        fnames = ['Mov', 'Fix', 'Dice_Avg', 'HD95_Avg', 'ASSD_Avg', 'RMSE_Avg', 'SSIM_Avg', 'NMI_Avg', 'NegJ', 'SDJ']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    model_dir = './Diff_{}_Seed_{}_Feat_{}_L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth/'.format(diff, global_seed, feat, using_l2, start_channel, smooth, trainingset, lr,bs)
    print(model_dir)
    # import time
    # start = time.time()
    test(model_dir, csv_name)
    # print((time.time()-start)/400.0)
