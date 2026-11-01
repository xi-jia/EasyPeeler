# Slices along axis 1 (middle) : 54 slices x 3 channels = 162
import os, pickle
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import nibabel as nib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = models.resnet18(weights='IMAGENET1K_V1').to(device).eval()

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super().__init__()
        self.submodule, self.extracted_layers = submodule, extracted_layers
    def forward(self, x):
        for name, layer in self.submodule._modules.items():
            x = layer(x)
            if name in self.extracted_layers:
                return x

feature_extractor = FeatureExtractor(model, ['maxpool'])

load_path = '/Datasets/L2R_MR_CT/L2R_Task1_MR/'
save_path = '/Datasets/L2R_MR_CT/L2R_Task1_MR_Res18_PreNorm_32_32_Dim2'
os.makedirs(save_path)
os.makedirs(os.path.join(save_path, 'Feat1_images'))

transform = T.Compose([
    T.Resize((128, 128), interpolation=T.InterpolationMode.BICUBIC),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

for name in np.loadtxt('chaos_mr_files.txt', dtype='str'):
    print(name)
    img = np.array(nib.load(os.path.join(load_path, name + '.nii.gz')).get_fdata(), dtype='float32')
    feat = torch.zeros((64, 32, 54, 32))
    with torch.no_grad():
        for i in range(54):
            img_3c = img[:, i * 3:(i + 1) * 3, :]                         # (192,3,192)
            img_3c = np.transpose(img_3c, (0, 2, 1))                      # (192,192,3)
            x = transform(torch.tensor(img_3c).permute(2, 0, 1)).unsqueeze(0).float().to(device)
            feat[:, :, i, :] = feature_extractor(x).squeeze().cpu()
    with open(os.path.join(save_path, 'Feat1_images', name + '.pkl'), 'wb') as f:
        pickle.dump(feat.numpy(), f)