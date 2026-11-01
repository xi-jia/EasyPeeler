# Single 2D slice feature extraction with ResNet152
import os, pickle
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import nibabel as nib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = models.resnet152(weights='DEFAULT').to(device).eval()

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

load_path = '/Datasets/Dataset_OASIS/'
save_path = '/Datasets/Dataset_OASIS/OASIS_resnet152_PreNorm_32x32/'
os.makedirs(save_path)

transform = T.Compose([
    T.Resize((128, 128), interpolation=T.InterpolationMode.BICUBIC),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

for name in np.loadtxt(os.path.join(load_path, 'subjects.txt'), dtype='str'):
    print(name)
    src = os.path.join(load_path, name)
    img_np = nib.load(os.path.join(src, 'slice_norm.nii.gz')).get_fdata()[:, :, 0]
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    x = transform(torch.tensor(img_3c).permute(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = feature_extractor(x)[0].cpu().numpy()

    os.makedirs(os.path.join(save_path, name), exist_ok=True)
    with open(os.path.join(save_path, name, 'slice_norm_0.pkl'), 'wb') as f:
        pickle.dump(features, f)

    for seg in ('slice_seg24.nii.gz', 'slice_seg4.nii.gz'):
        d = nib.load(os.path.join(src, seg)).get_fdata()
        nib.save(nib.Nifti1Image(d, np.eye(4)), os.path.join(save_path, name, seg))