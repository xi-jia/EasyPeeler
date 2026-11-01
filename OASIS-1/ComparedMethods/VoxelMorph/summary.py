from torchinfo import summary
from Models import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = VxmDense_2(inshape=(128, 128)).to(device)
summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
