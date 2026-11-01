from torchinfo import summary
from Models import *
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from models.TransMorph_diff import CONFIGS as CONFIGS_TM_Diff
import models.TransMorph_diff as TransMorphDiff

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

config = CONFIGS_TM['TransMorph']
model = TransMorph.TransMorph(config)
model.to(device)
summary(model, [(1, 2, 160, 192)])
