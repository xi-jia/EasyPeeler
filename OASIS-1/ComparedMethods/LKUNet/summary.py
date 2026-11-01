from torchinfo import summary
from Models import *

model = LKUNet(2, 2, 16)
summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
