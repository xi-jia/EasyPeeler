
from torchinfo import summary
from Models import *

# model = UNet(2, 2, 8)
# summary(model, [(1, 160, 192),(1, 160, 192)])


# model = LKUNet(2, 2, 8)
# summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
# model = UNet(16)
# summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
# model = UNet(24)
# summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
# model = UNet(32)
# summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
# model = UNet(36)
# summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
# model = UNet(40)
# summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])
model = UNet(48)
summary(model, [(1, 1, 160, 192),(1, 1, 160, 192)])

# summary(model, [(1, 40, 48),(1, 40, 48)])
# summary(model, [(1, 160, 192, 224),(1, 160, 192, 224)])
# summary(model, (1, 128,128))#,(1, 80, 96, 112)])
