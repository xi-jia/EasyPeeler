
from torchinfo import summary
from Models import *

# model = UNet(2, 2, 8)
# summary(model, [(1, 160, 192),(1, 160, 192)])


# model = UNet(6, 2, 4).cuda()
# summary(model, [(1, 160, 192),(1, 160, 192)])
# model = UNet(6, 2, 6).cuda()
# summary(model, [(1, 160, 192),(1, 160, 192)])
# model = UNet(16, 2)#.cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(24, 2)#.cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(32, 2)#.cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(40, 2)#.cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
model = UNet(48, 768)#.cuda()
summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(64, 2)#.cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(24, 2)#.cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(32, 2)#.cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(16, 2.5).cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(16, 3).cuda()
# summary(model, [(1, 768, 32, 32), (1, 768, 32, 32), (1, 1, 160, 192), (1, 1, 160, 192)] )
# model = UNet(6, 2, 12).cuda()
# summary(model, [(1, 160, 192),(1, 160, 192)])
# model = UNet(6, 2, 16).cuda()
# summary(model, [(1, 160, 192),(1, 160, 192)])



# model = UNet(6, 2, 8).cuda()
# summary(model, [(1, 160, 192),(1, 160, 192)])
# summary(model, [(1, 32, 32),(1, 32, 32)])
# summary(model, [(1, 160, 192, 224),(1, 160, 192, 224)])
# summary(model, (1, 128,128))#,(1, 80, 96, 112)])
