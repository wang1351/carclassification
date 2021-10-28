from torchsummary import summary
from alexnet import AlexNet

alex_net = AlexNet()
summary(alex_net, (3, 600, 420))