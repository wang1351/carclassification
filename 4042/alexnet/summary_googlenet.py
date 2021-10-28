from torchsummary import summary
from googlenet import GoogLeNet

google_net = GoogLeNet()
summary(google_net, (3, 600, 420))