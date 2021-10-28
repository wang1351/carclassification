import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import pdb


class ResNet18(torch.nn.Module):
  def __init__(self,n_classes):
    super(ResNet18, self).__init__()

    self.base_model = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
    base_model_out_size = list(self.base_model.parameters())[-1].size(0)
    self.preds = torch.nn.Linear(base_model_out_size, n_classes)

  def forward(self, images):
    features = self.base_model(images).view(images.shape[0],-1)
    return self.preds(features), features

