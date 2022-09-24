import torch
import os
import torchvision
from torchvision import transforms
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((31, 69)),
    transforms.ToTensor()
])

def load_prior(img_path='../../../prior/'):
    data = torchvision.datasets.ImageFolder(root=img_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=True, num_workers=1)
    data_iter = iter(data_loader)
    data = next(data_iter)
    return data

def compute_loss(pred_prior, tgt_prior):
    loss = nn.MSELoss()
    prior_loss = loss(pred_prior, tgt_prior)
    return prior_loss#torch.sum(prior - pred_prior)
