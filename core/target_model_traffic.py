import torch

torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F
from torchvision import models

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def resnet18_hybrid(retrain):
    res_net_model = models.resnet18(True)
    num_ftrs = res_net_model.fc.in_features

    res_net_model.fc = nn.Linear(num_ftrs, 14)

    for param in res_net_model.parameters():
        if retrain == True:
            param.requires_grad_ = True
        else:
            param.requires_grad_ = False

    model = res_net_model
    return model
##########

def vit32_hybrid(retrain):
    res_net_model = models.vit_b_32(pretrained=True)
    num_ftrs = res_net_model.heads.head.in_features
    res_net_model.heads.head = nn.Linear(num_ftrs, 14)

    for param in res_net_model.parameters():
        if retrain == True:
            param.requires_grad_ = True
        else:
            param.requires_grad_ = False

    model = res_net_model
    return model
#############

def resnet_50_hybrid(retrain):
    res_net_model = models.resnet50(True)
    num_ftrs = res_net_model.fc.in_features

    res_net_model.fc = nn.Linear(num_ftrs, 14)
    res_net_model.fc.requires_grad_ = True

    for param in res_net_model.parameters():
        if retrain == True:
            param.requires_grad_ = True
        else:
            param.requires_grad_ = False

    model = res_net_model
    return model

def effnet_b0_hybrid(retrain):
    mobile_net_model = models.efficientnet_b0(True)
    num_ftrs = mobile_net_model.classifier[1].in_features

    mobile_net_model.classifier = nn.Linear(num_ftrs, 14)
    mobile_net_model.classifier.requires_grad_ = True

    for param in mobile_net_model.parameters():
        if retrain == True:
            param.requires_grad_ = True
        else:
            param.requires_grad_ = False

    model = mobile_net_model
    return model

def densenet_hybrid(retrain):
    res_net_model = models.densenet121(True)
    num_ftrs = res_net_model.classifier.in_features

    res_net_model.classifier = nn.Linear(num_ftrs, 14)
    res_net_model.classifier.requires_grad_ = True

    for param in res_net_model.parameters():
        if retrain == True:
            param.requires_grad_ = True
        else:
            param.requires_grad_ = False

    model = res_net_model
    return model

