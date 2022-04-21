"""
Author : Tanmay Jain 
"""
import torch.nn as nn

def init_weight(module):
    """
    Intialiazing weights by sampling from a normal distribution 
    Note - Weights are modified inplace

    Parameters 
    ----------
    module : nn.Module
        Module with trainable weights
    """
    cls_name = module.__class__.__name__
    if cls_name in {"Conv2d" ,"ConvTranspose2d"}:
        nn.init.normal_(module.weight.data , 0.0 , 0.02)
    elif cls_name == "BatchNorm2d":
        nn.init.normal_(module.weight.data , 1.0 , 0.02)
        nn.init.constant_(module.bias.data , 0.0)

    