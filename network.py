"""
Author : Tanmay 
"""
from turtle import forward
import torch 
import torch.nn as nn



class Generator(nn.Module):
    """
    Generator Network 
    Parameters
    ----------
    latent_dim : int 
        The dimensionality of the input noise
    
    ngf : int 
        Number of generator filters , actual filters will be a multiple of this number
    Attributes
    ----------
    main : torch.Seqeuntial 
        The actual network composed of 'ConvTranspose2d" , 'BatchNormalization' , 'ReLU' blocks
    """ 
    def __init__(self , latent_dim , ngf = 64):
        super().__init__()
        self.main = nn.Sequential(
             nn.ConvTranspose2d(latent_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # (ngf * 16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf * 8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf * 4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf * 2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 64 x 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 3 x 128 x 128
        )

    def forward(self ,x):
        """
        Run the forward pass
        Parameters
        ----------
        x : torch.Tensor    
            Input noise of shape '(n_samples , latent_dim).'
        
        Returns
        -------
        torch.Tensor
            Generated image of shape '(n_samples , 3 , 128 , 128).'
        """
        x = x.reshape(*x.shape , 1 , 1)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator Network 
    Parameters
    ----------
    ndf : int 
        Number of discriminator filters.  Number of filtrs after first convolution block.
    augment_module : nn.Module or None
        If provided it represents the Kornia module that performs differential augmentation of the images
    Attributes
    ----------
    augment_module : nn.Module
        If the input paramter 'augment_module' then this is the same thing else the identity mapping     
    """
    def __init__(self , ndf , augment_module):
        super().__init__()
        self.main = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )
        if augment_module is not None:
            self.augment_module = augment_module
        else:
            self.augment_module = nn.Identity()
    def forward(self ,x):
        """
        Run the forward pass 
        Parameters 
        ---------
        x : torch.Tensor 
            Input images of shape (n_samples , 3 , 128 , 128)
        Returns
        -------
        torch.Tensor 
            Classification output of shape (n_samples , 1 ) 
        """
        if self.training:
            # Augmenting everywhere
            x = self.augment_module(x)
        x = self.main(x)
        x = x.squeeze()[: , None]
        return x
        