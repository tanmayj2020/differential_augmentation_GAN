import argparse
from fnmatch import translate
from genericpath import exists
from math import degrees
import pathlib
import pprint
from datetime import datetime

from cv2 import transform

import kornia.augmentation as K
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import tqdm 

from utils import init_weight
from network import Generator , Discriminator
from dataset import DatasetImages
from parser import get_parser


def main():
    args = get_parser().parse_args()
    args_dictionary = vars(args)
    print(args)

    img_size = 128

    # Additional Parameters 
    device = torch.device(args.device)
    mosaic_kwargs = {"nrow": args.mosaic_size, "normalize": True}
    n_mosaic_cells = args.mosaic_size * args.mosaic_size
    sample_showcase_ix = (
        0  # this one will be used to demonstrate the augmentations
    )

    # Differentiable augmentation module
    dif_augment_module = torch.nn.Sequential(
        K.RandomAffine(degrees= 0 , translate = (1/8 , 1/8), p = args.prob),
        K.RandomErasing((0.0 , 0.5) , p = args.prob),
    )

    #Loss Function 
    adversarial_loss = torch.nn.BCELoss()

    #Initialize generator and discriminator
    generator = Generator(args.latent_dim , args.ngf)
    discriminator = Discriminator(args.ndf , dif_augment_module if args.augment else None)
    
    # Sending to device
    generator.to(device)
    discriminator.to(device)

    #Intialize weights
    generator.apply(init_weight)
    discriminator.apply(init_weight)

    # Configure data loader
    data_path = pathlib.Path("data")
    tform= transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            #Normalized between -1 to 1 (tanh function)
            transforms.Normalize([0.5 , 0.5 , 0.5] , [0.5 , 0.5 , 0.5]),
        ]
    )

    dataset = DatasetImages(data_path , tform)
    dataloader = DataLoader(
        dataset,
        batch_size = args.batch_size , 
        shuffle=True
    )
    optimizer_G = torch.optim.Adam(
        generator.parameters() , lr = args.lr , betas=(args.b1 , args.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters() , lr = args.lr , betas = (args.b1 , args.b2)
    )
    #Output path and metadata 
    output_path = pathlib.Path("outputs")/args.name
    output_path.mkdir(exists_ok = True , parents= True)

    # Add other parameters(not included in CLI)
    args_dictionary["time"] = datetime.now()
    args_dictionary["kornia"] = str(dif_augment_module)
    
    #Prepare tensorboard writer
    writer = SummaryWriter(output_path)

    #Log hyperparameters as text
    writer.add_text("hyperparameter" , pprint.pformat(args_dictionary).replace(
        "\n" ," \n"
    ) , 
    0 ,
    )

    # # Log true data
    writer.add_image(
        "true_data",
        make_grid(
            torch.stack([dataset[i] for i in range(n_mosaic_cells)]),
            **mosaic_kwargs
        ),
        0,
    )
    # Log augmented data
    batch_showcase = dataset[sample_showcase_ix][None, ...].repeat(
        n_mosaic_cells, 1, 1, 1
    )
    batch_showcase_aug = discriminator.augment_module(batch_showcase)
    writer.add_image(
        "augmentations", make_grid(batch_showcase_aug, **mosaic_kwargs), 0
    )
    # Prepate evaluation noise
    z_eval = torch.randn(n_mosaic_cells, args.latent_dim).to(device)

    for epoch in tqdm(range(args.n_epochs)):
        for i, imgs in enumerate(dataloader):
            n_samples , *_ = imgs.shape
            batches_done = epoch * len(dataloader) + i
            # Adverial ground truth 
            valid = 0.9 * torch.ones(n_samples , 1, device= device , dtype=torch.float32)
            fake = torch.zeros(n_samples , 1 , device=device , dtype=torch.float32)
             # D preparation
            optimizer_D.zero_grad()

            # D loss on reals
            real_imgs = imgs.to(device)
            d_x = discriminator(real_imgs)
            real_loss = adversarial_loss(d_x, valid)
            real_loss.backward()

            # D loss on fakes
            z = torch.randn(n_samples, args.latent_dim).to(device)
            gen_imgs = generator(z)
            d_g_z1 = discriminator(gen_imgs.detach())

            fake_loss = adversarial_loss(d_g_z1, fake)
            fake_loss.backward()

            optimizer_D.step()  # we called backward twice, the result is a sum

            # G preparation
            optimizer_G.zero_grad()

            # G loss
            d_g_z2 = discriminator(gen_imgs)
            g_loss = adversarial_loss(d_g_z2, valid)

            g_loss.backward()
            optimizer_G.step()

            # Logging
            if batches_done % 50 == 0:
                writer.add_scalar("d_x", d_x.mean().item(), batches_done)
                writer.add_scalar("d_g_z1", d_g_z1.mean().item(), batches_done)
                writer.add_scalar("d_g_z2", d_g_z2.mean().item(), batches_done)
                writer.add_scalar(
                    "D_loss", (real_loss + fake_loss).item(), batches_done
                )
                writer.add_scalar("G_loss", g_loss.item(), batches_done)

            if epoch % args.eval_frequency == 0 and i == 0:
                generator.eval()
                discriminator.eval()

                # Generate fake images
                gen_imgs_eval = generator(z_eval)

                # Generate nice mosaic
                writer.add_image(
                    "fake",
                    make_grid(gen_imgs_eval.data, **mosaic_kwargs),
                    batches_done,
                )

                # Save checkpoint (and potentially overwrite an existing one)
                torch.save(generator, output_path / "model.pt")

                # Make sure generator and discriminator in the training mode
                generator.train()
                discriminator.train()

if __name__ =="__main__":
    main()