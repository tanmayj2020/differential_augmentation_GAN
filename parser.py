"""
Author : Tanmay 
"""
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name" , 
        help="Name of the experiment")
    parser.add_argument(
        "-a" , 
        "--augment" ,
         action="store_true" ,
          help="If true apply augmenttions")
    parser.add_argument("-b" ,
     "--batch-size" ,
      type=int ,
       default=16 ,
        help="Batch size")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="Adam optimizer hyperparamter",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="Adam optimizer hyperparamter",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=400,
        help="Generate generator images every `eval_frequency` epochs",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=100,
        help="Dimensionality of the random noise",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate"
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=32,
        help="Number of discriminator feature maps (after first convolution)",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=32,
        help="Number of generator feature maps (before last transposed convolution)",
    )
    parser.add_argument(
        "-n",
        "--n-epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--mosaic-size",
        type=int,
        default=10,
        help="Size of the side of the rectangular mosaic",
    )
    parser.add_argument(
        "-p",
        "--prob",
        type=float,
        default=0.9,
        help="Probability of applying an augmentation",
    )
    return parser