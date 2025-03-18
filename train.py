

import hydra
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification
import seaborn as sns
import numpy as np

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import trange, tqdm
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel, DDPMPipeline, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from matplotlib import pyplot as plt

from diffusion import Diffusion, set_all_seeds

@hydra.main(version_base=None, config_path="configs", config_name="main") 
def run_training():
    set_all_seeds(cfg.seed)
    train_dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last = True)

    test_dataset = torchvision.datasets.MNIST(root="mnist/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last = True)

    
    model = UNet2DModel(
            sample_size=(28, 28) , 
            in_channels=1,  
            out_channels=1, 
            layers_per_block=2, 
            block_out_channels=(64, 128, 256), 
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  
                "UpBlock2D",
                "UpBlock2D",
              ),
        )
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warm_up_steps,
            num_training_steps=(len(train_dataloader) * cfg.num_epochs),
        )
    
    writer = SummaryWriter(log_dir = cfg.log_path)
    diffusion = Diffusion(cfg.beta, cfg.T, model)
    diffusion.train(cfg.num_epochs, optimizer, lr_scheduler, train_dataloader, writer)
    torch.save({'model' : diffusion.model.state_dict()}, cfg.ckpt_path)
