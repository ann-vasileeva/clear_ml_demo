import hydra
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification
import seaborn as sns
import numpy as np
import random
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

from diffusion import Diffusion
from clearml import Task, OutputModel, Logger, PipelineController
from clearml.task_parameters import TaskParameters, param, percent_param

import logging


#@hydra.main(version_base=None, config_path="configs", config_name="main") 
def prepare_data_setup(): 
    train_dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last = True)

    test_dataset = torchvision.datasets.MNIST(root="mnist/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last = True)    
    return train_dataloader, test_dataloader

#@hydra.main(version_base=None, config_path="configs", config_name="main") 
def prepare_model_setup(): 
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="main")  
    unet = hydra.utils.instantiate(cfg.unet)    
    unet.train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=unet.parameters())
    lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warm_up_steps,
            num_training_steps=(len(train_dataloader) * cfg.num_epochs),
        )    
    return unet, optimizer, lr_scheduler
    
#@hydra.main(version_base=None, config_path="configs", config_name="main") 
def run_training(data_setup, model_setup):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="main")    
    train_dataloader, test_dataloader = data_setup
    unet, optimizer, lr_scheduler = model_setup
    
    writer = SummaryWriter(log_dir = cfg.log_path)
    diffusion = hydra.utils.instantiate(cfg.diffusion, unet = unet)
    #diffusion.train(cfg.num_epochs, optimizer, lr_scheduler, train_dataloader, writer)
    diffusion.train(0.01, optimizer, lr_scheduler, train_dataloader, writer)
    torch.save({'model' : diffusion.unet.state_dict()}, cfg.ckpt_path)
    output_model = OutputModel(diffusion.unet)
    output_model.update_weights(weights_filename=cfg.ckpt_path)
    return diffusion
    
@hydra.main(version_base=None, config_path="configs", config_name="main") 
def run_pipe(cfg):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    #with initialize(version_base=None, config_path="configs"):
    #    cfg = compose(config_name="main")
        
    pipe = PipelineController(
        project='Image generation Pipe',
        name='Pipeline demo',
        version='1.1',
        add_pipeline_tags=False,
    )

    pipe.set_default_execution_queue('default')

    pipe.add_function_step(
        name='Dataloaders_setup',
        function=prepare_data_setup,
        #function_kwargs=dict(cfg='${pipeline.cfg}'),
        function_return=['data_setup'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='Model_setup',
        parents=['Dataloaders_setup'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=prepare_model_setup,
        #function_kwargs=dict(cfg='${pipeline.cfg}'),
        function_return=['model_setup'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='Diffusion model training',
        parents=['Model_setup'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=run_training,
        function_kwargs=dict(data_setup='${Dataloaders_setup.data_setup}', model_setup='${Model_setup.model_setup}'),
        function_return=['model'],
        cache_executed_step=True,
    )

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    # pipe.start_locally(run_pipeline_steps_locally=False)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    pipe.start_locally(run_pipeline_steps_locally=True)
if __name__ == '__main__': 
    run_pipe()
