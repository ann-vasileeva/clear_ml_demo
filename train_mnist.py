import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
from clearml import Task
from torch.utils.tensorboard import SummaryWriter

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args


def main(args):
    task = Task.init(project_name="MNIST Diffusion Basic", task_name="train diffusion", auto_connect_frameworks={'torchvision':True})
    logger = task.get_logger()
    writer = SummaryWriter()

    device="cpu" if args.cpu else "cuda"
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #torchvision ema setting
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    #load checkpoint
    if args.ckpt:
        previous_task = Task.get_task(task_id=args.ckpt)
        checkpoint_path = previous_task.artifacts["checkpoint"].get_local_copy()
        ckpt = torch.load(checkpoint_path)
        
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps=0
    for epoch in range(args.epochs):
        model.train()
        for j,(image,target) in enumerate(train_dataloader):
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            pred=model(image,noise)
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            if j%args.log_freq==0:
                writer.add_scalar('Training loss', loss.detach().cpu().item(), global_steps)
                writer.add_scalar('Lr', scheduler.get_last_lr()[0], global_steps)
            global_steps+=1

        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}
        task.upload_artifact(name=f"checkpoint_{global_steps}", artifact_object=ckpt)

        model_ema.eval()
        samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        grid = make_grid(samples, nrow=int(math.sqrt(args.n_samples)), normalize=True)
        logger.report_image(
            title="Generated Samples",
            series="Steps",
            iteration=global_steps,
            image=grid.permute(1, 2, 0).cpu().numpy()  
        )

    writer.close()
    task.close()

if __name__=="__main__":
    args=parse_args()
    main(args)