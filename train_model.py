import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import make_grid
import math
import os
from clearml import Task, Dataset
from model import MNISTDiffusion
from utils import ExponentialMovingAverage

def load_data(dataset_id):
    dataset = Dataset.get(dataset_id=dataset_id)
    local_path = dataset.get_local_copy()
    data_dict = torch.load(os.path.join(local_path, 'mnist_data.pt'))
    return data_dict['train_data'], data_dict['test_data']

def main(args):
    task = Task.init(
        project_name="MNIST Diffusion Basic",
        task_name="train_diffusion",
        auto_connect_frameworks={'torchvision': True}
    )
    logger = task.get_logger()
    writer = SummaryWriter()

    device = "cpu" if args.cpu else "cuda"
    
    train_data, test_data = load_data(args.dataset_id)
    
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2,4]
    ).to(device)

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer, args.lr,
        total_steps=args.epochs * len(train_data),
        pct_start=0.25,
        anneal_strategy='cos'
    )
    loss_fn = nn.MSELoss(reduction='mean')

    if args.ckpt:
        previous_task = Task.get_task(task_id=args.ckpt)
        checkpoint_path = previous_task.artifacts["checkpoint"].get_local_copy()
        ckpt = torch.load(checkpoint_path)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    for epoch in range(args.epochs):
        model.train()
        for j, (image, target) in enumerate(train_data):
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            
            if j % args.log_freq == 0:
                writer.add_scalar('Training loss', loss.detach().cpu().item(), global_steps)
                writer.add_scalar('Lr', scheduler.get_last_lr()[0], global_steps)
            global_steps += 1

        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }
        task.upload_artifact(name=f"checkpoint_{global_steps}", artifact_object=ckpt)

        model_ema.eval()
        samples = model_ema.module.sampling(
            args.n_samples,
            clipped_reverse_diffusion=not args.no_clip,
            device=device
        )
        grid = make_grid(samples, nrow=int(math.sqrt(args.n_samples)), normalize=True)
        logger.report_image(
            title="Generated Samples",
            series="Steps",
            iteration=global_steps,
            image=grid.permute(1, 2, 0).cpu().numpy()
        )

    writer.close()
    task.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--model_base_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--model_ema_steps', type=int, default=10)
    parser.add_argument('--model_ema_decay', type=float, default=0.995)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--no_clip', action='store_true')
    args = parser.parse_args()
    main(args)