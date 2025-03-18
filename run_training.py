import logging
import torchvision
import hydra
import torch
from clearml import Task, OutputModel
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_cosine_schedule_with_warmup
from hydra import compose, initialize
from omegaconf import OmegaConf
#@hydra.main(version_base=None, config_path="configs", config_name="main") 
def run_training():
    
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="main")     
    task = Task.init(project_name="examples", task_name="Pipeline step 3 models")
    
    args = {
    'data_setup_task_id': 'e529350a538d47988f3302e8603a19c1',
    'model_setup_task_id': 'bc8bb65819824fec945f4516d656c02e',
    }
    task.connect(args)

    data_setup_task = Task.get_task(task_id=args['data_setup_task_id'])
    model_setup_task = Task.get_task(task_id=args['model_setup_task_id'])
    
    train_dataloader = data_setup_task.artifacts['train_dataloader'].get()
    test_dataloader = data_setup_task.artifacts['test_dataloader'].get()
    
    unet = model_setup_task.artifacts['UNET'].get()
    optimizer = model_setup_task.artifacts['Optimizer'].get()
    
    lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.warm_up_steps,
            num_training_steps=(len(train_dataloader) * cfg.num_epochs),
        )   
    
    writer = SummaryWriter(log_dir = cfg.log_path)
    diffusion = hydra.utils.instantiate(cfg.diffusion, unet = unet)
    #diffusion.train(cfg.num_epochs, optimizer, lr_scheduler, train_dataloader, writer)
    #diffusion.train(0.01, optimizer, lr_scheduler, train_dataloader, writer)
    torch.save({'model' : diffusion.unet.state_dict()}, cfg.ckpt_path)
    output_model = OutputModel(task = task)
    output_model.update_weights(weights_filename=cfg.ckpt_path)
    return diffusion

if __name__ == '__main__':
    run_training()