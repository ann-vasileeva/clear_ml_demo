import logging
import hydra
from clearml import Task
from hydra import compose, initialize
from omegaconf import OmegaConf

#@hydra.main(version_base=None, config_path="configs", config_name="main") 
def prepare_model_setup():
    with initialize(version_base=None, config_path="~/home/vlad/jptr/AnyaMNIST/configs"):
        cfg = compose(config_name="main")
    task = Task.init(project_name="examples", task_name="Pipeline step 2 models")
 
    unet = hydra.utils.instantiate(cfg.unet)    
    unet.train()
    logging.info("UNET instacialized")
    task.upload_artifact('UNET', artifact_object=unet)
    
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=unet.parameters())
    logging.info("Optimizer instacialized")
    task.upload_artifact('Optimizer', artifact_object=optimizer)

    return unet, optimizer

if __name__ == '__main__':
    prepare_model_setup()