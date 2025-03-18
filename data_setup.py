import logging
import torchvision
import hydra
import torch
from clearml import Task

def prepare_data_setup(): 
    task = Task.init(project_name="examples", task_name="Pipeline step 1 dataset artifact")
    train_dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last = True)
    logging.info("train_dataloader uploaded")
    test_dataset = torchvision.datasets.MNIST(root="mnist/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last = True) 
    
    logging.info("test_dataloader uploaded")
    task.upload_artifact('train_dataloader', artifact_object=train_dataloader)
    task.upload_artifact('test_dataloader', artifact_object=test_dataloader)
    logging.info("Loader uploaded as artifacts")
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    prepare_data_setup()

