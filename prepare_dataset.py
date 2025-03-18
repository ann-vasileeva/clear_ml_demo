import argparse
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from clearml import Task, Dataset
import os

def create_mnist_dataloaders(batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_dataloader, test_dataloader

def main(args):
    task = Task.init(
        project_name="MNIST Diffusion Basic",
        task_name="data_preparation",
        auto_connect_frameworks={'torchvision': True}
    )
    
    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    data_dict = {
        'train_data': [batch for batch in train_dataloader],
        'test_data': [batch for batch in test_dataloader]
    }
    
    data_path = './mnist_data.pt'
    torch.save(data_dict, data_path)
    
    task.upload_artifact(name='mnist_data', artifact_object=data_path)
    
    dataset = Dataset.create(
        dataset_name='mnist_diffusion_data',
        dataset_project='MNIST Diffusion Basic'
    )
    dataset.add_files(data_path)
    dataset.upload()
    dataset.finalize()
    
    print(f"Data preparation completed. Dataset ID: {dataset.id}")
    task.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=28)
    args = parser.parse_args()
    main(args)