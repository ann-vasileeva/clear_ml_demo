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
from matplotlib import pyplot as plt



class Diffusion:
  """
  Класс для работы с диффузионными моделями.
  Позволяет производить как обучение, так и семплирование.
  """
  def __init__(self, beta_start, beta_end, diffusion_steps, conditional, unet):
    """
    alpha = ()
    beta = ()
    """
    self.device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    # Модель
    self.unet = unet.to(self.device)
    # Лосс
    self.criterion = nn.MSELoss()
    # Параметры диффузии
    self.beta = (beta_start, beta_end)
    self.diffusion_steps = diffusion_steps
    self.conditional = conditional

  @property
  def alphas_hat(self):
    """
    Подсчет \alpha_hat
    """
    return  torch.cumprod(self.alphas, dim=0)

  @property
  def alphas(self):
    """
    Подсчет \alpha
    """
    return 1. - self.betas

  @property
  def betas(self):
    """
    Подсчет \beta
    """
    beta_start, beta_end = self.beta
    beta = torch.linspace(beta_start, beta_end, self.diffusion_steps, device = self.device)
    return beta


  def get_noised_images(self, images, noise, steps):
    """
    Получение зашумденного изображения на нужном шагк
    """
    sqrt_alpha_hat = torch.sqrt(self.alphas_hat[steps])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[steps])[:, None, None, None]
    noised_images = sqrt_alpha_hat * images + sqrt_one_minus_alpha_hat * noise

    return noised_images

  def train(self, num_epochs, optimizer, scheduler, dataloader, writer):
    """
    Обучение
    """
    batch_size = dataloader.batch_size

    for epoch in trange(num_epochs):
      for i, batch in enumerate(tqdm(dataloader)):

        # Картинки, метки
        images, labels = batch
        images = images.to(self.device)

        # Эмбеддинг меток
        labels_embedded = torch.nn.functional.one_hot( labels.unsqueeze(-1), 11).to(self.device).float()
        # Генерим картинки
        steps = torch.randint(low=1, high=self.diffusion_steps, size=(batch_size, ) )
        noise = torch.randn_like(images, device = self.device)
        noised_images = self.get_noised_images(images, noise, steps)

        # Подаем метки или нет
        if self.conditional:
          predicted_noise = self.unet(noised_images, steps.to(self.device),  labels_embedded, labels.unsqueeze(-1) ).sample
        else:
          predicted_noise = self.unet(noised_images, steps.to(self.device)).sample
        # Лосс, шаг оптимизатора
        loss = self.criterion(noise, predicted_noise)
        writer.add_scalar("Loss/MSE", loss.item(), i + epoch * len(dataloader))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
       if epoch % 2 == 0:
       	sampled_images = self.sample_images(2)
       	sampled_images  = sampled_images.cpu().numpy()
        cv2.imwrite(sampled_images, f"sampled_images_step{i}.png")
       

  def sample_images_one_step(self, noised_images, predicted_noise, steps, add_noise=False):
    """
    Генерит картинку на нужном шаге
    """
    if add_noise:
        noise = torch.randn_like(noised_images)
    else:
        noise = torch.zeros_like(noised_images)


    alpha_step = self.alphas[steps][:, None, None, None]
    alpha_hat_step = self.alphas_hat[steps][:, None, None, None]
    beta_step = self.betas[steps][:, None, None, None]

    return 1 / torch.sqrt(alpha_step) \
        * (noised_images - (1 - alpha_step) / (torch.sqrt(1 - alpha_hat_step)) * predicted_noise) \
        + torch.sqrt(beta_step) * noise

  def sample_images(self, num_images, start_step=1000, finish_step=1, noised_images=None, labels = None):
    """
    Пайплайн генерации
    """
    if noised_images is None:
        noised_images = torch.randn((num_images, 1, 28, 28), device = self.device)

    labels_embedded = torch.nn.functional.one_hot( labels.unsqueeze(-1), 11).to(self.device).float()

    for i in tqdm(reversed(range(finish_step, start_step)), position=0, total=start_step - finish_step):

        steps = (torch.ones(num_images) * i).long()
        with torch.no_grad():
          if self.conditional:

            predicted_noise = self.unet(noised_images, steps.to(self.device), labels_embedded, labels.unsqueeze(-1)).sample
          else:
            predicted_noise = self.unet(noised_images, steps.to(self.device)).sample

        add_noise = True if i > 1 else False
        noised_images = self.sample_images_one_step(noised_images, predicted_noise, steps, add_noise=add_noise)
    return noised_images


  def sampling_pipline(self, num_steps, strategy, num_images=64):
    """
    Сэмплирование с помощью diffusers
    """
    pipline_mapping = {
        'DDPM' : DDPMPipeline,
        'DDIM' : DDIMPipeline
    }

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=self.diffusion_steps, beta_start=self.beta[0], beta_end=self.beta[1])

    generator = torch.Generator(device=self.device).manual_seed(42)
    pipeline = pipline_mapping[strategy](unet=self.unet, scheduler=noise_scheduler)
    images = pipeline(batch_size=num_images, generator=generator, num_inference_steps=num_steps).images
    return images
