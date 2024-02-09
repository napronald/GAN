import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torchvision.utils import save_image

import model  


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def save_model(self, epoch, D_state, G_state, D_optim_state, G_optim_state):
    torch.save({"epoch": epoch, "D_state": D_state,
                "D_optim_state": D_optim_state}, os.path.join(self.D_path))
    torch.save({"epoch": epoch, "G_state": G_state,
                "G_optim_state": G_optim_state}, os.path.join(self.G_path))


def edge_promoting(image_path, save_path, n):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    rgb_img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    rgb_img = cv2.resize(rgb_img, (32, 32))
    pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
    gray_img = cv2.resize(gray_img, (32, 32))
    
    edges = cv2.Canny(gray_img, 100, 200)
    dilation = cv2.dilate(edges, kernel)

    gauss_img = np.copy(rgb_img)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        for j in range(3):  
            gauss_img[idx[0][i], idx[1][i], j] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, j], gauss))

    result = gauss_img
    cv2.imwrite(os.path.join(save_path, f'{n}.png'), result)  


def generate_edge_smoothed_dataset(csv_file, save_path):
    df = pd.read_csv(csv_file)

    if os.path.isdir(save_path):
        print(f"Edge-smoothed dataset already exists at {save_path}. Skipping generation.")
        return
    else:
        os.makedirs(save_path)
        print(f"Creating {save_path} and starting to generate edge-smoothed dataset...")
        
    existing_files = glob.glob(os.path.join(save_path, '*.png'))
    if existing_files:
        highest_number = max([int(os.path.splitext(os.path.basename(f))[0]) for f in existing_files])
    else:
        highest_number = 0  

    n = highest_number + 1  
    
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        edge_promoting(row['image'], save_path, n)
        n += 1



def generate_images_and_save(G_path, num_images, noise_dim=100, device='cpu'):
    G = model.generator(noise_dim=noise_dim).to(device)
    
    checkpoint = torch.load(G_path, map_location=device)
    G.load_state_dict(checkpoint['G_state'])
    
    G.eval()  
    os.makedirs("generated_images", exist_ok=True)
    
    for i in range(num_images):
        noise = torch.randn(1, noise_dim, 1, 1, device=device)
        
        with torch.no_grad():
            generated_image = G(noise)
        
        generated_image = (generated_image + 1) / 2
        
        img_path = os.path.join("generated_images", f'generated_image_{i}.png')
        save_image(generated_image, img_path)
        print(f"Saved {img_path}")