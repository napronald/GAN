import os
import torch
import numpy as np
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset

import model
import utils
from dataset import MyDataset

class Experiment:
    def __init__(self, args):
        self.root = os.path.abspath(args.dataset_root)
        self.num_epoch = args.num_epoch
        self.warmup_epoch = args.warmup_epoch
        self.batch_size = args.batch_size
        self.G_path = args.G_path
        self.D_path = args.D_path

        edge_smoothed_path = os.path.join(self.root, "edge_smoothed")

        utils.generate_edge_smoothed_dataset("dataset.csv", edge_smoothed_path)

        train_real_dataset = MyDataset(self.root, style="real", mode="train")
        train_edge_dataset = MyDataset(self.root, style="edge_smoothed", mode="train")
        val_real_dataset = MyDataset(self.root, style="real", mode="test")

        self.train_real_loader = DataLoader(train_real_dataset, batch_size=self.batch_size, shuffle=True, num_workers=28)
        self.train_edge_loader = DataLoader(train_edge_dataset, batch_size=self.batch_size, shuffle=True, num_workers=28)
        self.val_real_loader = DataLoader(val_real_dataset, batch_size=self.batch_size, shuffle=True, num_workers=28)

        self.combined_real_loader = DataLoader(ConcatDataset([train_real_dataset, val_real_dataset]), batch_size=self.batch_size, shuffle=True, num_workers=28)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.D = model.discriminator().to(self.device)
        self.G = model.generator().to(self.device)

        self.vgg19 = torchvision.models.vgg19(pretrained=True).features.to(self.device).eval()

        self.D_optimizer = optim.Adam(self.D.parameters(), args.D_lr, betas=(0.5, 0.99))
        self.G_optimizer = optim.Adam(self.G.parameters(), args.G_lr, betas=(0.5, 0.99))

        self.BCE_loss = nn.BCELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)
        self.content_loss_lambda = args.content_loss_lambda

        self.D_scheduler = MultiStepLR(self.D_optimizer, milestones=args.D_step, gamma=args.D_gamma)
        self.G_scheduler = MultiStepLR(self.G_optimizer, milestones=args.G_step, gamma=args.G_gamma)


    def train(self, e):
        self.D.train()
        self.G.train()

        D_losses = []
        G_losses = []
        Perceptual_losses = []  

        for i, (real_imgs, edge_smoothed_imgs) in enumerate(zip(self.combined_real_loader, self.train_edge_loader)):
            noise = torch.randn(real_imgs.size(0), 100).to(self.device) 
            edge_smoothed_imgs = edge_smoothed_imgs.to(self.device)

            D_real = self.D(edge_smoothed_imgs)
            D_real_loss = self.BCE_loss(D_real, torch.ones_like(D_real, device=self.device))

            fake_imgs = self.G(noise)
            D_fake = self.D(fake_imgs)
            D_fake_loss = self.BCE_loss(D_fake, torch.zeros_like(D_fake, device=self.device))

            D_loss = D_real_loss + D_fake_loss
            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()

            fake_imgs = self.G(noise)
            D_fake = self.D(fake_imgs)
            G_loss_adv = self.BCE_loss(D_fake, torch.ones_like(D_fake, device=self.device)) 

            fake_imgs_rgb = fake_imgs.repeat(1, 3, 1, 1)  
            edge_smoothed_imgs_rgb = edge_smoothed_imgs.repeat(1, 3, 1, 1)  

            fake_imgs_rgb_normalized = (fake_imgs_rgb + 1) / 2  
            edge_smoothed_imgs_rgb_normalized = (edge_smoothed_imgs_rgb + 1) / 2  

            fake_imgs_features = self.vgg19(fake_imgs_rgb_normalized)
            edge_smoothed_imgs_features = self.vgg19(edge_smoothed_imgs_rgb_normalized)

            min_size = min(fake_imgs_features.size(0), edge_smoothed_imgs_features.size(0))
            fake_imgs_features = fake_imgs_features[:min_size]
            edge_smoothed_imgs_features = edge_smoothed_imgs_features[:min_size]

            perceptual_loss = self.content_loss_lambda * self.L1_loss(fake_imgs_features, edge_smoothed_imgs_features.detach())

            G_loss = G_loss_adv + perceptual_loss

            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())
            Perceptual_losses.append(perceptual_loss.item())

        average_D_loss = np.mean(D_losses)
        average_G_loss = np.mean(G_losses)
        average_perceptual_loss = np.mean(Perceptual_losses)
        print(f"Epoch: {e}, Average D loss: {average_D_loss:.3f}, G loss: {average_G_loss:.3f}, Perceptual loss: {average_perceptual_loss:.3f}")

        self.G_scheduler.step()
        self.D_scheduler.step()
        return average_D_loss, average_G_loss, average_perceptual_loss



    def train_warming(self, e):
        self.G.train()

        Content_losses = []

        for i, real_imgs in enumerate(self.combined_real_loader):
            real_imgs = real_imgs.to(self.device)

            noise = torch.randn(real_imgs.size(0), 100).to(self.device) 

            fake_imgs = self.G(noise)

            fake_imgs_rgb = fake_imgs.repeat(1, 3, 1, 1)  
            real_imgs_rgb = real_imgs.repeat(1, 3, 1, 1) 

            fake_imgs_rgb_normalized = (fake_imgs_rgb + 1) / 2  
            real_imgs_rgb_normalized = (real_imgs_rgb + 1) / 2  

            fake_imgs_features = self.vgg19(fake_imgs_rgb_normalized)
            real_imgs_features = self.vgg19(real_imgs_rgb_normalized)
            content_loss = self.content_loss_lambda * self.L1_loss(fake_imgs_features, real_imgs_features.detach())

            self.G_optimizer.zero_grad()
            content_loss.backward()
            self.G_optimizer.step()

            Content_losses.append(content_loss.item())

        average_content_loss = np.mean(Content_losses)
        print(f"\nEpoch: {e}, Average content loss: {average_content_loss:.3f}\n")
        return average_content_loss



    def valid(self, e):
        self.save_path = os.path.join(os.getcwd(), 'images')
        os.makedirs(self.save_path, exist_ok=True)

        grid_size = 5
        image_size = 32  

        grid_image = Image.new('RGB', (image_size * grid_size, image_size * grid_size))

        with torch.no_grad():
            self.G.eval()
            for i in range(25):
                noise = torch.randn(1, 100).to(self.device) 
                generated_img = self.G(noise)

                gen_img = ((generated_img.cpu().numpy().squeeze() + 1) / 2 * 255).astype(np.uint8)
                gen_img_pil = Image.fromarray(gen_img)

                row = i // grid_size
                col = i % grid_size
                grid_image.paste(gen_img_pil, (col * image_size, row * image_size))

        grid_filename = f"generated_grid_{e}.png"
        grid_path = os.path.join(self.save_path, grid_filename)
        grid_image.save(grid_path)


    def run(self):
        warm_up_content_losses = []     
        training_D_losses = []
        training_G_losses = []
        training_content_losses = []

        print("Warming Up")
        for e in range(self.warmup_epoch):
            curr_content_loss = self.train_warming(e)
            warm_up_content_losses.append(curr_content_loss)
            self.valid(e)

        print("Training and Validating")
        for e in range(self.num_epoch):
            curr_D_loss, curr_G_loss, curr_content_loss = self.train(e)
            training_D_losses.append(curr_D_loss)
            training_G_losses.append(curr_G_loss)
            training_content_losses.append(curr_content_loss)
            self.valid(e)

        utils.save_model(self.num_epoch, self.D.state_dict(), self.G.state_dict(), self.D_optimizer.state_dict(), self.G_optimizer.state_dict())
        return warm_up_content_losses, training_D_losses, training_G_losses, training_content_losses