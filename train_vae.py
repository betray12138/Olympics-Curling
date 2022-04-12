from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
from tensorboardX import SummaryWriter
import numpy as np

class MyData(Dataset):
    def __init__(self, root_dir):
        self.path = root_dir
        self.img_path = os.listdir(self.path)

    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_item_path) / 255.0
        return torch.FloatTensor(img).permute(2, 0, 1)

    def __len__(self):
        return len(self.img_path)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(6, 6), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=(6, 6), stride=(2, 2)),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # output 128 size
        z = self.decode(z)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KL divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():

    batch_size = 128

    dataset = MyData("/home/autolab/hhhz/front_camera")
    writer = SummaryWriter("./carla")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(image_channels=3).to(device)
    #model.load_state_dict(torch.load('vae.torch'))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3000
    step = 0

    for epoch in range(epochs):
        for idx, images in enumerate(dataloader):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                    epochs, loss.item()/batch_size, bce.item()/batch_size, kld.item()/batch_size)
            writer.add_scalar("loss", loss.item() / batch_size, step)
            writer.add_scalar("binary_loss_entrypy", bce.item() / batch_size, step)
            writer.add_scalar("kld", kld.item() / batch_size, step)
            
            print(to_print)
        if epoch >= 500 and epoch % 100 == 0:
            torch.save(model.state_dict(), 'vae_front/vae_' + str(epoch))

    writer.close()

def test():
    with torch.no_grad():
        model = VAE(image_channels=3).to(device)
        model.load_state_dict(torch.load('carla/vae_1900'))
        images = cv2.imread("/home/autolab/hhhz/data/96053.png") / 255.0
        tensor_images = torch.FloatTensor(images).permute(2, 0, 1).view(1, 3, 64, 64).to(device)
        recon_image, _, _ = model(tensor_images)#
        recon_images = recon_image.squeeze().permute(1, 2, 0).cpu().numpy()
        cv2.imshow("old", images)
        cv2.imshow("new", recon_images)
        cv2.waitKey()


class PretrainVAE(nn.Module):
    def __init__(self) -> None:
        super(PretrainVAE, self).__init__()
        self.vae = VAE(image_channels=3).to(device)
        self.vae.load_state_dict(torch.load('vae/vae_rgb'))
    
    def forward(self, rgb_img):
        with torch.no_grad():
            z = torch.FloatTensor(rgb_img).permute(2, 0, 1).view(1, 3, 64, 64).to(device)
            z, _, _ = self.vae.encode(z)
            z = z.squeeze().cpu().numpy()
        return z
        

train()