import sys

import yaml

import torch
import torch.nn as nn

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.optim import Adam

import numpy as np

import matplotlib.pyplot as plt


default_conf_path = './config/default.yaml'


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.FC_input  = nn.Linear(input_dim,  hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean   = nn.Linear(hidden_dim, latent_dim)
        self.FC_var    = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h       = self.LeakyReLU(self.FC_input(x))
        h       = self.LeakyReLU(self.FC_input2(h))
        mean    = self.FC_mean(h)
        log_var = self.FC_var(h)

        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()

        self.FC_hidden  = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output  = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        x_rec = torch.sigmoid(self.FC_output(h))

        return x_rec

class Model(nn.Module):
    def __init__(self, encoder, decoder, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_rec = self.decoder(z)
        return x_rec, mean, log_var


def main(conf):
    cuda = conf['cuda']
    dataset_path = conf['dataset_path']
    batch_size = conf['batch_size']
    x_dim = conf['x_dim']
    hidden_dim = conf['hidden_dim']
    latent_dim = conf['latent_dim']
    lr = conf['lr']
    epochs = conf['epochs']

    device = 'cuda' if cuda else 'cpu'

    # データセット 読み込み
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True,  download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True, **kwargs)

    # モデル 定義
    encoder = Encoder(x_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, x_dim)

    model = Model(encoder, decoder, device=device).to(device)

    # ロス関数 定義
    def loss_function(x, x_rec, mean, log_var):
        rec_loss = nn.functional.binary_cross_entropy(x_rec, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return rec_loss + kl

    optimizer = Adam(model.parameters(), lr=lr)

    # 学習
    model.train()

    for epoch in range(epochs):
        loss_total = 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_rec, mean, log_var = model(x)
            loss = loss_function(x, x_rec, mean, log_var)

            loss_total += loss.item()

            loss.backward()
            optimizer.step()

        print(f'[Epoch {epoch + 1}] Ave Loss: {loss_total / (batch_idx * batch_size)}')

    # 評価
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(device)

            x_rec, _, _ = model(x)

            break

    def show_image(x, idx):
        x = x.view(batch_size, 28, 28)
        plt.imshow(x[idx].cpu().numpy())
        plt.show()

    show_image(x, idx=0)
    show_image(x_rec, idx=0)

    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(device)
        generated_images = decoder(noise)

    show_image(generated_images, idx=12)
    show_image(generated_images, idx=0)
    show_image(generated_images, idx=1)
    show_image(generated_images, idx=10)
    show_image(generated_images, idx=20)
    show_image(generated_images, idx=50)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('引数足りない')

    exp_conf_path = sys.argv[1]

    with open(default_conf_path) as f:
        conf = yaml.safe_load(f)

    with open(exp_conf_path) as f:
        conf.update(yaml.safe_load(f))

    print(conf)

    main(conf)

