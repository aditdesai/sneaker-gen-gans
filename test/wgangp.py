import torch
from torch import nn
import torchvision.transforms.functional as TF


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=16, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


def plot_generated_images():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("WGAN GP/model_200.pt", map_location=torch.device('cpu'))
    model.eval()

    noise = torch.randn(1, 100, 1, 1).to(device)
    fake = model(noise)

    fake_image = TF.to_pil_image((fake[0] * 0.5) + 0.5)

    return fake_image
