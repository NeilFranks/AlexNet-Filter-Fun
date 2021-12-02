import torch
import torch.nn as nn


class FilterPred_11x3(nn.Module):
    """
    Model to predict the convergence of filters with 3 channels, size 11x11

    Given 3 sequential frames, predict the frame it will be in 30 epochs
    """

    def __init__(self) -> None:
        super().__init__()
        self.number_of_sequential_frames_as_input = 3

        self.main = nn.Sequential(
            nn.Conv3d(self.number_of_sequential_frames_as_input,
                      32, kernel_size=4, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 192, kernel_size=4, padding=2),
            nn.BatchNorm3d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, kernel_size=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
