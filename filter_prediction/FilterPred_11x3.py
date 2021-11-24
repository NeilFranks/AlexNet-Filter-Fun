import torch
import torch.nn as nn


class FilterPred_11x3(nn.Module):
    """
    Model to predict the convergence of filters with 3 channels, size 11x11

    Given 3 sequential frames, predict the 4th frame
    """

    def __init__(self) -> None:
        super().__init__()
        self.number_of_sequential_frames_as_input = 3
        self.stretched_length_of_one_frame = 3*11*11

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 192, kernel_size=4, padding=2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, self.stretched_length_of_one_frame)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
