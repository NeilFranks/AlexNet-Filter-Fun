import os

import torch

from model import initialize_model

MODEL_FOLDER = "./model_from_scratch"

checkpoint = torch.load(os.join(MODEL_FOLDER, "best.pt"))

