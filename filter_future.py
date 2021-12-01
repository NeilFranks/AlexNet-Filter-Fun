import os

import torch

from filter_prediction.FilterPred_11x3 import FilterPred_11x3
from filters_utils import MODEL_FOLDER, get_filters_from_layer, plot_filter_multichannels

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model info; which epochs you wants, and which layer
layer_num = 0
last_epoch = 42

torch.set_grad_enabled(True)

# get all your little filters bro
filters = get_filters_from_layer(last_epoch-9, last_epoch+1, layer_num)

# load up the model
criterion = torch.nn.MSELoss()

MODEL_FOLDER = "./filter_models/11x3"
save_every_X_epochs = 150
model = FilterPred_11x3()

checkpoint = torch.load(os.path.join(MODEL_FOLDER, "best.pt"))
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

start_idx = 0
input_filters = [filter[start_idx:start_idx+3]
                 for filter in filters.values()]

inputs = []
groundtruth_outputs = []
for filter_num in range(len(input_filters)):
    inputs.append(torch.stack(input_filters[filter_num]))
inputs = torch.stack(tuple(inputs))
inputs.to(device, dtype=torch.float)

predicted_outputs = model(inputs)

a = 0
