import torch

from filter_prediction.FilterPred_11x3 import FilterPred_11x3
from filters_utils import get_filters_from_layer

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FilterPred_11x3()

filters = get_filters_from_layer(4, 7, 0)

input1 = torch.stack(filters[0])
inputs = torch.stack((input1, input1))
inputs.to(device, dtype=torch.float)

outputs = model(inputs)
b=0
