import torch

from filter_prediction.FilterPred_11x3 import FilterPred_11x3
from filters_utils import get_filters_from_layer, plot_filter_multichannels

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

layer_num = 0

model = FilterPred_11x3()
model.train()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.0005)

torch.set_grad_enabled(True)

start_epoch = 4
end_epoch = 42

input_filters = get_filters_from_layer(4, 7, layer_num)
groundtruth_outputs_filters = get_filters_from_layer(7, 8, layer_num)

inputs = []
groundtruth_outputs = []
for filter_num in range(len(input_filters)):
    inputs.append(torch.stack(input_filters[filter_num]))
    groundtruth_outputs.append(torch.stack(
        groundtruth_outputs_filters[filter_num]))
inputs = torch.stack(tuple(inputs))
inputs.to(device, dtype=torch.float)

groundtruth_outputs = torch.stack(tuple(groundtruth_outputs))
groundtruth_outputs.to(device, dtype=torch.float)

for _ in range(500):
    optimizer.zero_grad()

    predicted_outputs = model(inputs)

    loss = criterion(torch.squeeze(predicted_outputs),
                     torch.squeeze(groundtruth_outputs))
    loss.backward()

    optimizer.step()

    print(loss.item())

b = 0
