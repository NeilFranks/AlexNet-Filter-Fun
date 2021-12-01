import random
import os

import torch

from filter_prediction.FilterPred_11x3 import FilterPred_11x3
from filters_utils import get_filters_from_layer, plot_filter_multichannels

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model info; which epochs you wants, and which layer
layer_num = 0
start_epoch = 4
end_epoch = 43

model = FilterPred_11x3()
model.train()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.0005)

torch.set_grad_enabled(True)

# get all your little filters bro
filters = get_filters_from_layer(start_epoch, end_epoch, layer_num)

num_epochs = 0
save_every_X_epochs = 25
loss = float("inf")
loss_history = []
while loss > 0.0:
    start_idx = random.randrange(start_epoch, end_epoch-3) - start_epoch
    end_idx = start_idx+3
    input_filters = [filter[start_idx:end_idx] for filter in filters.values()]
    groundtruth_outputs_filters = [filter[end_idx]
                                   for filter in filters.values()]

    inputs = []
    groundtruth_outputs = []
    for filter_num in range(len(input_filters)):
        inputs.append(torch.stack(input_filters[filter_num]))
        groundtruth_outputs.append(torch.stack(
            [groundtruth_outputs_filters[filter_num]]))
    inputs = torch.stack(tuple(inputs))
    inputs.to(device, dtype=torch.float)

    groundtruth_outputs = torch.stack(tuple(groundtruth_outputs))
    groundtruth_outputs.to(device, dtype=torch.float)

    optimizer.zero_grad()

    predicted_outputs = model(inputs)

    loss = criterion(torch.squeeze(predicted_outputs),
                     torch.squeeze(groundtruth_outputs))
    loss.backward()
    loss_history.append(loss)
    print(loss.item())

    optimizer.step()

    if num_epochs % save_every_X_epochs == 0:
        print("Saving model after %s epochs" % num_epochs)
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_history': loss_history
        }, os.path.join("./filter_models/11x3", "%s.pt" % num_epochs))

    num_epochs += 1
