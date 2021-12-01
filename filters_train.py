import random
import os

import torch

from filter_prediction.FilterPred_11x3 import FilterPred_11x3
from filter_prediction.FilterPred_5x64 import FilterPred_5x64
from filters_utils import MODEL_FOLDER, get_filters_from_layer, plot_filter_multichannels

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model info; which epochs you wants, and which layer
layer_num = 0
start_epoch = 4
end_epoch = 43

torch.set_grad_enabled(True)

# get all your little filters bro
filters = get_filters_from_layer(start_epoch, end_epoch, layer_num)

# load up the model-
criterion = torch.nn.MSELoss()

MODEL_FOLDER = "./filter_models/11x3"
save_every_X_epochs = 150
model = FilterPred_5x64()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.0005)

num_epochs = 0
loss = float("inf")
loss_history = []

checkpoint = torch.load(os.path.join(MODEL_FOLDER, "best.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
num_epochs = checkpoint['epoch'] + 1
loss = checkpoint['loss']
loss_history = checkpoint['loss_history']

best_loss = loss

model.train()

prediction_span = 30  # predict what it will be in 30 epochs

while loss > 0.0:
    start_idx = random.randrange(
        start_epoch, end_epoch-prediction_span) - start_epoch
    end_idx = start_idx+prediction_span
    input_filters = [filter[start_idx:start_idx+3]
                     for filter in filters.values()]
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

    # if num_epochs % save_every_X_epochs == 0:
    if loss < best_loss:
        best_loss = loss.item()
        print("Saving model with loss=%s after %s epochs" %
              (best_loss, num_epochs))
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_history': loss_history,
            'best_loss': best_loss
        }, os.path.join(MODEL_FOLDER, "best.pt"))

    num_epochs += 1
