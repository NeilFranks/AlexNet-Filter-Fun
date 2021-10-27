import os

import imageio
import matplotlib.pyplot as plt
import torch
import torchvision.models as models

import filters_utils

MODEL_FOLDER = "./models/256_train_and_val_model"
NUM_CLASSES = 1000


def gif_from_all_filters_in_layer0():
    filters_utils.make_gifs_from_layer(
        MODEL_FOLDER, 0, 42, 0, single_channel=False)  # only done with 42 epochs


def plot_trajectory_of_pixel_in_filter_in_layer0(path_to_gif_dir, filter_idx, pixel_row, pixel_col):
    path_to_gif = os.path.join(
        path_to_gif_dir, "0", "%s.gif" % str(filter_idx))  # 0 for layer0
    gif = imageio.get_reader(path_to_gif)

    reds = []
    greens = []
    blues = []
    for frame in gif:
        (r, g, b, _) = frame[pixel_row][pixel_col]
        reds.append(r)
        greens.append(g)
        blues.append(b)

    plt.title("How %s changed at pixel (%s, %s)" %
              (path_to_gif, pixel_row, pixel_col))
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.plot(range(len(reds)), reds, 'r')
    plt.plot(range(len(greens)), greens, 'g')
    plt.plot(range(len(blues)), blues, 'b')

    plt.show()


def extract_weights_from_best_model():
    model = filters_utils.quick_initialize(
        os.path.join(MODEL_FOLDER, "best.pt"))

    layer0_weights = model.features[0].weight.data
    layer3_weights = model.features[3].weight.data

    torch.save({
        'layer0_weights': layer0_weights,
        'layer3_weights': layer3_weights
    }, os.path.join(MODEL_FOLDER, "best_filters_by_layer.pt"))


def make_and_save_untrained_model_with_best_filters():
    best_filters_by_layer = torch.load(os.path.join(
        MODEL_FOLDER, "best_filters_by_layer.pt"))

    model = models.alexnet(pretrained=False)
    numrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(numrs, NUM_CLASSES)
    model.features[0].weight.data = best_filters_by_layer["layer0_weights"]
    model.features[3].weight.data = best_filters_by_layer["layer3_weights"]
    model.features[0].weight.requires_grad = False
    model.features[3].weight.requires_grad = False

    # save this special boy to a special spot
    torch.save({
        'model_state_dict': model.state_dict()
    }, os.path.join(MODEL_FOLDER, "untrained_with_best_filters.pt"))


def make_and_save_untrained_model_with_random_filters():
    model = models.alexnet(pretrained=False)
    numrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(numrs, NUM_CLASSES)

    # save this special boy to a special spot
    torch.save({
        'model_state_dict': model.state_dict()
    }, os.path.join(MODEL_FOLDER, "untrained_with_random_filters.pt"))


# gif_from_all_filters_in_layer0()
# plot_trajectory_of_pixel_in_filter_in_layer0(
#     ".\\gifs\\models\\256_train_and_val_model", 1, 10, 0)
# extract_weights_from_best_model()
# make_and_save_untrained_model_with_best_filters()
make_and_save_untrained_model_with_random_filters()
