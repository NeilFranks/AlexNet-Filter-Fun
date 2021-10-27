import os

import imageio
import matplotlib.pyplot as plt

import filters_utils

MODEL_FOLDER = "./models/256_train_and_val_model"


def gif_from_all_filters_in_layer0():
    filters_utils.make_gifs_from_layer(
        MODEL_FOLDER, 0, 42, 0, single_channel=False)  # only done with 42 epochs


def plot_trajectory_of_pixel_in_filter(path_to_gif_dir, filter_idx, pixel_row, pixel_col):
    path_to_gif = os.path.join(
        path_to_gif_dir. filter_idx, "%s.gif" % filter_idx)
    gif = imageio.get_reader(path_to_gif)

    reds = []
    greens = []
    blues = []
    for frame in gif:
        (r, g, b) = frame[pixel_row][pixel_col]
        reds.append(r)
        greens.append(g)
        blues.append(b)

    plt.title("How %s pixel(%s, %s) changed" %
              (path_to_gif, pixel_row, pixel_col))
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.plot(range(len(reds), reds, 'r'))
    plt.plot(range(len(greens), greens, 'g'))
    plt.plot(range(len(blues), blues, 'b'))


# gif_from_all_filters_in_layer0()
plot_trajectory_of_pixel_in_filter(
    "./gifs/models/256_train_and_val_model/0", 0, 0, 0)
