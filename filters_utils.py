import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image

from model import quick_initialize


MODEL_FOLDER = "./models/256_train_and_val_model"


def make_gifs_from_layer(path_to_model_dir, start_number, end_number, layer_num, single_channel):
    images = {}
    model = quick_initialize(os.path.join(
        MODEL_FOLDER, "%s.pt" % start_number))
    weight_tensor = model.features[layer_num].weight.data

    for model_idx in range(start_number, end_number+1):
        print("Getting images from model %s/%s" %
              (model_idx, end_number), end='\r')
        model = quick_initialize(os.path.join(
            MODEL_FOLDER, "%s.pt" % model_idx))
        for filter_idx in range(weight_tensor.shape[0]):
            weight_tensor = model.features[layer_num].weight.data

            # looping through all the kernels in each channel
            npimg = np.array(weight_tensor[filter_idx].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5))
                               )
            npimg *= 255
            npimg = npimg.astype(np.uint8).transpose((1, 2, 0))

            if filter_idx not in images.keys():
                images[filter_idx] = []
            images[filter_idx].append(Image.fromarray(npimg, 'RGB'))

    print("\n")
    for filter_idx in images.keys():
        output_dir = "./gifs/%s/%s" % (
            path_to_model_dir.strip('./'), layer_num)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        imageio.mimsave("%s/%s.gif" %
                        (output_dir, str(filter_idx)), images[filter_idx])
        imageio.imsave("%s/%s_start.png" %
                       (output_dir, str(filter_idx)), images[filter_idx][0])
        imageio.imsave("%s/%s_middle.png" %
                       (output_dir, str(filter_idx)), images[filter_idx][len(images.keys())//2])
        imageio.imsave("%s/%s_end.png" %
                       (output_dir, str(filter_idx)), images[filter_idx][-1])


def get_filters_from_layer(start_number, end_number, layer_num):
    filters = {}
    model = quick_initialize(os.path.join(
        MODEL_FOLDER, "%s.pt" % start_number))

    for model_idx in range(start_number, end_number):
        print("Getting filters from model %s/%s" %
              (model_idx, end_number-1), end='\r')
        model = quick_initialize(os.path.join(
            MODEL_FOLDER, "%s.pt" % model_idx))

        weight_tensor = model.features[layer_num].weight.data

        for filter_idx in range(weight_tensor.shape[0]):
            if filter_idx not in filters.keys():
                filters[filter_idx] = []
            filters[filter_idx].append(weight_tensor[filter_idx])

    return filters


def scale_array(array, scale):
    # should do this one channel at a time, if you want multichannel.
    # ie, the array you pass in should be from one channel
    scaled_array = np.array([])
    for row in array:
        scaled_row = np.array([])
        for pixel in row:
            scaled_row = np.append(scaled_row, [pixel]*scale)
        for _ in range(scale):
            if scaled_array.size == 0:
                scaled_array = scaled_row
            else:
                scaled_array = np.vstack([scaled_array, scaled_row])

    return scaled_array


def plot_filter_multichannels(filter):

    filter = torch.squeeze(filter)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    # we convert the tensor to numpy
    npimg = np.array(filter.detach().numpy().reshape(11, 11, 3), np.float32)
    # standardize the numpy image
    npimg = (npimg - np.mean(npimg)) / np.std(npimg)
    npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
    ax1.imshow(npimg)
    plt.show()


"""
FROM https://colab.research.google.com/github/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch/CNNVisualisation.ipynb#scrollTo=qv-nJbDFuNuN
"""


def plot_filters_single_channel(t):

    # kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12

    nrows = 1 + nplots//ncols
    # convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])


"""
FROM https://colab.research.google.com/github/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch/CNNVisualisation.ipynb#scrollTo=qv-nJbDFuNuN
"""


def plot_filters_multi_channel(t):

    # get the number of kernels
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.show()


"""
FROM https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
"""


def plot_weights(model, layer_num, single_channel=True, collated=False):

    # extracting the model features at the particular layer number
    layer = model.features[layer_num]

    # checking whether the layer is convolution layer or not
    if isinstance(layer, nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = model.features[layer_num].weight.data

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor)
            else:
                plot_filters_single_channel(weight_tensor)

        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor)
            else:
                print(
                    "Can only plot weights with three channels with single channel = False")

    else:
        print("Can only visualize layers which are convolutional")
