import os
import time

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import mean_activity_transform
import utils
from model import initialize_model


"""
VARIOUS SETUP THINGS
"""

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "D:/256_train_and_val"
# data_dir = "D:/baby_256_train_and_val"
# MODEL_FOLDER = "./models/%s_model" % (data_dir.split("/")[-1])
MODEL_FOLDER = "./models/256_train_and_val_model"

# images are 224x224
input_size = 224

# Number of classes in the dataset
num_classes = 1000
# num_classes = 2

"""
From Alexnet paper:
"We trained our models using stochastic gradient descent
with a batch size of 128 examples"
"""
# Batch size for training (change depending on how much memory you have)
batch_size = 128
workers = 6

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False


def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        mean_activity_transform.mean_activity_transform()
    ])

    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )

    return val_dataset


def data_loader(data_dir, batch_size=batch_size, workers=workers, pin_memory=True):
    val_ds = val_dataset(data_dir)

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,  # why not shuffle?
        num_workers=workers,
        pin_memory=pin_memory
    )

    return val_loader


"""
GET READY FOR ACTUAL TRAINING
"""


def train_model(model, dataloader):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    seen_so_far = 0
    running_top_1_acc = 0
    running_top_5_acc = 0

    # Iterate over data.
    iterations = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            """
            From Alexnet paper:
            "At test time, the network makes a prediction by extracting five 224 × 224 patches 
            (the four corner patches and the center patch) as well as their horizontal reflections 
            (hence ten patches in all), and averaging the predictions made by the network’s softmax
            layer on the ten patches."
            """
            patches = utils.extract_10_patches(inputs)
            patch_outputs = torch.stack([model(patch) for patch in patches])
            outputs = torch.mean(patch_outputs, axis=0)

        # statistics
        seen_so_far += len(labels)

        # utils.top_k_accuracy(outputs, labels, (5,))

        top_1_acc, top_5_acc = utils.top_k_accuracy(outputs, labels, (1, 5))

        # weighted average
        running_top_1_acc = (
            ((seen_so_far-len(labels))*running_top_1_acc) + (len(labels)*top_1_acc))/seen_so_far
        running_top_5_acc = (
            ((seen_so_far-len(labels))*running_top_5_acc) + (len(labels)*top_5_acc))/seen_so_far

        iterations += torch.tensor(1)
        print("\tIterations: %s" % iterations.item())
        print("Top 1 accuracy=%s" % running_top_1_acc.item())
        print("Top 5 accuracy=%s" % running_top_5_acc.item())

    print()

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, running_top_1_acc, running_top_5_acc


if __name__ == "__main__":
    print("Initializing Dataset and Dataloader...")
    val_loader = data_loader(data_dir)
    dataloader = val_loader

    # Initialize the model for this run
    model = initialize_model(
        num_classes, feature_extract, use_pretrained=False)

    # Send the model to GPU
    model = model.to(device)

    # Load the best model checkpoint
    # checkpoint = torch.load(os.path.join(MODEL_FOLDER, "best.pt"))
    checkpoint = torch.load(os.path.join(
        MODEL_FOLDER, "untrained_with_random_filters.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Train and evaluate
    model, hist = train_model(model, dataloader)
