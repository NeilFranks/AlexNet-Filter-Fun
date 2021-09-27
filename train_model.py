import copy
import datetime
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import intensity_transform
import mean_activity_transform
import utils
from model import initialize_model


"""
With tips from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

"""
VARIOUS SETUP THINGS
"""

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "D:/256_train_and_val"
# data_dir = "D:/baby_256_train_and_val"
MODEL_FOLDER = "./models/%s_model" % (data_dir.split("/")[-1])

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
workers = 7

"""
From Alexnet paper:
"We trained the network for roughly 90 cycles..."
"""
# Number of epochs to train for
num_epochs = 90

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

def train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        intensity_transform.intensity_transform(),
        mean_activity_transform.mean_activity_transform()
    ])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )
    
    return train_dataset

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
    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

"""
GET READY FOR ACTUAL TRAINING
"""
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    val_acc_history = []

    # Load the best model checkpoint, if you have one
    if os.path.isfile(os.path.join(MODEL_FOLDER, "best.pt")):
        checkpoint = torch.load(os.path.join(MODEL_FOLDER, "best.pt"))
        best_model_wts = checkpoint['model_state_dict']
        best_acc = checkpoint['acc']
    else:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{} - {}'.format(epoch, num_epochs - 1, datetime.datetime.now()))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_val_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iterations = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs = model(inputs)
                    else:
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
                    loss = criterion(outputs, labels.long())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        val_loss = None
                    else:
                        val_loss = loss

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if val_loss:
                    running_val_loss += val_loss.item() * inputs.size(0)

                if iterations%10 == 0:
                    with open('preds.txt', 'w') as file:
                        file.write(json.dumps(
                            Counter(
                                np.asarray(copy.deepcopy(preds).cpu(), dtype=str)
                            )
                        ))
                        file.write("\n\n%s" % running_corrects)

                iterations += torch.tensor(1)
                print("\tIterations: %s" % iterations.item(), end='\r')

            print()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            """
            From Alexnet paper:
            "The heuristic which we followed was to divide the learning rate by 10 when the validation error
            rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and
            reduced three times prior to termination. "
            """
            scheduler.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # save this special boy to a special spot
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'acc': epoch_acc
                    }, os.path.join(MODEL_FOLDER, "best.pt"))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        # save after every epoch, since it takes so long to do an epoch and takes 1 second to save it...
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'acc': epoch_acc
            }, os.path.join(MODEL_FOLDER, "%s.pt" % epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

"""
MAKE THE MODEL
"""
if __name__ == "__main__":
    print("Initializing Datasets and Dataloaders...")
    train_loader, val_loader = data_loader(data_dir)
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Initialize the model for this run
    model = initialize_model(num_classes, feature_extract, use_pretrained=False)

    # print(model)

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)


    """
    From Alexnet paper:
    "We trained our models using stochastic gradient descent
    with ... momentum of 0.9, and weight decay of 0.0005"
    ...
    "We used an equal learning rate for all layers, which we adjusted manually throughout training.
    The heuristic which we followed was to divide the learning rate by 10 when the validation error
    rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and
    reduced three times prior to termination. "

    TODO: Uh, it appears the SGD optimizer just does not train? Ongoing discussion here https://discuss.pytorch.org/t/image-recognition-alexnet-training-loss-is-not-decreasing/66885/9
    Using Adam until I figure out what's up with that...
    """
    # Observe that all parameters are being optimized
    # optimizer = torch.optim.SGD(params_to_update, lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params_to_update, lr=0.01, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, threshold=0.02, min_lr=0.00001)

    # Setup the loss fxn
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs)
