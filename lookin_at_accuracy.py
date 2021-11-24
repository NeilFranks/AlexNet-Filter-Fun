import os

import torch

import matplotlib.pyplot as plt


MODEL_FOLDER = "./models/256_train_and_val_model"
BOOT_MODEL_FOLDER = "./models/256_train_and_val_model_from_bootstrapped_weights"


def compare_accuracy_to_bootstrapped_accuracy(path_to_real_model_dir, real_start, real_end, path_to_bootstrapped_model_dir, boot_start, boot_end):

    checkpoint = torch.load(os.path.join(
        path_to_real_model_dir, "%s.pt" % real_end))

    real_accs = [a.cpu() for a in checkpoint['val_acc_history']][real_start:]

    checkpoint = torch.load(os.path.join(
        path_to_bootstrapped_model_dir, "%s.pt" % boot_end))

    boot_accs = [a.cpu() for a in checkpoint['val_acc_history']][boot_start:]

    plt.title("Accuracy of model vs bootstrapped model")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(range(len(real_accs)), real_accs)
    plt.plot(range(len(boot_accs)), boot_accs)

    plt.show()
    b = 0


compare_accuracy_to_bootstrapped_accuracy(
    MODEL_FOLDER, 0, 46, BOOT_MODEL_FOLDER, 0, 3),
