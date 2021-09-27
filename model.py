import torch
import torchvision.models as models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    
    model = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)
    numrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(numrs,num_classes)

    return model

# """
# Fix val history
# """
# import os
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data_dir = "D:/256_train_and_val"
# MODEL_FOLDER = "./models/%s_model" % (data_dir.split("/")[-1])
# num_classes = 1000
# feature_extract = False

# val_acc_history = []

# for i in range(3):
#     model = initialize_model(num_classes, feature_extract, use_pretrained=False)
#     model = model.to(device)
#     params_to_update = model.parameters()
#     print("Params to learn:")
#     if feature_extract:
#         params_to_update = []
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 params_to_update.append(param)
#                 print("\t",name)
#     else:
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 print("\t",name)
#     optimizer = torch.optim.Adam(params_to_update, lr=0.0001, weight_decay=0.0005)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, threshold=0.02, min_lr=0.00001)


#     if os.path.isfile(os.path.join(MODEL_FOLDER, "best.pt")):
#         checkpoint = torch.load(os.path.join(MODEL_FOLDER, "%s.pt" % i))
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         current_epoch = checkpoint['epoch']
#         epoch_acc = checkpoint['acc']
#         val_acc_history.append(checkpoint['acc'])

#         torch.save({
#             'epoch': current_epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'acc': epoch_acc,
#             'val_acc_history': val_acc_history
#             }, os.path.join(MODEL_FOLDER, "%s.pt" % current_epoch))