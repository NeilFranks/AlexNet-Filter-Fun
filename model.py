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