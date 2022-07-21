import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(output_size, fixed_feature=False,):
    """get corresponding pretrained model. Either fixed_feature or fine_tune"""
    
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features

    if fixed_feature:
        # ConvNet as fixed feature extractor
        _=[setattr(param, "require_grad", False) for param in model.parameters()]
    
    model.fc = nn.Linear(num_ftrs, output_size)
    
    return model


model = get_model(6, fixed_feature=False)
model.to(device)


# Define the loss function with Classification Cross-Entropy loss and an optimizer
# with Adam optimizer

optimizers = {
    "Adam" : Adam(model.parameters(), lr=0.1, weight_decay=0.0001)
}
loss_fns = {
    "CrossEntropy" : nn.CrossEntropyLoss(),
    "MSELoss" : nn.MSELoss(),
}

optimizer_name = "Adam"
loss_fn_name = "CrossEntropy"

optimizer = optimizers[optimizer_name]
loss_fn = loss_fns[loss_fn_name]


lambda1 = lambda epoch: 0.65 ** epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


