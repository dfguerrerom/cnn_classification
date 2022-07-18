
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchvision import models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.require_grad = False
    
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 10.
model.fc = nn.Linear(num_ftrs, 19)
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
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



