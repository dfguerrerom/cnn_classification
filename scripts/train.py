import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from tqdm.auto import tqdm

from .model import loss_fn, model, optimizer, optimizer_name, loss_fn_name, scheduler
from .scripts import save_model
from .meters import AverageMeter

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to test the model with the test dataset and print the 
# accuracy for the test images
def test_accuracy(model, test_loader):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for _,sample in enumerate(tqdm(test_loader, leave=False)):
            images, labels = sample.values()
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.squeeze()).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    print(f"accuracy: {accuracy}")

def train_one_epoch(train_loader, train_loss, epoch):
    """Train the training dataloader for one epoch. It will return the average
    loss to the epoch."""
    
    for i, sample in enumerate(tqdm(train_loader, leave=False)):

        # get the inputs
        images = sample["image"]
        labels = sample["label"].squeeze()

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        # Gather data and report
        train_loss.update(loss.item(), epoch, images.size(0),)
        

def train(model, num_epochs, train_loader, val_loader, suffix):
    
    # Create a random model identificator
    model_id = random.randint(999,9999)
    
    loaders = {
        "train" : train_loader,
        "val" : val_loader,
    }
    
    best_vloss = 1_000_000.
    best_accuracy = 0.0

    # Define your execution device
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    
    train_loss = AverageMeter('Train loss', len(train_loader), model_id, suff="train")
    val_loss = AverageMeter('Validation loss', len(val_loader), model_id, suff="val")

    for epoch in range(num_epochs):
        
        model.train(True)
        avg_loss = train_one_epoch(loaders["train"], train_loss, epoch)

        model.train(False)
        model.eval()
        
        for i, sample in enumerate(tqdm(loaders["val"])):
            
            vimages = sample["image"]
            vlabels = sample["label"].squeeze()
            
            voutputs = model(vimages)
            vloss = loss_fn(voutputs, vlabels)
            
            val_loss.update(vloss.item(), epoch, vimages.size(0), )
                    
        test_accuracy(model, loaders["val"])
        
        print(train_loss.summary())
        print(val_loss.summary())
        
        train_loss.save()
        val_loss.save()

        # Track best performance, and save the model's state
        if val_loss.avg < best_vloss:

            best_vloss = val_loss.avg
            save_model(model, model_id, optimizer_name, loss_fn_name, suffix)
            
        print("lr", optimizer.param_groups[0]["lr"])
        scheduler.step()