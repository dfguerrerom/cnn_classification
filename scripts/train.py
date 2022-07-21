import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from tqdm.auto import tqdm

from .model import loss_fn, model, optimizer, optimizer_name, loss_fn_name, scheduler, device
from .scripts import get_model_path




# Function to test the model with the test dataset and print the 
# accuracy for the test images
def test_accuracy(model, test_loader, acc_writer, epoch):
    
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
    acc_writer.update((100 * accuracy / total), epoch)

def train_one_epoch(model, train_loader, train_writer, epoch):
    """Train the training dataloader for one epoch. It will return the average
    loss to the epoch."""
    
    model.train(True)
    
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
        train_writer.update(loss.item(), epoch, images.size(0),)
        
def validate(model, val_loader, val_writer, epoch):
    
    model.train(False)
    model.eval()

    for i, sample in enumerate(tqdm(val_loader)):

        vimages = sample["image"]
        vlabels = sample["label"].squeeze()

        voutputs = model(vimages)
        vloss = loss_fn(voutputs, vlabels)

        val_writer.update(vloss.item(), epoch, vimages.size(0), )

        

def train(
    model, 
    num_epochs, 
    train_loader, 
    val_loader, 
    suffix,
    train_writer,
    val_writer,
    acc_writer,
):
    
    timestr = time.strftime("%Y%m%d-%H%M")
    
    # Create a random model identificator
    model_id = random.randint(999,9999)
    model_path = get_model_path(
        timestr, model_id, optimizer_name, loss_fn_name, suffix
    )
    
    loaders = {
        "train" : train_loader,
        "val" : val_loader,
    }
    
    best_vloss = 1_000_000.
    best_accuracy = 0.0

    print("The model will be running on", device, "device")    
    
    for epoch in range(num_epochs):
        
        
        train_one_epoch(model, loaders["train"], train_writer, epoch)

        validate(model, loaders["val"], val_writer, epoch)
        
        test_accuracy(model, loaders["val"], acc_writer, epoch)

        train_writer.save(model_path.stem)
        val_writer.save(model_path.stem)
        acc_writer.save(model_path.stem)

        # Track best performance, and save the model's state
        if val_writer.avg < best_vloss:

            best_vloss = val_writer.avg
                        
            print(f"Saving model...{model_path.stem}")
            torch.save(model.state_dict(), model_path)
            
        print("lr", optimizer.param_groups[0]["lr"])
        scheduler.step()
        
        train_writer.plot_metrics()