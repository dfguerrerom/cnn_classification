import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from tqdm.auto import tqdm

from .scripts import get_model_path
from scripts.model import get_settings



def train_one_epoch(model, optimizer, loss_fn, train_loader, train_writer, epoch):
    """Train the training dataloader for one epoch. It will return the average
    loss to the epoch."""
    
    model.train(True)
    running_loss = 0.0
    # Count the number of images that are passing each iteration
    total = 0.0

    for i, sample in enumerate(tqdm(train_loader, leave=False)):

        # get the inputs
        images = sample["image"]
        labels = sample["label"].squeeze()

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        total += images.size(0)
        running_loss += loss.item() * images.size(0)
        
    writer.update("training", running_loss / total, epoch)
        
def validate(model, loss_fn, val_loader, val_writer, epoch):
    
    model.train(False)
    model.eval()
    
    running_loss = 0.0
    # Count the number of images that are passing each iteration
    total = 0.0
    
    for i, sample in enumerate(tqdm(val_loader)):

        vimages = sample["image"]
        vlabels = sample["label"].squeeze()

        voutputs = model(vimages)
        vloss = loss_fn(voutputs, vlabels)
        
        total += vimages.size(0)
        running_loss += vloss.item() * vimages.size(0)
    
    writer.update("validation", running_loss / total, epoch)
        
def test_accuracy(model, test_loader, writer, epoch):
    
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
    writer.update("accuracy",  accuracy / total, epoch)

def train(
    TEST_NAME,
    num_epochs, 
    train_loader, 
    val_loader, 
    suffix,
    writer,
):

    (
        model,
        model_name, 
        optimizer, 
        loss_fn, 
        scheduler,
        variable, 
        batch_size, 
        rescale_factor
    ) = get_settings(TEST_NAME)
    
    timestr = time.strftime("%Y%m%d-%H%M")
    
    # Create a random model identificator
    model_id = random.randint(999,9999)
    model_path = get_model_path(TEST_NAME, model_id, timestr, suffix)
    
    loaders = {
        "train" : train_loader,
        "val" : val_loader,
    }
    
    best_vloss = 1_000_000.
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        
        train_one_epoch(model, optimizer, loss_fn, loaders["train"], writer, epoch)
        validate(model, loss_fn, loaders["val"], writer, epoch)
        test_accuracy(model, loaders["val"], writer, epoch)

        writer.save(model_path.stem)

        # Track best performance, and save the model's state
        if val_writer.avg < best_vloss:

            best_vloss = val_writer.avg
                        
            print(f"Saving model...{model_path.stem}")
            torch.save(model.state_dict(), model_path)
            
        print("lr", optimizer.param_groups[0]["lr"])
        scheduler.step()
        
        train_writer.plot_metrics()