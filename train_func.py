from torch.utils.data import DataLoader
import torch.nn as nn 
from tqdm import tqdm
from dataset import dataset 
import torch
import os
import gc

def train_model(model, batch_size, epochs, device, log_file_name, weight):
    """
    Train the model

    Parameters:
        model: Model to be trained
        epochs: Number of epochs to train

    Returns:
        loss_list (list): List containing the loss value for each epoch
    """

    # initialize optimization parameters 
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2)
    criterion = nn.MSELoss() 
    data = dataset() 
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    LOG_FILE = open(log_file_name, "w")
    model.to(device)    
    model.train()
    for epoch in range(epochs):
        loss_list = []
        print(f"Epoch: {epoch + 1}")
        for data, label, _, _ in tqdm(data_loader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)

            # training procedure 
            optimizer.zero_grad()
            loss = criterion(pred, label) 
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())

            # Clear cache (prevent cuda out of memory)
            torch.cuda.empty_cache() # clear cache
            del data
            del label
            del pred
            del loss
            gc.collect()

        loss_val = sum(loss_list)/len(loss_list)
        print(f"loss: {loss_val:.6f}")

        LOG_FILE.write(f"Epoch {epoch+1:2d}, loss: {loss_val:.6f}\n")
        LOG_FILE.flush()

        torch.save(model.state_dict(), weight) # save weight for each epoch

    LOG_FILE.close()