import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import dataset
from torch.utils.data import DataLoader
from train_func import train_model
from PIL import Image
from tqdm import tqdm 
import sys
import os
import gc 

def test_few(model, result_path, device, batch_size): 
    model.eval() 

    data = dataset(train=False) 
    test_data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Do not need to test label (only predict on test dataset)
    for data, label, img_name, label_name in test_data_loader:
        data = data.to(device)
        label = label.to(device)
        predictions = model(data)

        # Display prediction result
        for i in range(len(predictions)):
            pred = predictions[i].permute(1,2,0).cpu().detach().numpy()
            img = data[i].permute(1,2,0).cpu().detach().numpy()

            # original image
            orig_img = Image.fromarray((img * 255).astype(np.uint8))
            orig_img.save(result_path + img_name[i])

            # unprocessed mask 
            plt.figure()
            plt.imshow(pred, cmap='gray')
            plt.tick_params(left=False, right=False , labelleft=False , labelbottom=False, bottom=False)
            plt.savefig(result_path + "unpro_mask_" + label_name[i])

            # unprocessed imposed image 
            test_img=data[i].permute(1,2,0).cpu().detach().numpy()
            plt.imshow(test_img, alpha = 0.6)
            plt.tick_params(left=False, right=False , labelleft=False , labelbottom=False, bottom=False)
            plt.savefig(result_path + "unpro_result_" + img_name[i])

            # post-processing
            thresh = 0.5 # Post-processing threshold
            thresh_pred = pred
            thresh_pred[thresh_pred < thresh] = 0 # post-processing, value less than threshold is set to 0
            thresh_pred[thresh_pred > thresh] = 1 # post-processing, value greater than threshold is set to 1
            proc_black_pred = black_rgb_label(thresh_pred) # convert to 3 channels
            proc_black_label = Image.fromarray((proc_black_pred * 255).astype(np.uint8))
            proc_black_label.save(result_path + "fake_" + label_name[i])

            # imposed red prediction
            proc_red_pred = rgb_label(proc_black_pred) # convert to rgb (red)
            red_label = Image.fromarray((proc_red_pred * 255).astype(np.uint8))
            imposed_red_img = Image.blend(orig_img, red_label, alpha=0.3)
            imposed_red_img.save(result_path + "result_" + img_name[i])

            # Clear cache 
            torch.cuda.empty_cache() # prevent cuda out of memory
            del pred
            del thresh
            del thresh_pred
            gc.collect() # collect garbage

        break # only predict first batch of test images


def test_all(model, result_path, device, batch_size):
    model.eval()

    data = dataset(train=False) 
    test_data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Do not need to test label (only predict on test dataset)
    for data, label, img_name, label_name in tqdm(test_data_loader):
        data = data.to(device)
        label = label.to(device)
        predictions = model(data)

        # Display prediction result
        for i in range(len(predictions)):
            # print(f"\n\npredictions.shape: {predictions.shape}\n\n")
            pred = predictions[i].permute(1,2,0).cpu().detach().numpy()
            img = data[i].permute(1,2,0).cpu().detach().numpy()
            orig_img = Image.fromarray((img * 255).astype(np.uint8))

            # Predict all test images
            thresh = 0.5 # Post-processing threshold
            thresh_pred = pred
            thresh_pred[thresh_pred < thresh] = 0 # post-processing, value less than threshold is set to 0
            thresh_pred[thresh_pred > (thresh)] = 1 # post-processing, value greater than threshold is set to 1
            
            # save mask (black and white)
            # thresh_pred = pred
            proc_black_pred = black_rgb_label(thresh_pred) # convert to 3 channels
            proc_black_label = Image.fromarray((proc_black_pred * 255).astype(np.uint8))
            proc_black_label.save(result_path + "mask_" + label_name[i])
            
            # save superimposed image (prediction in red)
            proc_red_pred = rgb_label(thresh_pred) # convert to rgb (red)
            red_label = Image.fromarray((proc_red_pred * 255).astype(np.uint8))
            imposed_red_img = Image.blend(orig_img, red_label, alpha=0.3)
            imposed_red_img.save(result_path + "result_" + img_name[i])

            # Clear cache
            torch.cuda.empty_cache() # prevent cuda out of memory
            del pred
            del thresh_pred
            gc.collect() # collect garbage



def rgb_label(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			if label[h][w][0] != 1: # if the prediction is not white
				pred_rgb[h][w][0] = 1
				pred_rgb[h][w][1] = 0
				pred_rgb[h][w][2] = 0
	return pred_rgb


def black_rgb_label(label):
	pred_rgb = np.ones((label.shape[0], label.shape[1], 3))
	for h in range(label.shape[0]):
		for w in range(label.shape[1]):
			if label[h][w][0] != 1: # if the prediction is not white
				pred_rgb[h][w][0] = 0
				pred_rgb[h][w][1] = 0
				pred_rgb[h][w][2] = 0
	return pred_rgb
