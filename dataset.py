import os
import cv2
import torch 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as func
from torch.utils.data import DataLoader
from tqdm import tqdm 
import random 
import matplotlib.pyplot as plt 


data_path = "data/"

class dataset(Dataset):
    def __init__(self, train=True, train_split=0.8, transf=transforms.ToTensor(), random_seed=42):
        self.train = train
        self.train_split = train_split
        self.trans = transf

        self.img_dir = data_path + "image/"
        self.label_dir = data_path + "label/" 
        self.image_files = [file for file in os.listdir(self.img_dir) if file.endswith(('jpg', 'jpeg', 'png'))] 
        self.label_files = [file for file in os.listdir(self.label_dir) if file.endswith(('jpg', 'jpeg', 'png'))] 
        self.num_data = len(self.image_files) 


        # test if all images have corresponding labels 
        # for img_name in self.image_files: 
        #     label_name = img_name.replace("sync_image", "sync_groundtruth_depth")
        #     if label_name not in self.label_files: 
        #         print("label not found!!!")
        #         break 
        # print("all labels found!!!")

        # print(self.image_files[0][-23:])
        # print(self.label_files[0])
        # print(self.label_files[0][-23:])
        # print(self.label_files[0])

        # 2011_10_03_drive_0047_sync_image_0000000758_image_03
        # 2011_09_26_drive_0002_sync_groundtruth_depth_0000000008_image_03

        # print(len(self.image_files))
        # print(len(self.label_files))

        # randomly select training and testing data 
        random.seed(random_seed) 
        indices = [i for i in range(self.num_data)]
        random.shuffle(indices) # randomly shuffle indices 
        train_data_num = int(train_split * self.num_data) 

        if train: 
            self.indices = indices[:train_data_num]
        else:
            self.indices = indices[train_data_num:]

    def __len__(self):
        return len(self.indices) 

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img_name = self.image_files[idx] 
        img = Image.open(self.img_dir + img_name) 
        img = img.resize((1216, 1216)) 

        label_name = img_name.replace("sync_image", "sync_groundtruth_depth")
        label = Image.open(self.label_dir + label_name) 
        # label = label.convert('RGB')
        label = label.resize((1216,1216)) 

        # return img, label, img_name, label_name 
        return self.trans(img), self.trans(label).type(torch.float32), img_name, label_name 


def process_img(data_name):
    '''
    Increase contrast of input image
    Used as a part of transform in dataloader
    '''
    # print("\n\nData name: ", data_name, "\n\n")
    img = cv2.imread(data_name, cv2.COLOR_BGR2RGB)

    # converting to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    rgb_enhanced_img = np.flip(enhanced_img, 2) # change from BGR to RGB
    return rgb_enhanced_img


if __name__ == "__main__": 
    data = dataset() 
    img, label, img_name, label_name = data[0] 
    # print(img.shape) # (3, 1216, 1216) 
    # print(label.shape) # (1, 1216, 1216) 
    
    # # try save the data and label for visual inspection (need to comment out to_Tensor)
    # img.save("image.jpg") 
    # label.save("label.jpg") 

    # # stretch images back to original dimension 
    # img = img.resize((1216, 352))
    # label = label.resize((1216, 352))
    # img.save("image2.jpg") 
    # label.save("label2.jpg") 


    # data = dataset(train=False) 