import torch.nn as nn 
import torch 

class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=7), 
            nn.ReLU(), 
            nn.BatchNorm2d(32), 
            nn.Dropout2d(), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=5, stride=5), 
            nn.ReLU(), 
            nn.BatchNorm2d(128), 
            nn.Dropout2d(), 
        )

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2), 
            nn.ReLU(), 
            nn.BatchNorm2d(32), 
            nn.Dropout2d(), 
        )
    
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=4), 
            nn.ReLU(), 
            nn.BatchNorm2d(4), 
            nn.Dropout2d(), 
        )

        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(4, 1, kernel_size=5, stride=5), 
            nn.ReLU(), 
            nn.BatchNorm2d(1), 
            nn.Dropout2d(), 
            nn.Upsample(1216), 
        )
    
    def forward(self, x):
        # print(f"input shape: {x.shape}") 
        x = self.conv1(x)
        # print(f"conv1 shape: {x.shape}") 
        x = self.conv2(x)
        # print(f"conv2 shape: {x.shape}") 
        x = self.tconv1(x)
        # print(f"tconv1 shape: {x.shape}") 
        x = self.tconv2(x)
        # print(f"tconv2 shape: {x.shape}") 
        x = self.tconv3(x)
        # print(f"tconv3 shape: {x.shape}") 
        return x 
