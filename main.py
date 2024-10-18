import torch
from train_func import train_model
from test_func import test_few, test_all 
from model import CNN 
import os 

if __name__ == "__main__":
    batch_size = 8 
    num_epochs = 3
    is_red_pred = True
    cuda = "cuda:0"

    exp_name = f"{num_epochs}_epo_trial_1" 
    device = torch.device(cuda) 
    model = CNN()

    weight_dir = "weight"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    weight_name = weight_dir + "/" + exp_name + ".pt"
    log_file_name = log_dir + "/" + exp_name + ".txt"
    train_model(model, batch_size, num_epochs, device, log_file_name, weight_name)
    


    # test 
    model.load_state_dict(torch.load(weight_name))
    model.to(device)

    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    # test 8 images
    result_path = result_dir + "/" + exp_name + "/" 
    test_few(model, result_path, device, batch_size) 

    # test all images 
    # result_path = result_dir + "/" + exp_name + "_full/" 
    # test_all(model, result_path, device, batch_size) 
