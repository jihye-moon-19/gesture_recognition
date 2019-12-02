import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import shutil 
from models_motion_attention import *
from dataset_motion_attention import *
from torch.utils.data import DataLoader
from torch.autograd import Variable 

from opts import parse_opts_offline
#from mean import get_mean, get_std
#from spatial_transforms import *
#from temporal_transforms import *
#from target_transforms import ClassLabel, VideoID
#from target_transforms import Compose as TargetCompose
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
import pdb
import time
import datetime
import pathlib

print("CnnLSTM+Attention")

#os.environ['CUDA_VISIBLE_DEVICES']='3'

#import torch as th
#th.cuda.set_device(2)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

best_prec1 = 0

if __name__ == '__main__':
    opt = parse_opts_offline()
    
    device = torch.device(opt.torch_device)
    if opt.root_path != '':
        # Join some given paths with root path 
        if opt.result_path:
            opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.annotation_path:
            opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
        if opt.video_path:
            opt.video_path = os.path.join(opt.root_path, opt.video_path)
        
    #opt.scales = [opt.initial_scale]
    #for i in range(1, opt.n_scales):
    #    opt.scales.append(opt.scales[-1] * opt.scale_step)
    #opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    #opt.mean = get_mean(opt.norm_value)
    #opt.std = get_std(opt.norm_value)
    #print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    #if opt.no_mean_norm and not opt.std_norm:
    #    norm_method = Normalize([0, 0, 0], [1, 1, 1])
    #elif not opt.std_norm:
    #    norm_method = Normalize(opt.mean, [1, 1, 1])
    #else:
    #    norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        training_data = Gesturedata("train.txt")                        
        train_dataloader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch','num_epochs','batch_i','loss','loss(mean)','acc'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch','num_epochs','batch_i','loss','loss(mean)','acc'])
        
    if not opt.no_val:
        validation_data = Gesturedata("valid.txt")
        test_dataloader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
        
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), 
            ['epoch','num_epochs','batch_i','loss','loss(mean)','acc'])
    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)
    #opt.latent_dim = 512
    # Define network
    model = ConvLSTM(
        num_classes=27,
        latent_dim=128,
        lstm_layers=1,
        hidden_dim=512,
        bidirectional=True,
        attention=True,
    )
    model = model.to(device)

    # Add weights from checkpoint model if specified

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #checkpoint_interval=5
    checkpoint_interval=1
    num_epochs=100
    
    print("completed setting")
    
    def test_model(epoch, num_epochs, criterion):
        """ Evaluate the model on the test set """
        print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y, attention_w) in enumerate(test_dataloader):
            #print(X.shape)
            image_sequences = Variable(X.to(device), requires_grad=False)
            #print("===3333========================================================== SQ ",image_sequences.shape)
            labels = Variable(y, requires_grad=False).to(device)
            attention_w = Variable(attention_w.to(device), requires_grad=False)
            with torch.no_grad():
                # Reset LSTM hidden state
                model.lstm.reset_hidden_state()
                # Get sequence predictions
                predictions = model(image_sequences, attention_w)
            # Compute metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = criterion(predictions, labels).item()
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            # Log test performance
            sys.stdout.write(
                "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    batch_i,
                    len(test_dataloader),
                    loss,
                    np.mean(test_metrics["loss"]),
                    acc,
                    np.mean(test_metrics["acc"]),
                )
            )
            val_logger.log({
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'batch_i': batch_i,
                    'loss':loss,
                    'loss(mean)':np.mean(test_metrics["loss"]),
                    'acc': np.mean(test_metrics["acc"])
            })
        return np.mean(test_metrics["acc"])  
    
    def train_model(train_dataloader, criterion, metrics):
        model.train()
        prev_time = time.time()
        for batch_i, (X, y, attention_w) in enumerate(train_dataloader):

            if X.size(0) == 1:
                continue

            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)
            attention_w = Variable(attention_w.to(device), requires_grad=True)
            optimizer.zero_grad()

            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            # Get sequence predictions
            predictions = model(image_sequences, attention_w)

            # Compute metrics
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward()
            optimizer.step()

            # Keep track of epoch metrics
            metrics["loss"].append(loss.item())
            metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(metrics["loss"]),
                    acc,
                    np.mean(metrics["acc"]),
                    time_left,
                )
            )
            train_batch_logger.log({
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'batch_i': batch_i,
                    'loss':loss.item(),
                    'loss(mean)':np.mean(metrics["loss"]),
                    'acc': np.mean(metrics["acc"])
                    })
            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_logger.log({
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'batch_i': batch_i,
                    'loss':loss.item(),
                    'loss(mean)':np.mean(epoch_metrics["loss"]),
                    'acc': np.mean(epoch_metrics["acc"])
                    })  
        return loss
    
    for epoch in range(num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        print("--- Epoch {epoch} ---")

        loss = train_model(train_dataloader, cls_criterion, epoch_metrics)
        
        # Evaluate the model on the test set
        prec1 = test_model(epoch, num_epochs, cls_criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
        save_checkpoint(state, is_best)
        

        
        '''
        # Save model checkpoint
        if epoch % checkpoint_interval == 0:
            pathlib.Path("model_checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"model_checkpoints/{model.__class__.__name__}_{epoch}.pth")
        '''
