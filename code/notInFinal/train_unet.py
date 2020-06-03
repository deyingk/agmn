import copy
import time
import os
import json

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from models.unet import UNet
from dataset.my_datasets import CMUPanopticHandmaskDataset

def confirm_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train_model(model, dataloaders,  optimizer, scheduler, check_point, num_epochs):

    confirm_dir(check_point)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    criterion = nn.CrossEntropyLoss()

    loss_recod_dict ={'train':[], 'test':[]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        since = time.time()

        # Each epoch has a training and test phase
        for phase in ['train','test']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print('LR', param_group['lr'])
                model.train() #set model to training mode

            else:
                model.eval() # set model to evaluate mode

            epoch_samples = 0
            total_loss = 0
            for batch_idx, data in enumerate(dataloaders[phase]):
                if batch_idx%100==0:
                    print('Epoch {}/{}, batch {}'.format(epoch, num_epochs-1, batch_idx))
                
                if phase =='train':
                    image = Variable(data['image']).cuda()
                    mask = Variable(data['mask']).cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    output = model(image)

                    loss = criterion(output.permute(0,2,3,1).contiguous().view(-1,2), mask.view(-1))
                    # print(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                else:
                    image = Variable(data['image'], volatile=True).cuda()
                    mask = Variable(data['mask']).cuda()

                    # forward
                    output = model(image)
                    loss = criterion(output.permute(0,2,3,1).contiguous().view(-1,2), mask.view(-1))    
                    total_loss += loss.item()

            epoch_loss = total_loss/(batch_idx+1)

            print('{} loss:{}'.format(phase, epoch_loss))

            if phase =='train':
                loss_recod_dict['train'].append(epoch_loss)
                with open(os.path.join(check_point,'loss_record_dict.json'),'w') as f:
                    json.dump(loss_recod_dict, f)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'scheduler_state_dict': scheduler,
                            'loss': loss
                    }, os.path.join(check_point,'epoch_'+str(epoch).zfill(2)+'.tar'))
            
            if phase == 'test':
                loss_recod_dict['test'].append(epoch_loss)
                with open(os.path.join(check_point,'loss_record_dict.json'),'w') as f:
                    json.dump(loss_recod_dict, f)           

            if phase == 'test' and epoch_loss < best_loss:
                print("saveing the best model")
                best_loss = epoch_loss
                torch.save({
                            'epoch': epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'scheduler_state_dict': scheduler,
                            'loss': loss
                    }, os.path.join(check_point,'best.tar'))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def main():

    # construct dataloader\

    partition_file = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/partitions/partition.json'
    image_dir = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/imgs'
    gdtt_mask_dir = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/gdtt_masks'
    with open(partition_file, 'r') as f:
        partition_dict = json.load(f)

    train_lst = partition_dict['train']
    train_set = CMUPanopticHandmaskDataset(train_lst, image_dir, gdtt_mask_dir)

    valid_lst = partition_dict['valid']
    valid_set = CMUPanopticHandmaskDataset(valid_lst, image_dir, gdtt_mask_dir)

    # # construct dataloader 
    # with open('./dataset/partition_dict.json', 'r') as f:
    #     partition_dict = json.load(f)
    # train_lst = partition_dict['train']
    # train_set = HandmaskDataset(train_lst, './raw_data/train_crop_op/train_data', './raw_data/train_crop_op/train_mask')

    # valid_lst = partition_dict['valid']
    # valid_set = HandmaskDataset(valid_lst, './raw_data/valid_crop_op/valid_data', './raw_data/valid_crop_op/valid_mask')

    dataloaders = {
        'train': DataLoader(train_set, batch_size=12, shuffle=True, num_workers=4),
        'test': DataLoader(valid_set, batch_size=12, shuffle=False, num_workers=4)
    }

    model = UNet(n_classes=2, in_channels=3, padding=True, up_mode='upsample').cuda()
    model = torch.nn.DataParallel(model)

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model=model,
                        dataloaders=dataloaders, 
                        optimizer=optimizer_ft, 
                        scheduler=exp_lr_scheduler,
                        check_point='../checkpoint/Unet/3_29',
                        num_epochs=100)


if __name__ =='__main__':
    main()

