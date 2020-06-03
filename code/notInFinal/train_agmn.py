"""
This script is used to train adaptive graphical model network.

train_mode: 0: joint training, given pre-trained binary and unary branch
            1: freeze unary branch, train binary branch, use pairwise loss and last score map loss
            2: freeze unary branch, train binary branch, only use pairwise loss
            3: train from guided trained binary branch



"""



import json
import os
import argparse
from datetime import datetime
import shutil
import sys
import logging
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import dataset.my_datasets as datasets_module
import models
from utils import utils



def confirm_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_and_evaluate(model, dataloaders, optimizer, loss_fn, model_dir, device, gdtt_all_label_dict, num_epochs=100, restore_file_path=None, train_mode=0, loss_coef=None):
    """ Train the model and evaluate the model on validation dataset every epoch.

    Args:
        model: the neural network
        train_dataloader:
        val_dataloader:
        optimizer:
        loss_fun:
        model_dir: directory containing config, weights and logging
        device:
        gdtt_all_label_dict: a dictionary gdtt labels for all images
        num_epochs:
        restore_file_path: (string) optional - path of the file to restore from

    """

    #reload weights from restore file if specified
    if restore_file_path is not None:
        restore_path = os.path.join()
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    log_train = open(os.path.join(model_dir, 'log_train.txt'), 'a')
    log_valid = open(os.path.join(model_dir, 'log_valid.txt'), 'a')
    
    best_val_acc = 0.0
    best_epoch = 0


    for epoch in range(num_epochs):
        # run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, num_epochs))
        train_loss = []
        val_loss = []
        #-------------------train model---------------------------------#
        print('training.....')
        model.train()
        train_pred_label_dict = {}
        with tqdm(total=len(dataloaders['train'])) as t:
            for batch_id, data in enumerate(dataloaders['train']):
                image = data['image'].to(device)
                gdtt_heatmap = data['heatmap']
                gdtt_heatmap = torch.stack([gdtt_heatmap]*6, dim=1).to(device)
                normalized_heatmap = data['normalized_heatmap'].to(device)

                gdtt_gm_kernels = data['gm_kernels']
                gdtt_gm_kernels = torch.stack([gdtt_gm_kernels]*6, dim=1).to(device)

                optimizer.zero_grad()
                gm_kernels_6, pred_7 = model(image)  

                if train_mode == 0 or train_mode ==3:
                    if loss_coef==None:
                        loss = 1*loss_fn(pred_7[:,-1,...], normalized_heatmap) + 0.1*loss_fn(pred_7[:,0:6,...], gdtt_heatmap) + 0.1* loss_fn(gm_kernels_6, gdtt_gm_kernels)
                    else:
                        a1= loss_coef[0]
                        a2 = loss_coef[1]
                        a3 = loss_coef[2]
                        loss = a1*loss_fn(pred_7[:,-1,...], normalized_heatmap) + a2*loss_fn(pred_7[:,0:6,...], gdtt_heatmap) + a3* loss_fn(gm_kernels_6, gdtt_gm_kernels)
                elif train_mode == 1:
                    loss = loss_fn(pred_7[:,-1,...], normalized_heatmap) + 0.1* loss_fn(gm_kernels_6, gdtt_gm_kernels)
                elif train_mode == 2:
                    loss = loss_fun(gm_kernels_6, gdtt_gm_kernels)                  
                loss.backward()
                # print(loss.item())
                optimizer.step()

                loss_last = loss_fn(pred_7[:,-1,...], normalized_heatmap)
                train_loss.append(loss_last.item())

                # get predicted label
                original_image_sizes = data['original_image_size']
                image_names = data['image_name']

                pred_7 = pred_7.cpu()
                for image_i in range(len(image_names)):
                    # print('last score #############################################')
                    # print(pred_7[image_i, -1, ...])
                    # print(pred_7[image_i, -1, ...].shape)
                    
                    predicted_label = utils.get_labels_from_heatmap(pred_7[image_i, -1, ...], original_image_sizes[image_i])
                    train_pred_label_dict[image_names[image_i]] = {}
                    train_pred_label_dict[image_names[image_i]]['pred_label'] =  predicted_label
                    train_pred_label_dict[image_names[image_i]]['resol'] =  float(original_image_sizes[image_i][0])
                t.update()
        confirm_dir(os.path.join(model_dir,'train_pred_labels'))
        with open(os.path.join(model_dir,'train_pred_labels','epoch_'+str(epoch).zfill(3)+'.json'),'w') as f:
            json.dump(train_pred_label_dict, f)

        #------------------validation------------------------------------#
        print('validation......')
        model.eval()
        val_pred_label_dict = {}
        with torch.no_grad():
            with tqdm(total=len(dataloaders['valid'])) as t:
                for batch_id, data in enumerate(dataloaders['valid']):
                    image = data['image'].to(device)
                    gdtt_heatmap = data['heatmap']
                    labels = data['label']
                    gdtt_heatmap = torch.stack([gdtt_heatmap]*6, dim=1).to(device)
                    gdtt_gm_kernels = data['gm_kernels']
                    gdtt_gm_kernels = torch.stack([gdtt_gm_kernels]*6, dim=1).to(device)
                    normalized_heatmap = data['normalized_heatmap'].to(device)
                
                    gm_kernels_6, pred_7 = model(image)  
                    # gm_kernels_6: batch_size x 6 x 40 x 45 x 45
                    # pred_7: batch_size x 7 x 40 x 46 x 46
                    loss_last = loss_fn(pred_7[:,-1,...], normalized_heatmap)
                    val_loss.append(loss_last.item())

                    # get predicted label
                    original_image_sizes = data['original_image_size']
                    image_names = data['image_name']


                    pred_7 = pred_7.cpu()
                    for image_i in range(len(image_names)):
                        predicted_label = utils.get_labels_from_heatmap(pred_7[image_i, -1, ...], original_image_sizes[image_i])
                        val_pred_label_dict[image_names[image_i]] = {}
                        val_pred_label_dict[image_names[image_i]]['pred_label'] =  predicted_label
                        val_pred_label_dict[image_names[image_i]]['resol'] =  float(original_image_sizes[image_i][0])


                    t.update()
            confirm_dir(os.path.join(model_dir,'valid_pred_labels'))
            with open(os.path.join(model_dir,'valid_pred_labels','epoch_'+str(epoch).zfill(3)+'.json'),'w') as f:
                json.dump(val_pred_label_dict, f)
        
        #------------------calculate pcks and save models----------------#
        # pck on training set
        train_cur_pck = 0.0
        for image_name in train_pred_label_dict.keys():
            pred_label = train_pred_label_dict[image_name]['pred_label']
            pred_resol = train_pred_label_dict[image_name]['resol']
            gdtt_label = gdtt_all_label_dict[image_name]
            pck = utils.PCK(pred_label, gdtt_label, pred_resol, sigma=0.04)
            train_cur_pck = train_cur_pck + pck
        train_acc = train_cur_pck/len(train_pred_label_dict)
        # pck on validation set
        val_cur_pck = 0.0
        for image_name in val_pred_label_dict.keys():
            pred_label = val_pred_label_dict[image_name]['pred_label']
            pred_resol = val_pred_label_dict[image_name]['resol']
            gdtt_label = gdtt_all_label_dict[image_name]
            pck = utils.PCK(pred_label, gdtt_label, pred_resol, sigma=0.04)
            val_cur_pck = val_cur_pck + pck
        val_acc = val_cur_pck/len(val_pred_label_dict)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                        'epoch': epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'val_acc': val_acc,
                }, os.path.join(model_dir,'best.tar'))

        torch.save({
                    'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'val_acc': val_acc,
            }, os.path.join(model_dir,'epoch_'+str(epoch).zfill(3)+'.tar'))

        # **************** save loss for one epoch ****************
        average_train_loss = sum(train_loss)/ len(train_loss)
        average_val_loss = sum(val_loss) / len(val_loss)
        logger.info('epoch ' + str(epoch))
        logger.info('train epoch loss ' + str(average_train_loss))
        logger.info('validation epoch loss ' + str(average_val_loss))
        logger.info('valid current pck ' + str(val_acc))
        logger.info('max valid pck so far' + str(best_val_acc))
        logger.info('best epoch so far ' + str(best_epoch))
        log_train.write(str(epoch).zfill(3)+','+str(average_train_loss)+','+str(train_acc) + '\n')
        log_valid.write(str(epoch).zfill(3)+','+str(average_val_loss)+','+str(val_acc) + '\n')
        log_train.flush()
        log_valid.flush()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='config file for the experiment')
    args = parser.parse_args()

    # get configurations
    with open(os.path.join('./configs', args.config_file),'r') as f:
        config = json.load(f)

    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_mode = config['train_mode']

    # make experiment directory
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M-%S")
    model_dir = os.path.join('../checkpoint',config['data_loader']['name'], config['model_name'], date_time)

    # model_dir = os.path.join('../checkpoint',config['model_name'], 'train_mode_'+str(train_mode) ,date_time)
    # if config['data_loader']['name'] == 'UCIHandDataset':
    #     model_dir = os.path.join('../checkpoint_UCI',config['model_name'], 'train_mode_'+str(train_mode) ,date_time)

    confirm_dir(model_dir)
    # copy the config file to this directory
    shutil.copy(os.path.join('./configs', args.config_file), model_dir)

    # set logger
    logger = utils.set_logger(os.path.join(model_dir,'train.log'))

    # --------------------------data preparation ----------------------------------------#
    logger.info('Loading the datasets')

    # load all groundtruth labels, would be used in calculating pck
    with open(config['data']['all_labels'], 'r') as f:
        gdtt_all_label_dict = json.load(f)

    # get dataset records
    with open(config['data']['partition']) as f:
        partition = json.load(f)
    train_records = partition['train']
    val_records = partition['valid']
    test_records = partition['test']

    # get dataloaders
    dl_param = config['data_loader'] # dataloader parameters
    dataloaders = {}
    my_dataset = getattr(datasets_module, dl_param['name'])
    #train
    train_set = my_dataset(train_records, config['data']['path'])
    dataloaders['train'] = DataLoader(train_set, batch_size=dl_param['batch_size'], shuffle=True, num_workers=dl_param['number_workers'])
    #val
    val_set = my_dataset(val_records, config['data']['path'])
    dataloaders['valid'] = DataLoader(val_set, batch_size=dl_param['batch_size'], shuffle=False, num_workers=dl_param['number_workers'])
    #test
    test_set = my_dataset(test_records, config['data']['path'])
    dataloaders['test'] = DataLoader(test_set, batch_size=dl_param['batch_size'], shuffle=False, num_workers=dl_param['number_workers'])
    logger.info('- done.')

    # ----------------------------- model preparation ----------------------------------#
    logger.info('Loading model...')
    # get model
    model_class = getattr(models,config['model_name'])
    model = model_class(21)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    
    # if train_mode == 0:
    #     logger.info('loading pretrained network components separately..........')
    #     binary_branch_dir = config['binary_branch_dir']
    #     unary_branch_dir = config['unary_branch_dir']
    #     utils.load_checkpoint(os.path.join(binary_branch_dir, 'best.tar'), model.module.binarybranch, use_module=True)
    #     utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)
    # elif train_mode ==1:
    #     logger.info('loading pretrained unary network component..........')
    #     unary_branch_dir = config['unary_branch_dir']
    #     utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)
    # elif train_mode == 2:
    #     raise("not implemented")
    # else:
    #     raise("Wrong parameter for train mode!")

    if train_mode == 0:
        logger.info('loading pretrained network components separately..........')
        binary_branch_dir = config['binary_branch_dir']
        unary_branch_dir = config['unary_branch_dir']
        utils.load_checkpoint(os.path.join(binary_branch_dir, 'best.tar'), model.module.binarybranch, use_module=True)
        utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)
    elif train_mode ==1:
        logger.info('loading pretrained unary network component..........')
        unary_branch_dir = config['unary_branch_dir']
        utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)
    elif train_mode == 2:
        logger.info('loading pretrained unary network component..........')
        unary_branch_dir = config['unary_branch_dir']
        utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)
    elif train_mode == 3:
        logger.info('loading pretrained amgn ..........')
        agmn_dir = config['agmn_dir']
        utils.load_checkpoint(os.path.join(agmn_dir, 'best.tar'), model)        
    else:
        raise("Wrong parameter for train mode!")


    # print(model)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ----------------------------- optimizer prepration --------------------------------#
    # if train_mode == 1:
    #     logger.info('Loading optimizer...')
    #     optimizer_param = config['optimizer']
    #     optimizer = getattr(optim, optimizer_param['name'])(model.module.binarybranch.parameters(), **optimizer_param['param'])
    #     logger.info('- done.')
    # elif train_mode == 0:
    #     logger.info('Loading optimizer...')
    #     optimizer_param = config['optimizer']
    #     optimizer = getattr(optim, optimizer_param['name'])(model.parameters(), **optimizer_param['param'])
    #     logger.info('- done.')
    # else:
    #     raise("Not implemented yet.")

    if train_mode == 1 or train_mode == 2:
        logger.info('Loading optimizer...')
        optimizer_param = config['optimizer']
        optimizer = getattr(optim, optimizer_param['name'])(model.module.binarybranch.parameters(), **optimizer_param['param'])
        logger.info('- done.')
    elif train_mode == 0 or train_mode == 3:
        logger.info('Loading optimizer...')
        optimizer_param = config['optimizer']
        optimizer = getattr(optim, optimizer_param['name'])(model.parameters(), **optimizer_param['param'])
        logger.info('- done.')  
    else:
        raise("Not implemented yet.")


    # ----------------------------- loss function -------------------------#
    logger.info('Loading loss function...')
    loss_fn = getattr(models, config['loss_fn']['name'])
    logger.info('--done')

    num_epochs = config['num_epochs']
    loss_coef = config.get("loss_coef",None)

    train_and_evaluate(model, dataloaders, optimizer, loss_fn, model_dir, device, gdtt_all_label_dict, num_epochs=num_epochs, restore_file_path=None, train_mode=train_mode, loss_coef=loss_coef)