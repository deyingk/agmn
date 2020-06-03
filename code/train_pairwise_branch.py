### train pairwise branch guided by unary branch.


import json
import os
import argparse
from datetime import datetime
import shutil
import sys
import logging
from tqdm import tqdm
import time

import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import dataset.my_datasets as datasets_module
import models
from utils import utils


############### set random seeds ##############
def set_seeds(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seeds()

def _init_fn(worker_id):
    # This function will be fed to the DataLoader, to make it deterministic
    seed=1024
    np.random.seed(seed)

def confirm_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_and_evaluate(model, dataloaders, optimizer, loss_fn, model_dir, device, gdtt_all_label_dict, num_epochs=100, restore_file_path=None):
    """ Train the model and evaluate the model on validation dataset every epoch.

    Args:
        model: the neural network to be trained
        dataloader:
        optimizer:
        loss_fun:
        model_dir: directory where config, models and logging would be saved
        device:
        gdtt_all_label_dict: a dictionary of ground truth labels for all images
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
    
    # best_val_acc = 0.0
    best_val_loss = 100000.0
    best_epoch = 0



    for epoch in range(num_epochs):
        # run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, num_epochs))
        train_loss = []
        val_loss = []
        #-------------------train model---------------------------------#
        print('training.....')
        model.train()
        model.module.unarybranch.eval()
        train_pred_label_dict = {}
        with tqdm(total=len(dataloaders['train'])) as t:
            for batch_id, data in enumerate(dataloaders['train']):
                image = data['image'].to(device)
                gdtt_gm_kernels = data['gm_kernels']
                gdtt_gm_kernels = torch.stack([gdtt_gm_kernels]*6, dim=1).to(device)

                optimizer.zero_grad()
                gm_kernels_6, pred_7 = model(image)  
                loss = loss_fn(gm_kernels_6, gdtt_gm_kernels)
                loss.backward()
                optimizer.step()

                loss_last = loss_fn(gm_kernels_6[:, -1, ...], gdtt_gm_kernels[:, -1, ...]) # last stage loss
                train_loss.append(loss_last.item())

                t.update()

        #------------------validation------------------------------------#
        print('validation......')
        model.eval()
        val_pred_label_dict = {}
        with torch.no_grad():
            with tqdm(total=len(dataloaders['valid'])) as t:
                for batch_id, data in enumerate(dataloaders['valid']):
                    image = data['image'].to(device)
                    gdtt_gm_kernels = data['gm_kernels']


                    labels = data['label']
                    gdtt_gm_kernels = torch.stack([gdtt_gm_kernels]*6, dim=1).to(device)

                    gm_kernels_6, pred_7 = model(image)
                    loss_last = loss_fn(gm_kernels_6[:,-1,...], gdtt_gm_kernels[:, -1, ...])
                    val_loss.append(loss_last.item())

                    t.update()

        average_train_loss = sum(train_loss)/ len(train_loss)
        average_val_loss = sum(val_loss) / len(val_loss)

        if average_val_loss< best_val_loss:
            best_val_loss = average_val_loss
            best_epoch = epoch
            torch.save({
                        'epoch': epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        #'val_acc': val_acc,
                }, os.path.join(model_dir,'best.tar'))

        torch.save({
                    'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    #'val_acc': val_acc,
            }, os.path.join(model_dir,'epoch_'+str(epoch).zfill(3)+'.tar'))

        # **************** save loss for one epoch ****************

        logger.info('epoch ' + str(epoch))
        logger.info('train epoch loss ' + str(average_train_loss))
        logger.info('validation epoch loss ' + str(average_val_loss))
        # logger.info('valid current pck ' + str(val_acc))
        logger.info('lowest valid loss so far' + str(best_val_loss))
        logger.info('best epoch so far ' + str(best_epoch))
        # log_train.write(str(epoch).zfill(3)+','+str(average_train_loss)+','+str(train_acc) + '\n')
        # log_valid.write(str(epoch).zfill(3)+','+str(average_val_loss)+','+str(val_acc) + '\n')
        log_train.write(str(epoch).zfill(3)+','+str(average_train_loss) + '\n')
        log_valid.write(str(epoch).zfill(3)+','+str(average_val_loss) + '\n')
        log_train.flush()
        log_valid.flush()


def test(model, model_dir, dataloaders, device, gdtt_all_label_dict):
    """
    Test the best trained model on test set.
    Args:
        model:  the model to be tested.
        model_dir:  the directory where the model is saved. 
        dataloaders:
        device:
        gdtt_all_label_dict
    """

    log_test = open(os.path.join(model_dir, 'log_test.txt'), 'w')

    model.eval()
    pred_label_dict = {}
    with torch.no_grad():
        with tqdm(total=len(dataloaders['test'])) as t:
            for batch_id, data in enumerate(dataloaders['test']):

                image = data['image'].to(device)
                batch_size = image.shape[0]
                gdtt_heatmap = data['heatmap']
                labels = data['label']

                gm_kernels_6, pred_7 = model(image) 

                # get predicted label
                final_pred = pred_7[:,-1,...]  # last stage
                final_pred = final_pred.cpu()

                original_image_sizes = data['original_image_size']
                image_names = data['image_name']

                for image_i in range(len(image_names)):
                    predicted_label = utils.get_labels_from_heatmap(final_pred[image_i], original_image_sizes[image_i])
                    pred_label_dict[image_names[image_i]] = {}
                    pred_label_dict[image_names[image_i]]['pred_label'] =  predicted_label
                    pred_label_dict[image_names[image_i]]['resol'] =  float(original_image_sizes[image_i][0])

                t.update()

        confirm_dir(os.path.join(model_dir, 'test'+'_set'))
        with open(os.path.join(model_dir, 'test'+'_set','pred_labels.json'),'w') as f:
            json.dump(pred_label_dict, f)

        ###calculate pcks for sigma in the range from 0.01 to 0.16
        sigmas = [0.01 * (i+1) for i in range(16)]
        cur_pck = np.zeros(len(sigmas))

        for image_name in pred_label_dict.keys():
            pred_label = pred_label_dict[image_name]['pred_label']
            pred_resol = pred_label_dict[image_name]['resol']
            gdtt_label = gdtt_all_label_dict[image_name]
            pck = utils.PCKs(pred_label, gdtt_label, pred_resol, sigmas)
            cur_pck = cur_pck + pck
        ave_acc = cur_pck/len(pred_label_dict)

        logger.info('test' + ' accuracy :' + ', '.join(ave_acc.astype(str)) )
        log_test.write('test' + ' accuracy :' + ', '.join(ave_acc.astype(str)))

        with open(os.path.join(model_dir, 'test'+'_set','test_pcks.npy'),'wb') as f:
            np.save(f, ave_acc)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='config file for the experiment')
    args = parser.parse_args()

    # get configurations
    with open(os.path.join('./configs', args.config_file),'r') as f:
        config = json.load(f)

    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make experiment directory
    model_dir = os.path.join('../checkpoint',config['data_loader']['name'], config['model_name'], 'PairwiseBranchTrained')
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
    dataloaders['train'] = DataLoader(train_set, batch_size=dl_param['batch_size'], shuffle=True, num_workers=dl_param['number_workers'], worker_init_fn=_init_fn)
    #val
    val_set = my_dataset(val_records, config['data']['path'])
    dataloaders['valid'] = DataLoader(val_set, batch_size=dl_param['batch_size'], shuffle=False, num_workers=dl_param['number_workers'], worker_init_fn=_init_fn)
    #test
    test_set = my_dataset(test_records, config['data']['path'])
    dataloaders['test'] = DataLoader(test_set, batch_size=dl_param['batch_size'], shuffle=False, num_workers=dl_param['number_workers'], worker_init_fn=_init_fn)
    logger.info('- done.')

    # ----------------------------- model preparation ----------------------------------#
    logger.info('Loading model...')
    # get model
    model_class = getattr(models,config['model_name'])
    model = model_class()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    # print(model)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    logger.info('loading pretrained unary network component..........')
    unary_branch_dir = config['unary_branch_dir']
    utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)

    # ----------------------------- optimizer prepration --------------------------------#
    logger.info('Loading optimizer...')
    optimizer_param = config['optimizer']
    optimizer = getattr(optim, optimizer_param['name'])(model.module.pairwisebranch.parameters(), **optimizer_param['param'])
    logger.info('- done.')
    
    for param in model.module.unarybranch.parameters():
        param.requires_grad = False

    # ----------------------------- loss function -------------------------#
    logger.info('Loading loss function...')
    loss_fn = getattr(models, config['loss_fn']['name'])
    logger.info('--done')

    num_epochs = config['num_epochs']

    #train
    train_and_evaluate(model, dataloaders, optimizer, loss_fn, model_dir, device, gdtt_all_label_dict, num_epochs=num_epochs, restore_file_path=None)
    # test
    # load the best model
    logger.info('Test the best model \n Loading model...')
    model_class = getattr(models,config['model_name'])
    model = model_class()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    utils.load_checkpoint(os.path.join(model_dir, 'best.tar'), model)
    # perform testing
    test(model, model_dir, dataloaders, device, gdtt_all_label_dict)