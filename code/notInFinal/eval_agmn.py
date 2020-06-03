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

import torch.nn.functional as F

def confirm_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate(model, dataloaders, model_dir, device, gdtt_all_label_dict):

    log_train = open(os.path.join(model_dir, 'log_train.txt'), 'a')
    log_valid = open(os.path.join(model_dir, 'log_valid.txt'), 'a')

    evaluate_datasets = ['valid','test']

    for eval_dataset in evaluate_datasets:
        if eval_dataset == 'valid':
            print('validation......')
        else:
            print('testing.........')
        model.eval()
        pred_label_dict = {}
        with torch.no_grad():
            with tqdm(total=len(dataloaders[eval_dataset])) as t:
                for batch_id, data in enumerate(dataloaders[eval_dataset]):

                    image = data['image'].to(device)
                    batch_size = image.shape[0]
                    gdtt_heatmap = data['heatmap']
                    gm_kernels = data['gm_kernels'].to(device)
                    bias = torch.ones(batch_size,40)*0.0000000004
                    bias = bias.to(device)
                    labels = data['label']

                    _, pred_7 = model(image)  # batch_size x 7 x 21 x 46 x 46

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

                    # pred_6 = pred_6.cpu()
                    # for image_i in range(len(image_names)):
                    #     predicted_label = utils.get_labels_from_heatmap(pred_6[image_i, -1, ...], original_image_sizes[image_i])
                    #     pred_label_dict[image_names[image_i]] = {}
                    #     pred_label_dict[image_names[image_i]]['pred_label'] =  predicted_label
                    #     pred_label_dict[image_names[image_i]]['resol'] =  float(original_image_sizes[image_i][0])


                    t.update()
            confirm_dir(os.path.join(model_dir, eval_dataset+'_set'))
            with open(os.path.join(model_dir, eval_dataset+'_set','pred_labels.json'),'w') as f:
                json.dump(pred_label_dict, f)

            ###calculate pck
            cur_pck = 0.0
            for image_name in pred_label_dict.keys():
                pred_label = pred_label_dict[image_name]['pred_label']
                pred_resol = pred_label_dict[image_name]['resol']
                gdtt_label = gdtt_all_label_dict[image_name]
                pck = utils.PCK(pred_label, gdtt_label, pred_resol, sigma=0.04)
                cur_pck = cur_pck + pck
            ave_acc = cur_pck/len(pred_label_dict)
            logger.info(eval_dataset + ' accuracy ' + str(ave_acc))




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
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M-%S")
    if  config['joint_trained']==True:
        model_dir = os.path.join('../checkpoint', config['data_loader']['name'], config['model_name'], 'eval_'+'joint_trained', date_time)
    
    else:
        if config.get('guided_binary',False) == True:
            model_dir = os.path.join('../checkpoint', config['data_loader']['name'], config['model_name'], 'eval_'+'sep_guided_trained', date_time)
        else:
            model_dir = os.path.join('../checkpoint', config['data_loader']['name'], config['model_name'], 'eval_'+'sep_trained', date_time)

    # if  config['joint_trained']==False:
    #     model_dir = os.path.join('../checkpoint',config['data_loader']['name'], config['model_name'], 'eval_'+'sep_trained', date_time)
    # else:
    #     model_dir = os.path.join('../checkpoint',config['data_loader']['name'], config['model_name'], 'eval_'+'joint_trained', date_time)


    # if config['data_loader']['name'] == 'UCIHandDataset':
    #     if  config['joint_trained']==False:
    #         model_dir = os.path.join('../checkpoint_UCI', config['model_name'], 'eval_'+'sep_trained', date_time)
    #     else:
    #         model_dir = os.path.join('../checkpoint_UCI', config['model_name'], 'eval_'+'joint_trained', date_time)
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

    #### load trained model
    if config['joint_trained'] or config.get('guided_binary', 'False'):
        logger.info('loading the whole agmn network ..........')
        agmn_dir = config['agmn_dir']
        utils.load_checkpoint(os.path.join(agmn_dir, 'best.tar'), model, use_module=False)
    else:
        logger.info('loading network components separately..........')
        binary_branch_dir = config['binary_branch_dir']
        unary_branch_dir = config['unary_branch_dir']
        utils.load_checkpoint(os.path.join(binary_branch_dir, 'best.tar'), model.module.binarybranch, use_module=True)
        utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)

    # #### load trained model
    # if config['joint_trained']:
    #     logger.info('loading the joint trained agmn network ..........')
    #     agmn_dir = config['agmn_dir']
    #     utils.load_checkpoint(os.path.join(agmn_dir, 'best.tar'), model, use_module=False)
    # else:
    #     logger.info('loading network components separately..........')
    #     binary_branch_dir = config['binary_branch_dir']
    #     unary_branch_dir = config['unary_branch_dir']
    #     utils.load_checkpoint(os.path.join(binary_branch_dir, 'best.tar'), model.module.binarybranch, use_module=True)
    #     utils.load_checkpoint(os.path.join(unary_branch_dir, 'best.tar'), model.module.unarybranch, use_module=True)
    
    evaluate(model, dataloaders, model_dir, device, gdtt_all_label_dict)
