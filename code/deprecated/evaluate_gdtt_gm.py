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

class TreeModel():
    """
        0---1---2---3---4
        |---5---6---7---8
        |---9---10---11---12
        |---13---14---15---16
        |---17---18---19---20
    """
    def __init__(self):
        super().__init__()
        self.kernel_size = 45
        self.directed_edges = [                  # start of schedule from leaves to root
                                [[4,3], []],     # edge 0, [[from_joint=4, to_joint=3],[empty other incoming edge]]
                                [[3,2], [0]],    # edge 1
                                [[2,1], [1]],    # edge 2
                                [[1,0], [2]],    # edge 3

                                [[8,7], []],     # edge 4
                                [[7,6], [4]],    # edge 5
                                [[6,5], [5]],    # edge 6
                                [[5,0], [6]],    # edge 7

                                [[12,11], []],   # edge 8
                                [[11,10], [8]],  # edge 9
                                [[10,9], [9]],   # edge 10
                                [[9,0], [10]],   # edge 11

                                [[16,15], []],   # edge 12
                                [[15,14], [12]], # edge 13
                                [[14,13], [13]], # edge 14
                                [[13,0], [14]],  # edge 15

                                [[20,19], []],   # edge 16
                                [[19,18], [16]], # edge 17
                                [[18,17], [17]], # edge 18
                                [[17,0], [18]],  # edge 19   # end of schedule from leaves to root

                                [[0,1], [7, 11, 15, 19]], # edge 20  # start of schedule from root to leaves
                                [[1,2], [20]],   # edge 21
                                [[2,3], [21]],   # edge 22
                                [[3,4], [22]],   # edge 23

                                [[0,5], [3,11,15,19]],  # edge 24
                                [[5,6], [24]],   # edge 25
                                [[6,7], [25]],   # edge 26
                                [[7,8], [26]],   # edge 27

                                [[0,9], [3,7,15,19]],   # edge 28
                                [[9,10], [28]],  # edge 29
                                [[10,11], [29]], # edge 30
                                [[11,12], [30]], # edge 31

                                [[0,13], [3,7,11,19]],   # edge 32
                                [[13,14], [32]], # edge 33
                                [[14,15], [33]], # edge 34
                                [[15,16], [34]], # edge 35

                                [[0,17], [3,7,11,15]],   # edge 36
                                [[17,18], [36]], # edge 37
                                [[18,19], [37]], # edge 38
                                [[19,20], [38]]  # edge 39         # end of schedule from root to leaves 
                            ]
        self.incoming_edges_list=[                                 # this list records the incoming edges for each joint
                                    [3,7,11,15,19],  # incoming edges for joint 0
                                    [2, 20],         # joint 1
                                    [1, 21],
                                    [0, 22],
                                    [23],

                                    [6, 24],         # joint 5
                                    [5, 25],
                                    [4, 26],
                                    [27],

                                    [10, 28],        # joint 9
                                    [9,29],
                                    [8,30],
                                    [31],

                                    [14,32],         # joint 13
                                    [13,33],
                                    [12,34],
                                    [35],

                                    [18, 36],        # joint 17
                                    [17, 37],
                                    [16, 38],
                                    [39]
                            ]

    def mul_by_log_exp(self, *args):
        """
        Performs multiplication by taking log first then take exp.
        Returns a scaled product of the args.

        Args:
            bs x 1 x 45 x 45
        """
        # log_y = torch.log(args[0]+1e-16)
        # for x in args[1:]:
        #     log_y = log_y + torch.log(x+1e-16)
        log_y = torch.log(torch.stack(args)+1e-32)
        log_y = torch.sum(log_y, dim=0)
        
        # deal with precision
        max_last_dim = torch.max(log_y,dim=-1,keepdim=True)[0]
        max_last_twodims = torch.max(max_last_dim,dim=-2,keepdim=True)[0]
        log_y = log_y - max_last_twodims
        
        return torch.exp(log_y)  

    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """

        return x/torch.sum(x,dim=[-1,-2],keepdim=True)


    def __call__(self, x, kernels, bias):
        """Args:
                x : batch_size x 21 x 46 x 46
        """
        
        batch_size = x.shape[0]

        
        # clamp x to positive
        x = torch.clamp(x, min=1e-32)
        x = self.normalize_prob(x)
        kernels = torch.clamp(kernels, min=1e-32)
        kernels = self.normalize_prob(kernels)

        # push weights to non-negative
        # SoftPlus, no need to softplus here
        # beta = 1
        # kernels = 1/beta * torch.log(1+torch.exp(beta* self.kernels))
        # kernels = self.normalize_prob(kernels)
        # bias = 1/beta * torch.log(1+torch.exp(beta* self.bias))

        ################## message passing ##############################
        # initialize message matrix
        messageMatrix = torch.ones(batch_size, 40, x.shape[-2], x.shape[-1]).to(x.device)  # batch_size x 40 x 46 x 46
        for edge_id in range(len(self.directed_edges)):
            edge = self.directed_edges[edge_id]
            if len(edge[1]) == 0: # message from a leaf node
                from_joint = edge[0][0]
                inter_belief = x[:,from_joint,...]

            elif len(edge[1]) == 1:
                from_joint = edge[0][0]
                incoming_msg = messageMatrix[:, edge[1][0], ...].clone()
                inter_belief = x[:,from_joint,...] * incoming_msg
             

            elif len(edge[1]) == 4:
                from_joint = edge[0][0]
                incoming_msg_0 = messageMatrix[:, edge[1][0], ...].clone()
                incoming_msg_1 = messageMatrix[:, edge[1][1], ...].clone()
                incoming_msg_2 = messageMatrix[:, edge[1][2], ...].clone()
                incoming_msg_3 = messageMatrix[:, edge[1][3], ...].clone()
                inter_belief = self.mul_by_log_exp(x[:,from_joint,...], incoming_msg_0, incoming_msg_1, incoming_msg_2, incoming_msg_3) 

            # print('inter_belief device', inter_belief.device)
            # print('kernels device', kernels.device)
            # print('bias device', bias.device)
            msg_edge = F.conv2d(inter_belief.unsqueeze(0), 
                                kernels[:,edge_id,...].unsqueeze(1),
                                padding = int(self.kernel_size/2), 
                                groups=batch_size).squeeze(0) + bias[:,edge_id].unsqueeze(-1).unsqueeze(-1)  # batch_size x 46 x 46
            messageMatrix[:, edge_id, ...] = self.normalize_prob(msg_edge)

        #print(messageMatrix.shape)

        ############## calculate marginal############################################
        marginal_mat = torch.zeros(x.shape).to(x.device)   # batch_size x 21 x 46 x 46
        for joint_id in range(len(self.incoming_edges_list)):
            incoming_edges = self.incoming_edges_list[joint_id]
            msgs = []
            for incoming_edge in incoming_edges:
                msgs.append(messageMatrix[:,incoming_edge,...])
            marginal_mat[:, joint_id, ...] = self.mul_by_log_exp(*msgs, x[:,joint_id,...])
        # print(marginal_mat)
        out = self.normalize_prob(marginal_mat)

        return out

def confirm_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_gdtt_gm(model, dataloaders, model_dir, device, gdtt_all_label_dict):

    log_train = open(os.path.join(model_dir, 'log_train.txt'), 'a')
    log_valid = open(os.path.join(model_dir, 'log_valid.txt'), 'a')

    tree_model = TreeModel()

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
                    gdtt_heatmap = torch.stack([gdtt_heatmap]*6, dim=1).to(device)

                    pred_6 = model(image)  # batch_size x 6 x 21 x 46 x 46

                    # get predicted label
                    original_image_sizes = data['original_image_size']
                    image_names = data['image_name']

                    pred_last = pred_6[:,-1,...]  # last stage

                    final_pred = tree_model(pred_last, gm_kernels, bias)

                    final_pred = final_pred.cpu()
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
            confirm_dir(os.path.join(model_dir, eval_dataset+'_gm_pred_labels'))
            with open(os.path.join(model_dir, eval_dataset+'_gm_pred_labels','gdtt_gm_labels.json'),'w') as f:
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
            print(ave_acc)




if __name__ == '__main__':



    # set logger
    # pretrained model dir
    # model_dir = '../checkpoint/OpenPoseHand/03-31-16-57-25/'
    # model_dir = '../checkpoint/OpenPoseHand/04-03-17-22-22/'
    model_dir = '../checkpoint/UnaryBranch/04-15-15-04-42/'
    logger = utils.set_logger(os.path.join(model_dir,'eval_gdtt_gm.log'))


    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='config file for the experiment')
    args = parser.parse_args()

    # get configurations
    with open(os.path.join('./configs', args.config_file),'r') as f:
        config = json.load(f)

    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    utils.load_checkpoint(os.path.join(model_dir, 'best.tar'), model)
    
    evaluate_gdtt_gm(model, dataloaders, model_dir, device, gdtt_all_label_dict)
