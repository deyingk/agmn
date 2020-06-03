import logging
import numpy as np
import os
import torch
from collections import OrderedDict

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # ensure a handler is added only once
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    return logger


def load_checkpoint(checkpoint, model, use_module=False, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model_state_dict = checkpoint['model_state_dict']
    if use_module:
        # create new OrderedDict that does not contain `module.`
        new_model_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] # remove `module.`
            new_model_state_dict[name] = v
        model_state_dict = new_model_state_dict
        
    model.load_state_dict(model_state_dict)
    print('the best epoch is {}'.format(checkpoint['epoch']))

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def get_labels_from_heatmap(predict_heatmap, original_image_size):
    """
    get predicted labels in the original image from predicted heatmap
    
    Args:
        predict_heatmaps:    3D Tensor     21 x 46 x 46
        original_image_size: 1D Tensor, [width, height]

    Returns:
        predicted labels: 
    """
    original_image_size = original_image_size.numpy() # must do this!!!
    # long tensor has wierd behavior of devision
    w, h = original_image_size[0], original_image_size[1]
    # print(w, h)
    label_list = []

    # print(predict_heatmap)

    for i in range(21):
        tmp_pre = np.asarray(predict_heatmap[i].data)  # 2D
        # print(tmp_pre)
        #  get label of original image
        corr = np.where(tmp_pre == np.max(tmp_pre))
        x = corr[1][0] * (w/ 46.0)
        x = int(x)
        y = corr[0][0] * (h/ 46.0)
        y = int(y)

        label_list.append([x, y])  # save img label to json        
    
    return label_list

def PCK(predict, target, label_size=256, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         list 21 * 2     [[x1,y1], [x2,y2], ..., [x21,y21]]
    :param target:          list 21 * 2     [[x1,y1], [x2,y2], ..., [x21,y21]]
    :param label_size:
    :param sigma:
    :return: 0/21, 1/21, ...
    """
    pck = 0
    for i in range(21):
        pre = predict[i]
        tar = target[i]
        dis = np.sqrt((pre[0] - tar[0]) ** 2 + (pre[1] - tar[1]) ** 2)
        if dis < sigma * label_size:
            pck += 1
    return pck / 21.0


def PCKs(predict, target, label_size=256, sigmas=[0.01 * (i+1) for i in range(16)]):
    """
    calculate possibility of correct key point of one single image, for diferent PCK sigmas
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         list 21 * 2     [[x1,y1], [x2,y2], ..., [x21,y21]]
    :param target:          list 21 * 2     [[x1,y1], [x2,y2], ..., [x21,y21]]
    :param label_size:
    :param sigma:
    :return: 
    """
    pcks = np.zeros(len(sigmas))
    for i, sigma in enumerate(sigmas):
        pcks[i] = PCK(predict, target, label_size, sigma)
    return pcks