
__all__ = ['loss_mse','loss_mse_10','loss_mse_30','loss_mse_100', 'loss_kl_div']

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_prob(x):
    """
    Normalize along last two dimensions
    """

    return x/(torch.sum(x, dim=(-1,-2), keepdim=True)+1e-16)


# mse loss
loss_mse = nn.MSELoss()
# usage 
# loss_fn_1(outputs, targets): Compute the MSE loss between the output heatmaps and the target heatmaps.

def loss_mse_10(outputs, targets):
	return 10*loss_mse(outputs, targets)
	
def loss_mse_30(outputs, targets):
	return 30*loss_mse(outputs, targets)

def loss_mse_100(outputs, targets):
	return 100*loss_mse(outputs, targets)

def loss_kl_div(outputs, targets):
	"""
	Args:
		outputs: batch_size x channel_num x resol x resol
		targets: the same size as outputs
	"""
	
	P = F.relu(outputs) + 1e-32
	P = normalize_prob(P)
	Q = targets + 1e-32
	batch_size = outputs.shape[0]
	channel_num = outputs.shape[1]
	kl_div = torch.sum(P*(torch.log(P)-torch.log(Q)))/batch_size/channel_num/1000
	# print(kl_div.shape)
	# print(kl_div)
	return kl_div

def accuracy():
	pass