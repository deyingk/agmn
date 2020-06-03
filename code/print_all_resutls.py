"""
This script would print out the sumary results.

"""

import numpy as np


num_to_print = 10

print("\nPCK_sigma:")
print('\t'.join(map(str,[i*0.01 for i in range(1,num_to_print+1)])),'\n')
# load baseline result
pcks_cpm = np.load('../checkpoint/CMUPanopticHandDataset/UnaryBranch/test_set/test_pcks.npy')
pcks_cpm = pcks_cpm[:num_to_print]
print('CPM baseline:')
print ('\t'.join(map(str, np.around(pcks_cpm*100, decimals = 3) )),'\n')


# load intermediate result
pcks_inter = np.load('../checkpoint/CMUPanopticHandDataset/AdaptiveGraphicalModelNetwork/PairwiseBranchTrained/test_set/test_pcks.npy')
pcks_inter = pcks_inter[:num_to_print]
print('Pairwise branch trained seperately:')
print ('\t'.join(map(str, np.around(pcks_inter*100, decimals = 3))),'\n')

# load final result, training_coef:[1.0, 0, 0]
# pcks_final = np.load('../checkpoint/CMUPanopticHandDataset/AdaptiveGraphicalModelNetwork/JointlyTrained/08-28-15-30-18/test_set/test_pcks.npy')

# pcks_final = np.load('../checkpoint/CMUPanopticHandDataset/AdaptiveGraphicalModelNetwork/JointlyTrained/08-26-20-57-38/test_set/test_pcks.npy')
pcks_final = np.load('../checkpoint/CMUPanopticHandDataset/AdaptiveGraphicalModelNetwork/JointlyTrained/08-26-14-18-31/test_set/test_pcks.npy')
pcks_final = pcks_final[:num_to_print]
print('Jointly trained AGMN, with coefficients: [1.0, 0.0, 0.0]')
print ('\t'.join(map(str, np.around(pcks_final*100, decimals = 3))),'\n')

# print out improvement
improvement = np.array([pcks_final[i] - pcks_cpm[i] for i in range(len(pcks_cpm))])
improvement = improvement[:num_to_print]
print('Improvements:')
print ('\t'.join(map(str, np.around(improvement*100, decimals = 3))),'\n')
