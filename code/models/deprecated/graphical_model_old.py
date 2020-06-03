
import torch
from torch import nn
import torch.nn.functional as F

class TreeModel(nn.Module):
    """
        0---1---2---3---4
        |---5---6---7---8
        |---9---10---11---12
        |---13---14---15---16
        |---17---18---19---20
    """
    def __init__(self):
        super().__init__()
        batch_size = 4
        self.kernel_size = 45
        self.kernels = nn.Parameter(torch.randn(batch_size, 40, int(self.kernel_size), int(self.kernel_size))) # batch_size, 20x2 edges (upwards and downwards). 45x45 resolution
        self.bias = nn.Parameter(torch.zeros(batch_size,40))
        self.edges = {
                    'upwards': [[4,3], [3,2], [2,1], [1,0],       # thumb
                                [8,7], [7,6], [6,5], [5,0],       # index finger    
                                [12,11], [11,10], [10,9],[9,0],   # middle finger
                                [16,15],[15,14],[14,13],[13,0],   # ring finger
                                [20,19],[19,18],[18,17],[17,0]    # little finger
                    ],
                    'downwards':[[0,1], [1,2], [2,3], [3,4],
                                 [0,5], [5,6], [6,7], [7,8],
                                 [0,9], [9,10],[10,11],[11,12],
                                 [0,13], [13,14],[14,15],[15,16],
                                 [0,17],[17,18],[18,19],[19,20]
                    ]
        }
    def mul_by_log_exp(sefl, *args):
        """
        Performs multiplication by taking log first then take exp.
        Returns a scaled product of the args.

        Args:
            bs x 1 x 45 x 45
        """
        # log_y = torch.log(args[0]+1e-16)
        # for x in args[1:]:
        #     log_y = log_y + torch.log(x+1e-16)
        log_y = torch.log(torch.stack(args)+1e-16)
        log_y = torch.sum(log_y, dim=0)
        
        # # deal with precision
        # max_last_dim = torch.max(log_y,dim=-1,keepdim=True)[0]
        # max_last_twodims = torch.max(max_last_dim,dim=-2,keepdim=True)[0]
        # log_y = log_y - max_last_twodims
        
        return torch.exp(log_y)  

    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """

        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def forward(self, x):
        """Args:
                x : batch_size x 21 x 46 x 46
        """
        
        batch_size = x.shape[0]

        # clamp x to positive
        x = torch.clamp(x, min=1e-8)

        # push weights to non-negative
        # SoftPlus
        beta = 1
        kernels = 1/beta * torch.log(1+torch.exp(beta* self.kernels))
        kernels = self.normalize_prob(kernels)
        bias = 1/beta * torch.log(1+torch.exp(beta* self.bias))

        ################## message passing from leaves to root #####################################################

        # thumb
        # message sent from 4th joint to 3rd joint, 0-th edge
        print(x[:,4,...].unsqueeze(0).shape, self.kernels[:,0,...].unsqueeze(1).shape,bias[:,0].shape)
        edge_id = 0
        from_joint = 4
        to_joint = 3
        m_4_3 = F.conv2d(x[:,from_joint,...].unsqueeze(0), 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2), 
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46
        # edge 1
        edge_id = 1
        b_3 = x[:,3,...].unsqueeze(0) * m_4_3  # 1 x batch_size x 46 x 46
        m_3_2 = F.conv2d(b_3, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 2
        edge_id = 2
        b_2 = x[:,2,...].unsqueeze(0) * m_3_2  # 1 x batch_size x 46 x 46
        m_2_1 = F.conv2d(b_2, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 3
        edge_id = 3
        b_1 = x[:,1,...].unsqueeze(0) * m_2_1  # 1 x batch_size x 46 x 46
        m_1_0 = F.conv2d(b_1, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        #### index finger ####
        edge_id = 4
        from_joint = 8
        to_joint = 7
        m_8_7 = F.conv2d(x[:,from_joint,...].unsqueeze(0), 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2), 
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 5
        edge_id = 5
        b_7 = x[:,7,...].unsqueeze(0) * m_8_7  # 1 x batch_size x 46 x 46
        m_7_6 = F.conv2d(b_7, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 6
        edge_id = 6
        b_6 = x[:,6,...].unsqueeze(0) * m_7_6  # 1 x batch_size x 46 x 46
        m_6_5 = F.conv2d(b_6, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 7
        edge_id = 7
        b_5 = x[:,5,...].unsqueeze(0) * m_6_5  # 1 x batch_size x 46 x 46
        m_5_0 = F.conv2d(b_5, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        #### middle finger ####
        edge_id = 8
        from_joint = 12
        to_joint = 11
        m_12_11 = F.conv2d(x[:,from_joint,...].unsqueeze(0), 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2), 
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 9
        edge_id = 9
        b_11 = x[:,11,...].unsqueeze(0) * m_12_11  # 1 x batch_size x 46 x 46
        m_11_10 = F.conv2d(b_11, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 10
        edge_id = 10
        b_10 = x[:,10,...].unsqueeze(0) * m_11_10  # 1 x batch_size x 46 x 46
        m_10_9 = F.conv2d(b_10, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 11
        edge_id = 11
        b_9 = x[:,9,...].unsqueeze(0) * m_10_9  # 1 x batch_size x 46 x 46
        m_9_0 = F.conv2d(b_9, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        #### ring finger ####
        edge_id = 12
        from_joint = 16
        to_joint = 15
        m_16_15 = F.conv2d(x[:,from_joint,...].unsqueeze(0), 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2), 
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 13,  15->14
        edge_id = 13
        b_15 = x[:,15,...].unsqueeze(0) * m_16_15  # 1 x batch_size x 46 x 46
        m_15_14 = F.conv2d(b_15, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 14,  14->13
        edge_id = 14
        b_14 = x[:,14,...].unsqueeze(0) * m_15_14  # 1 x batch_size x 46 x 46
        m_14_13 = F.conv2d(b_14, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 15,  13->0
        edge_id = 15
        b_13 = x[:,13,...].unsqueeze(0) * m_14_13  # 1 x batch_size x 46 x 46
        m_13_0 = F.conv2d(b_13, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        #### pinky finger ####
        edge_id = 16
        from_joint = 20
        to_joint = 19
        m_20_19 = F.conv2d(x[:,from_joint,...].unsqueeze(0), 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2), 
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 17,  19->18
        edge_id = 17
        b_19 = x[:,19,...].unsqueeze(0) * m_20_19  # 1 x batch_size x 46 x 46
        m_19_18 = F.conv2d(b_19, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 18,  18->17
        edge_id = 18
        b_18 = x[:,18,...].unsqueeze(0) * m_19_18  # 1 x batch_size x 46 x 46
        m_18_17 = F.conv2d(b_18, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        # edge 19,  17->0
        edge_id = 19
        b_17 = x[:,17,...].unsqueeze(0) * m_18_17  # 1 x batch_size x 46 x 46
        m_17_0 = F.conv2d(b_17, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46

        ################## message passing from root to leaves #####################################################
        ###thumb###
        #edge 20, 0->1 
        edge_id = 20
        b_0_1 = self.mul_by_log_exp(x[:,0,...].unsqueeze(0), m_5_0, m_9_0, m_13_0, m_17_0) # 1 x batch_size x 46 x 46
        m_0_1 = F.conv2d(b_0_1, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46     

        #edge 21, 1->2 
        edge_id = 21
        b_1_2 = x[:,1,...].unsqueeze(0) * m_0_1 # 1 x batch_size x 46 x 46
        m_1_2 = F.conv2d(b_1_2, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 22, 2->3 
        edge_id = 22
        b_2_3 = x[:,2,...].unsqueeze(0) * m_1_2 # 1 x batch_size x 46 x 46
        m_2_3 = F.conv2d(b_2_3, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 23, 3->4 
        edge_id = 23
        b_3_4 = x[:,3,...].unsqueeze(0) * m_2_3 # 1 x batch_size x 46 x 46
        m_3_4 = F.conv2d(b_3_4, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        ###index finger###
        #edge 24, 0->5 
        edge_id = 24
        b_0_5 = self.mul_by_log_exp(x[:,0,...].unsqueeze(0), m_1_0, m_9_0, m_13_0, m_17_0) # 1 x batch_size x 46 x 46
        m_0_5 = F.conv2d(b_0_5, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46     

        #edge 25, 5->6 
        edge_id = 25
        b_5_6 = x[:,5,...].unsqueeze(0) * m_0_5 # 1 x batch_size x 46 x 46
        m_5_6 = F.conv2d(b_5_6, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 26, 6->7 
        edge_id = 26
        b_6_7 = x[:,6,...].unsqueeze(0) * m_5_6 # 1 x batch_size x 46 x 46
        m_6_7 = F.conv2d(b_6_7, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 27, 7->8 
        edge_id = 27
        b_7_8 = x[:,7,...].unsqueeze(0) * m_6_7 # 1 x batch_size x 46 x 46
        m_7_8 = F.conv2d(b_7_8, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 

        ###midlle finger###
        #edge 28, 0->9 
        edge_id = 28
        b_0_9 = self.mul_by_log_exp(x[:,0,...].unsqueeze(0), m_1_0, m_5_0, m_13_0, m_17_0) # 1 x batch_size x 46 x 46
        m_0_9 = F.conv2d(b_0_9, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46     

        #edge 29, 9->10 
        edge_id = 29
        b_9_10 = x[:,9,...].unsqueeze(0) * m_0_9 # 1 x batch_size x 46 x 46
        m_9_10 = F.conv2d(b_9_10, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 30, 10->11 
        edge_id = 30
        b_10_11 = x[:,10,...].unsqueeze(0) * m_9_10 # 1 x batch_size x 46 x 46
        m_10_11 = F.conv2d(b_10_11, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 31, 11->12 
        edge_id = 31
        b_11_12 = x[:,11,...].unsqueeze(0) * m_10_11 # 1 x batch_size x 46 x 46
        m_11_12 = F.conv2d(b_11_12, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 

        ###ring finger###
        #edge 32, 0->13 
        edge_id = 32
        b_0_13 = self.mul_by_log_exp(x[:,0,...].unsqueeze(0), m_1_0, m_5_0, m_9_0, m_17_0) # 1 x batch_size x 46 x 46
        m_0_13 = F.conv2d(b_0_13, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46     

        #edge 33, 13->14 
        edge_id = 33
        b_13_14 = x[:,13,...].unsqueeze(0) * m_0_13 # 1 x batch_size x 46 x 46
        m_13_14 = F.conv2d(b_13_14, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 34, 14->15 
        edge_id = 34
        b_14_15 = x[:,14,...].unsqueeze(0) * m_13_14 # 1 x batch_size x 46 x 46
        m_14_15 = F.conv2d(b_14_15, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 35, 15->16 
        edge_id = 35
        b_15_16 = x[:,15,...].unsqueeze(0) * m_14_15 # 1 x batch_size x 46 x 46
        m_15_16 = F.conv2d(b_15_16, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 

        ###pinky finger###
        #edge 36, 0->17 
        edge_id = 36
        b_0_17 = self.mul_by_log_exp(x[:,0,...].unsqueeze(0), m_1_0, m_5_0, m_9_0, m_13_0) # 1 x batch_size x 46 x 46
        m_0_17 = F.conv2d(b_0_17, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46     

        #edge 37, 17->18 
        edge_id = 37
        b_17_18 = x[:,17,...].unsqueeze(0) * m_0_17 # 1 x batch_size x 46 x 46
        m_17_18 = F.conv2d(b_17_18, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 38, 18->19 
        edge_id = 38
        b_18_19 = x[:,18,...].unsqueeze(0) * m_17_18 # 1 x batch_size x 46 x 46
        m_18_19 = F.conv2d(b_18_19, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 
        #edge 39, 19->20 
        edge_id = 39
        b_19_20 = x[:,19,...].unsqueeze(0) * m_18_19 # 1 x batch_size x 46 x 46
        m_19_20 = F.conv2d(b_19_20, 
                        kernels[:,edge_id,...].unsqueeze(1),
                        padding = int(self.kernel_size/2),
                        groups=batch_size) + bias[:,edge_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)# 1 x batch_size x 46 x 46 

        
        ############## calculate marginal############################################
        marginal_0 = self.mul_by_log_exp(m_1_0, m_5_0, m_9_0, m_13_0, m_17_0, x[:,0,...].unsqueeze(0))

        marginal_1 = self.mul_by_log_exp(m_0_1, m_2_1, x[:,1,...].unsqueeze(0))
        marginal_2 = self.mul_by_log_exp(m_1_2, m_3_2, x[:,2,...].unsqueeze(0))
        marginal_3 = self.mul_by_log_exp(m_2_3, m_4_3, x[:,3,...].unsqueeze(0))
        marginal_4 = self.mul_by_log_exp(m_3_4, x[:,4,...].unsqueeze(0))

        marginal_5 = self.mul_by_log_exp(m_0_5, m_6_5, x[:,5,...].unsqueeze(0))
        marginal_6 = self.mul_by_log_exp(m_5_6, m_7_6, x[:,6,...].unsqueeze(0))
        marginal_7 = self.mul_by_log_exp(m_6_7, m_8_7, x[:,7,...].unsqueeze(0))
        marginal_8 = self.mul_by_log_exp(m_7_8, x[:,8,...].unsqueeze(0))

        marginal_9 = self.mul_by_log_exp(m_0_9, m_10_9, x[:,9,...].unsqueeze(0))
        marginal_10 = self.mul_by_log_exp(m_9_10, m_11_10, x[:,10,...].unsqueeze(0))
        marginal_11 = self.mul_by_log_exp(m_10_11, m_12_11, x[:,11,...].unsqueeze(0))
        marginal_12 = self.mul_by_log_exp(m_11_12, x[:,12,...].unsqueeze(0))

        marginal_13 = self.mul_by_log_exp(m_0_13, m_14_13, x[:,13,...].unsqueeze(0))
        marginal_14 = self.mul_by_log_exp(m_13_14, m_15_14, x[:,14,...].unsqueeze(0))
        marginal_15 = self.mul_by_log_exp(m_14_15, m_16_15, x[:,15,...].unsqueeze(0))
        marginal_16 = self.mul_by_log_exp(m_15_16, x[:,16,...].unsqueeze(0))

        marginal_17 = self.mul_by_log_exp(m_0_17, m_18_17, x[:,17,...].unsqueeze(0))
        marginal_18 = self.mul_by_log_exp(m_17_18, m_19_18, x[:,18,...].unsqueeze(0))
        marginal_19 = self.mul_by_log_exp(m_18_19, m_20_19, x[:,19,...].unsqueeze(0))
        marginal_20 = self.mul_by_log_exp(m_19_20, x[:,20,...].unsqueeze(0))




        # 
        out = torch.cat((marginal_0, marginal_1, marginal_2, marginal_3, marginal_4, 
                        marginal_5, marginal_6, marginal_7, marginal_8, 
                        marginal_9, marginal_10, marginal_11, marginal_12,
                        marginal_13, marginal_14, marginal_15, marginal_16,
                        marginal_17, marginal_18, marginal_19, marginal_20
                        ),dim=0).permute(1,0,2,3)

        return self.normalize_prob(out)



if __name__ == '__main__':
    
    treemodel = TreeModel()
    x = torch.randn(4,21,46,46)
    y = treemodel(x)
    print(y.shape)    
