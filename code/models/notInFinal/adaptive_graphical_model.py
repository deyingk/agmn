"""
Open pose version of CPM.
"""

__all__ =['AdaptiveGraphicalModel']

import torch
import torch.nn as nn
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

        return x/(torch.sum(x,dim=[-1,-2],keepdim=True)+1e-16)


    def __call__(self, x, kernels, bias):
        """Args:
                x : batch_size x 21 x 46 x 46
        """
        
        batch_size = x.shape[0] 
        # clamp x to positive
        x = torch.clamp(x, min=1e-32)
        x = self.normalize_prob(x)


        # push weights to non-negative
        # SoftPlus, no need to softplus here
        kernels = torch.clamp(kernels, min=1e-16)
        kernels = self.normalize_prob(kernels)
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


class ConvBlock3x3(nn.Module):
    def __init__(self, in_c, outc):
        super(ConvBlock3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_c, outc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(outc * 3, outc, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return self.conv4(torch.cat([x1, x2, x3], dim=1))


class Stage(nn.Module):
    def __init__(self, in_c, outc):
        """
        CNNs for inference at Stage t (t>=2)
        :param outc:
        """
        super(Stage, self).__init__()
        self.Mconv1 = ConvBlock3x3(in_c, 128)
        self.Mconv2 = ConvBlock3x3(128, 128)
        self.Mconv3 = ConvBlock3x3(128, 128)
        self.Mconv4 = ConvBlock3x3(128, 128)
        self.Mconv5 = ConvBlock3x3(128, 128)
        self.Mconv6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv7 = nn.Conv2d(128, outc, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.Mconv1(x))
        x = self.relu(self.Mconv2(x))
        x = self.relu(self.Mconv3(x))
        x = self.relu(self.Mconv4(x))
        x = self.relu(self.Mconv5(x))
        x = self.relu(self.Mconv6(x))
        x = self.Mconv7(x)
        return x


class Stage1(nn.Module):
    def __init__(self, inc, outc):
        super(Stage1, self).__init__()
        self.stage1_1 = nn.Conv2d(inc, 512, kernel_size=3, padding=1)
        self.stage1_2 = nn.Conv2d(512, outc, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: feature map   4D Tensor   batch size * 128 * 46 * 46
        :return: x              4D Tensor   batch size * 21  * 46 * 46
        """
        x = self.relu(self.stage1_1(x))  # batch size * 512 * 46 * 46
        x = self.stage1_2(x)             # batch size * 21 * 46 * 46

        return x


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1_stage1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2_stage1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3_stage1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))  # 64
        x = self.relu(self.conv1_2(x))  # 64
        x = self.pool1_stage1(x)

        x = self.relu(self.conv2_1(x))  # 128
        x = self.relu(self.conv2_2(x))  # 128
        x = self.pool2_stage1(x)

        x = self.relu(self.conv3_1(x))  # 256
        x = self.relu(self.conv3_2(x))  # 256
        x = self.relu(self.conv3_3(x))  # 256
        x = self.relu(self.conv3_4(x))  # 256
        x = self.pool3_stage1(x)

        x = self.relu(self.conv4_1(x))  # 512
        x = self.relu(self.conv4_2(x))  # 512
        x = self.relu(self.conv4_3(x))  # 512
        x = self.relu(self.conv4_4(x))  # 512

        x = self.relu(self.conv5_1(x))  # 512
        x = self.relu(self.conv5_2(x))  # 512
        x = self.relu(self.conv5_3(x))  # 128

        return x  # Batchsize * 128 * H/8 * W/8


class AdaptiveGraphicalModel(nn.Module):
    def __init__(self, outc, gm_outc=40):
        super().__init__()

        self.outc = outc                       # 21
        self.gm_outc = gm_outc
        self.vgg19 = VGG19()                    # backbone     

        # network for cpm
        self.confidence_map_stage1 = Stage1(128, self.outc)
        self.confidence_map_stage2 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage3 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage4 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage5 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage6 = Stage(self.outc + 128, self.outc)

        # network for learning GM parameters
        # self.upsample = torch.nn.Upsample(size=(45, 45), mode='bilinear',align_corners=False)
        self.gm_net_stage1 = Stage1(128, self.gm_outc)
        self.gm_net_stage2 = Stage(self.gm_outc + 128, self.gm_outc)
        self.gm_net_stage3 = Stage(self.gm_outc + 128, self.gm_outc)   

        self.treemodel = TreeModel()

    def forward(self, image):
        """
        :param image:  4D Tensor batch_size * 3 * 368 * 368
        :return:
        """
        # ************* backbone *************
        features = self.vgg19(image)        # batch size * 128 * 46 * 46

        # ************* CM stage *************
        stage1 = self.confidence_map_stage1(features)   # batch size * 21 * 46 * 46
        stage2 = self.confidence_map_stage2(torch.cat([features, stage1], dim=1))
        stage3 = self.confidence_map_stage3(torch.cat([features, stage2], dim=1))
        stage4 = self.confidence_map_stage4(torch.cat([features, stage3], dim=1))
        stage5 = self.confidence_map_stage5(torch.cat([features, stage4], dim=1))
        stage6 = self.confidence_map_stage6(torch.cat([features, stage5], dim=1))
        

        # Adaptive GM
        # batch size * 128* 46 * 46 ---> batch size * 128 * 45 * 45
        features = F.interpolate(features, size=(45,45), mode='bilinear', align_corners=False)
        gm_stage1 = self.gm_net_stage1(features) 
        gm_stage2 = self.gm_net_stage2(torch.cat([features, gm_stage1], dim=1))
        gm_stage3 = self.gm_net_stage3(torch.cat([features, gm_stage2], dim=1))

        
        all_gm_kernels = torch.stack([gm_stage1, gm_stage2, gm_stage3], dim=1)
        # beta = 1
        # all_gm_kernels = torch.stack([1/beta * torch.log(1+torch.exp(beta* gm_stage1)),
        #                              1/beta * torch.log(1+torch.exp(beta* gm_stage2)),
        #                              1/beta * torch.log(1+torch.exp(beta* gm_stage3))], dim=1)

        # perform inference
        # kernels = F.relu(gm_stage3) 
        # kernels = torch.clamp(gm_stage3, min=1e-16)
        bias = torch.ones(image.shape[0],40)*0.0000000004
        bias = bias.to(image.device)
        final_score = self.treemodel(stage6, gm_stage3, bias)

        out_scores = torch.stack([stage1, stage2, stage3, stage4, stage5, stage6, final_score], dim=1)
        return all_gm_kernels, out_scores


if __name__ == "__main__":
    net = AdaptiveGraphicalModel(21)
    #net = Stage1(3, 128)
    print ('test forward ......')
    x = torch.randn(2, 3, 368, 368)
    #x = torch.randn(2, 3, 46, 46)
    print (x.shape)
    all_gm_kernels, out_scores = net(x)            # (2, 6, 21, 46, 46)
    print (all_gm_kernels.shape)
    print (out_scores.shape)
    y = torch.sum(out_scores)
    y.backward()
