"""
Adaptive Graphical Model Network.
"""

__all__ =['AdaptiveGraphicalModelNetwork', 'PairwiseBranch', 'UnaryBranch']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .graphical_model import TreeModel

#----------------------------------building blocks-----------------------------#

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


#-------------------------UnaryBranch, which is actually the CPM baseline --------------#

class UnaryBranch(nn.Module):
    def __init__(self, outc):
        super().__init__()
        self.outc = outc                       # 21
        self.vgg19 = VGG19()                    # backbone
        self.confidence_map_stage1 = Stage1(128, self.outc)
        self.confidence_map_stage2 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage3 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage4 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage5 = Stage(self.outc + 128, self.outc)
        self.confidence_map_stage6 = Stage(self.outc + 128, self.outc)

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

        return torch.stack([stage1, stage2, stage3, stage4, stage5, stage6], dim=1)


#-----------------PairwiseBranch, which is used to regress the parameters of the graphical models------------#

class PairwiseBranch(nn.Module):
    def __init__(self, outc):
        super().__init__()
        self.outc = outc                       # 40
        self.vgg19 = VGG19()                    # backbone
        self.confidence_map_stage1 = Stage1(128, self.outc)
        self.confidence_map_stage2 = Stage(self.outc + 128 + 21, self.outc)
        self.confidence_map_stage3 = Stage(self.outc + 128 + 21, self.outc)
        self.confidence_map_stage4 = Stage(self.outc + 128 + 21, self.outc)
        self.confidence_map_stage5 = Stage(self.outc + 128 + 21, self.outc)
        self.confidence_map_stage6 = Stage(self.outc + 128 + 21, self.outc)

    def forward(self, image,unary_maps):
        """
        """
        # ************* backbone *************
        features = self.vgg19(image)        # batch size * 128 * 46 * 46
        features = F.interpolate(features, size=(45,45), mode='bilinear', align_corners=False) # batch size * 128 * 45 * 45
        unary_map_1 = F.interpolate(unary_maps[:,0,...], size=(45,45), mode='bilinear', align_corners=False)
        unary_map_2 = F.interpolate(unary_maps[:,1,...], size=(45,45), mode='bilinear', align_corners=False)
        unary_map_3 = F.interpolate(unary_maps[:,2,...], size=(45,45), mode='bilinear', align_corners=False)
        unary_map_4 = F.interpolate(unary_maps[:,3,...], size=(45,45), mode='bilinear', align_corners=False)
        unary_map_5 = F.interpolate(unary_maps[:,4,...], size=(45,45), mode='bilinear', align_corners=False)

        # ************* CM stage *************
        stage1 = self.confidence_map_stage1(features)   # batch size * 21 * 46 * 46
        stage2 = self.confidence_map_stage2(torch.cat([features, stage1, unary_map_1], dim=1))
        stage3 = self.confidence_map_stage3(torch.cat([features, stage2, unary_map_2], dim=1))
        stage4 = self.confidence_map_stage4(torch.cat([features, stage3, unary_map_3], dim=1))
        stage5 = self.confidence_map_stage5(torch.cat([features, stage4, unary_map_4], dim=1))
        stage6 = self.confidence_map_stage6(torch.cat([features, stage5, unary_map_5], dim=1))

        return torch.stack([stage1, stage2, stage3, stage4, stage5, stage6], dim=1)


##---------------------Adaptive Graphical Model Network (AGMN)------------------------------------#

class AdaptiveGraphicalModelNetwork(nn.Module):
    def __init__(self, joint_outc=21, gm_outc=40):
        super().__init__()


        self.joint_outc = joint_outc           # 21
        self.gm_outc = gm_outc                 # 40 
        
        self.unarybranch = UnaryBranch(joint_outc)
        self.pairwisebranch = PairwiseBranch(gm_outc)
        self.treemodel = TreeModel()

    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """
        return x/(torch.sum(x,dim=[-1,-2],keepdim=True)+1e-16)

    def forward(self, image):
        """
        :param image:  4D Tensor batch_size * 3 * 368 * 368
        :return:
        """
        inter_score_maps = self.unarybranch(image)
        all_gm_kernels = self.pairwisebranch(image, inter_score_maps)

        last_score_map = inter_score_maps[:,-1,...]
        last_gm_kernel = all_gm_kernels[:,-1,...]

        # clamp x to positive
        last_score_map = torch.clamp(last_score_map, min=1e-32)
        last_score_map = self.normalize_prob(last_score_map)
        # push weights to non-negative
        last_gm_kernel = torch.clamp(last_gm_kernel, min=1e-16)
        last_gm_kernel = self.normalize_prob(last_gm_kernel)

        bias = torch.ones(image.shape[0],40)*0.0000000004
        bias = bias.to(image.device)

        final_score = self.treemodel(last_score_map, last_gm_kernel, bias)               
        out_scores = torch.cat((inter_score_maps, final_score.unsqueeze(1)), dim=1)
        return all_gm_kernels, out_scores




if __name__ == "__main__":
    net = AdaptiveGraphicalModelNetwork(21)
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
