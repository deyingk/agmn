import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import json
from PIL import Image
import pickle
import cv2




class Dataset_Hand_1(Dataset):
    """
    This dataset returns {image, label, gdtt heatmap, folder_name, image_name}

    The images are resized to (368,368), and normalized with mean(0.4816, 0.4729, 0.4566)
    and std(0.2153, 0.2329, 0.2698).
    """ 
    
    def __init__(self, list_records, data_path, transform = None, sigma=1, resol=46):
        
        super().__init__()
        self.height = 368
        self.width = 368

        self.list_records = list_records
        self.image_dir = os.path.join(data_path,'images')
        self.label_dir = os.path.join(data_path,'labels')

        self.transform = transform
        self.joints = 21                    # 21 heat maps
        self.sigma = sigma
        self.resol = resol

        self.toTensor = transforms.ToTensor()
        self.normalize_img = transforms.Normalize((0.4816, 0.4729, 0.4566),(0.2153, 0.2329, 0.2698))

    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps

    def __len__(self):
        return len(self.list_records)

    def __getitem__(self, idx):
        
        folder_and_image = self.list_records[idx]
        folder_name = folder_and_image[0]
        image_name = folder_and_image[1]

        img_folder_path = os.path.join(self.image_dir, folder_name)
        label_path = os.path.join(self.label_dir, folder_name)+'.json'
        all_labels = json.load(open(label_path))

        #print(folder_name)
        #print(img_name)

        #initialize
        heatmap_reso = self.resol

        # get image
        img = Image.open(os.path.join(img_folder_path, image_name))
        # resize the original image
        w, h = img.size                                     # 256 * 256 * 3
        ratio_x = self.width / float(w)
        ratio_y = self.height / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.width, self.height))         # 368 * 368 * 3
        img = self.toTensor(img)
        img = self.normalize_img(img)

        # get label
        label = all_labels[image_name.split('.')[0][1:]]                  # list       21 * 2
        label = torch.Tensor(label)
        label[:,0] = heatmap_reso/w*label[:,0]
        label[:,1] = heatmap_reso/h*label[:,1]

        # get heatmap        
        # heatmap = self.generate_gaussian_groundtruth_cuda(label)
        heatmap = self.generate_gaussian_groundtruth(label)

        return {'image':img,
           'label': label,
           'heatmap': heatmap,
           'folder_name': folder_name,
           'image_name': image_name
           }




def Dataloader_CPM(image_dir, label_dir, batch_size, shuffle, num_workers=4, transform=None, sigma=1, resol=46):
    dataset = Dataset_CPM(image_dir,label_dir,transform, sigma, resol) 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers) 


class CMUPanopticHandmaskDataset(Dataset):
    """Handmask dataset for CMU Panoptic data
    This dataset is used for training U-Net
    """
    def __init__(self, lst, image_dir, mask_dir):
        """
        Args:
            lst (list) : a list of image_names
            image_dir (string) : the root dir with all images
            mask_dir (string) : the root dir with all masks
        """
        self.records = lst
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        print("the dataset contains {} instances!".format(len(lst)))


    def __len__(self):
        return len(self.records)

    def __getitem__(self,idx):
        image_name= self.records[idx]
        original_image = Image.open(os.path.join(self.image_dir,image_name))
        original_image_size = np.array(original_image.size)
        #transform image
        image = original_image.resize((256,256)) # resize to (256,256)
        image = transforms.ToTensor()(image)
        iamge = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(image)


        mask = np.load(os.path.join(self.mask_dir,image_name[:-4]+'.npy'))
        mask = cv2.resize(mask.astype(float), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)>0.5
        mask = np.array(mask, dtype=int)
        mask = torch.LongTensor(mask)

        sample = {'original_image_size': original_image_size, 'image':image, 'mask':mask, 'image_name':image_name}

        return sample



class CMUPanopticHandDataset(Dataset):
    """Hand dataset for CMU Panoptic data
    """

    def __init__(self, lst, dataset_root, img_resol=368, heatmap_resol=46):
        """
        Args:
            lst (list) : a list of image_names
            dataset_root (string) : the root dir which contains a folder named 'imgs' and a file named 'labels.json'
            img_resol: the resolution of the images returned by the getitem
            heatmap_resol: the resolution of the groundtruth heatmaps
        """
        self.records = lst
        self.image_dir = os.path.join(dataset_root, 'imgs')
        
        with open(os.path.join(dataset_root,'labels.json'),'r') as f:
            all_labels = json.load(f)

        self.all_labels = all_labels
        self.img_resol = img_resol
        self.heatmap_resol = heatmap_resol

        print("the dataset contains {} instances!".format(len(lst)))

    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """

        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps


    def __len__(self):
        return len(self.records)


    def __getitem__(self,idx):

        image_name = self.records[idx]

        # get image and label
        img = Image.open(os.path.join(self.image_dir, image_name))
        original_label = self.all_labels[image_name]

        # resize image and modify label correspondingly
        original_image_size = np.array(img.size)
        w, h = img.size                                     # 256 * 256 * 3 for example
        ratio_x = self.img_resol / float(w)
        ratio_y = self.img_resol / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.img_resol, self.img_resol))         # 368 * 368 * 3

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)

        label = torch.Tensor(original_label)
        label[:,0] = self.heatmap_resol/w*label[:,0]
        label[:,1] = self.heatmap_resol/h*label[:,1]        

        # get heatmap
        heatmap = self.generate_gaussian_groundtruth(label)
        # normalization
        heatmap = self.normalize_prob(heatmap)

        sample = {
           'original_image_size': original_image_size,
           'image':img,
           'label': label,
           'original_label': torch.Tensor(original_label),
           'heatmap': heatmap,
           'image_name': image_name
           }

        return sample

class CMUPanopticHandDataset_1(Dataset):
    """Hand dataset for CMU Panoptic data

       Same as CMUPanopticHandDataset, but this dataset doesn't normalize the returned joint heatmap.
    """

    def __init__(self, lst, dataset_root, img_resol=368, heatmap_resol=46):
        """
        Args:
            lst (list) : a list of image_names
            dataset_root (string) : the root dir which contains a folder named 'imgs' and a file named 'labels.json'
            img_resol: the resolution of the images returned by the getitem
            heatmap_resol: the resolution of the groundtruth heatmaps
        """
        self.records = lst
        self.image_dir = os.path.join(dataset_root, 'imgs')
        
        with open(os.path.join(dataset_root,'labels.json'),'r') as f:
            all_labels = json.load(f)

        self.all_labels = all_labels
        self.img_resol = img_resol
        self.heatmap_resol = heatmap_resol

        print("the dataset contains {} instances!".format(len(lst)))

    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """

        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps


    def __len__(self):
        return len(self.records)


    def __getitem__(self,idx):

        image_name = self.records[idx]

        # get image and label
        img = Image.open(os.path.join(self.image_dir, image_name))
        original_label = self.all_labels[image_name]

        # resize image and modify label correspondingly
        original_image_size = np.array(img.size)
        w, h = img.size                                     # 256 * 256 * 3 for example
        ratio_x = self.img_resol / float(w)
        ratio_y = self.img_resol / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.img_resol, self.img_resol))         # 368 * 368 * 3

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)

        label = torch.Tensor(original_label)
        label[:,0] = self.heatmap_resol/w*label[:,0]
        label[:,1] = self.heatmap_resol/h*label[:,1]        

        # get heatmap
        heatmap = self.generate_gaussian_groundtruth(label)

        sample = {
           'original_image_size': original_image_size,
           'image':img,
           'label': label,
           'original_label': torch.Tensor(original_label),
           'heatmap': heatmap,
           'image_name': image_name
           }

        return sample


class CMUPanopticHandDataset_2(Dataset):
    """Hand dataset for CMU Panoptic data,
        which will also provide the groundtruth for parameters of graphical model.
        would return addtionally a gaussian kernel of size 40x45x45 
        Used for training the adaptive graphical model NN. 
    """

    def __init__(self, lst, dataset_root, img_resol=368, heatmap_resol=46, gm_kernel_size=45):
        """
        Args:
            lst (list) : a list of image_names
            dataset_root (string) : the root dir which contains a folder named 'imgs' and a file named 'labels.json'
            img_resol: the resolution of the images returned by the getitem
            heatmap_resol: the resolution of the groundtruth heatmaps
            gm_kernel_size: the size of the graphical model kernel size
        """
        self.records = lst
        self.image_dir = os.path.join(dataset_root, 'imgs')
        
        with open(os.path.join(dataset_root,'labels.json'),'r') as f:
            all_labels = json.load(f)

        self.all_labels = all_labels
        self.img_resol = img_resol
        self.heatmap_resol = heatmap_resol
        self.gm_kernel_size = gm_kernel_size
        self.directed_edges = [                  # start of schedule from leaves to root
                        [4,3],     # edge 0, [from_joint=4, to_joint=3]
                        [3,2],     # edge 1
                        [2,1],     # edge 2
                        [1,0],     # edge 3

                        [8,7],     # edge 4
                        [7,6],     # edge 5
                        [6,5],     # edge 6
                        [5,0],     # edge 7

                        [12,11],   # edge 8
                        [11,10],   # edge 9
                        [10,9],    # edge 10
                        [9,0],     # edge 11

                        [16,15],   # edge 12
                        [15,14],   # edge 13
                        [14,13],   # edge 14
                        [13,0],    # edge 15

                        [20,19],  # edge 16
                        [19,18],  # edge 17
                        [18,17],  # edge 18
                        [17,0],   # edge 19   # end of schedule from leaves to root

                        [0,1],    # edge 20  # start of schedule from root to leaves
                        [1,2],    # edge 21
                        [2,3],    # edge 22
                        [3,4],    # edge 23

                        [0,5],    # edge 24
                        [5,6],    # edge 25
                        [6,7],    # edge 26
                        [7,8],    # edge 27

                        [0,9],    # edge 28
                        [9,10],   # edge 29
                        [10,11],  # edge 30
                        [11,12],  # edge 31

                        [0,13],   # edge 32
                        [13,14],  # edge 33
                        [14,15],  # edge 34
                        [15,16],  # edge 35

                        [0,17],   # edge 36
                        [17,18],  # edge 37
                        [18,19],  # edge 38
                        [19,20],  # edge 39         # end of schedule from root to leaves 
                    ]

        print("the dataset contains {} instances!".format(len(lst)))


    def calculate_rela_positions(self, label):
        """ calculate relative positions of each pair of joints forming an edge
        """
        rela_positions = torch.zeros(len(self.directed_edges), 2)
        for edge_id in range(len(self.directed_edges)):
            edge = self.directed_edges[edge_id]
            from_joint = edge[0]
            to_joint = edge[1]
            #print('**************')
            #print(label[from_joint])
            #print(label[to_joint])
            rela_positions[edge_id] = label[to_joint] - label[from_joint]
            #print(rela_positions[edge_id])
            #print(torch.sqrt(torch.sum(rela_positions[edge_id]**2)))
            #print('**************')
        return rela_positions

    def generate_graphical_model_kernels(self, rela_positions):

        center = float(self.gm_kernel_size//2)
        coords = torch.tensor([[center, center]]) - rela_positions # note the minus sign here
        gm_kernels = self.generate_gaussian_groundtruth(coords, sigma=1,resol=self.gm_kernel_size)
        return gm_kernels



    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """

        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps


    def __len__(self):
        return len(self.records)


    def __getitem__(self,idx):

        image_name = self.records[idx]

        # get image and label
        img = Image.open(os.path.join(self.image_dir, image_name))
        original_label = self.all_labels[image_name]

        # resize image and modify label correspondingly
        original_image_size = np.array(img.size)
        w, h = img.size                                     # 256 * 256 * 3 for example
        ratio_x = self.img_resol / float(w)
        ratio_y = self.img_resol / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.img_resol, self.img_resol))         # 368 * 368 * 3

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)

        label = torch.Tensor(original_label)
        label[:,0] = self.heatmap_resol/w*label[:,0]
        label[:,1] = self.heatmap_resol/h*label[:,1]        

        # rela positions
        rela_positions = self.calculate_rela_positions(label)
        gm_kernels = self.generate_graphical_model_kernels(rela_positions)

        # get heatmap
        heatmap = self.generate_gaussian_groundtruth(label)
        # normalization
        heatmap = self.normalize_prob(heatmap)

        sample = {
           'original_image_size': original_image_size,
           'image':img,
           'label': label,
           'original_label': torch.Tensor(original_label),
           'heatmap': heatmap,
           'gm_kernels': gm_kernels,
           'image_name': image_name
           }

        return sample



class CMUPanopticHandDataset_3(Dataset):
    """Hand dataset for CMU Panoptic data,
        which will also provide the groundtruth for parameters of graphical model.
        would return addtionally a gaussian kernel of size 40x45x45 (not normalized) 
        Used for training the adaptive graphical model NN.

        intermediate score map: not normalized
        final score map: normalized 
    """

    def __init__(self, lst, dataset_root, img_resol=368, heatmap_resol=46, gm_kernel_size=45):
        """
        Args:
            lst (list) : a list of image_names
            dataset_root (string) : the root dir which contains a folder named 'imgs' and a file named 'labels.json'
            img_resol: the resolution of the images returned by the getitem
            heatmap_resol: the resolution of the groundtruth heatmaps
            gm_kernel_size: the size of the graphical model kernel size
        """
        self.records = lst
        self.image_dir = os.path.join(dataset_root, 'imgs')
        
        with open(os.path.join(dataset_root,'labels.json'),'r') as f:
            all_labels = json.load(f)

        self.all_labels = all_labels
        self.img_resol = img_resol
        self.heatmap_resol = heatmap_resol
        self.gm_kernel_size = gm_kernel_size
        self.directed_edges = [                  # start of schedule from leaves to root
                        [4,3],     # edge 0, [from_joint=4, to_joint=3]
                        [3,2],     # edge 1
                        [2,1],     # edge 2
                        [1,0],     # edge 3

                        [8,7],     # edge 4
                        [7,6],     # edge 5
                        [6,5],     # edge 6
                        [5,0],     # edge 7

                        [12,11],   # edge 8
                        [11,10],   # edge 9
                        [10,9],    # edge 10
                        [9,0],     # edge 11

                        [16,15],   # edge 12
                        [15,14],   # edge 13
                        [14,13],   # edge 14
                        [13,0],    # edge 15

                        [20,19],  # edge 16
                        [19,18],  # edge 17
                        [18,17],  # edge 18
                        [17,0],   # edge 19   # end of schedule from leaves to root

                        [0,1],    # edge 20  # start of schedule from root to leaves
                        [1,2],    # edge 21
                        [2,3],    # edge 22
                        [3,4],    # edge 23

                        [0,5],    # edge 24
                        [5,6],    # edge 25
                        [6,7],    # edge 26
                        [7,8],    # edge 27

                        [0,9],    # edge 28
                        [9,10],   # edge 29
                        [10,11],  # edge 30
                        [11,12],  # edge 31

                        [0,13],   # edge 32
                        [13,14],  # edge 33
                        [14,15],  # edge 34
                        [15,16],  # edge 35

                        [0,17],   # edge 36
                        [17,18],  # edge 37
                        [18,19],  # edge 38
                        [19,20],  # edge 39         # end of schedule from root to leaves 
                    ]

        print("the dataset contains {} instances!".format(len(lst)))


    def calculate_rela_positions(self, label):
        """ calculate relative positions of each pair of joints forming an edge
        """
        rela_positions = torch.zeros(len(self.directed_edges), 2)
        for edge_id in range(len(self.directed_edges)):
            edge = self.directed_edges[edge_id]
            from_joint = edge[0]
            to_joint = edge[1]
            #print('**************')
            #print(label[from_joint])
            #print(label[to_joint])
            rela_positions[edge_id] = label[to_joint] - label[from_joint]
            #print(rela_positions[edge_id])
            #print(torch.sqrt(torch.sum(rela_positions[edge_id]**2)))
            #print('**************')
        return rela_positions

    def generate_graphical_model_kernels(self, rela_positions):

        center = float(self.gm_kernel_size//2)
        coords = torch.tensor([[center, center]]) - rela_positions # note the minus sign here
        gm_kernels = self.generate_gaussian_groundtruth(coords, sigma=1,resol=self.gm_kernel_size)
        return gm_kernels



    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """

        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps


    def __len__(self):
        return len(self.records)


    def __getitem__(self,idx):

        image_name = self.records[idx]

        # get image and label
        img = Image.open(os.path.join(self.image_dir, image_name))
        original_label = self.all_labels[image_name]

        # resize image and modify label correspondingly
        original_image_size = np.array(img.size)
        w, h = img.size                                     # 256 * 256 * 3 for example
        ratio_x = self.img_resol / float(w)
        ratio_y = self.img_resol / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.img_resol, self.img_resol))         # 368 * 368 * 3

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)

        label = torch.Tensor(original_label)
        label[:,0] = self.heatmap_resol/w*label[:,0]
        label[:,1] = self.heatmap_resol/h*label[:,1]        

        # rela positions
        rela_positions = self.calculate_rela_positions(label)
        gm_kernels = self.generate_graphical_model_kernels(rela_positions)

        # get heatmap
        heatmap = self.generate_gaussian_groundtruth(label)
        # normalization
        normalized_heatmap = self.normalize_prob(heatmap)

        sample = {
           'original_image_size': original_image_size,
           'image':img,
           'label': label,
           'original_label': torch.Tensor(original_label),
           'heatmap': heatmap,
           'normalized_heatmap': normalized_heatmap,
           'gm_kernels': gm_kernels,
           'image_name': image_name
           }

        return sample





class UCIHandDataset(Dataset):
    """Hand dataset for CMU Panoptic data,
        which will also provide the groundtruth for parameters of graphical model.
        would return addtionally a gaussian kernel of size 40x45x45 (not normalized) 
        Used for training the adaptive graphical model NN.

        intermediate score map: not normalized
        final score map: normalized 
    """

    def __init__(self, lst, dataset_root, img_resol=368, heatmap_resol=46, gm_kernel_size=45):
        """
        Args:
            lst (list) : a list of image_names
            dataset_root (string) : the root dir which contains a folder named 'imgs' and a file named 'labels.json'
            img_resol: the resolution of the images returned by the getitem
            heatmap_resol: the resolution of the groundtruth heatmaps
            gm_kernel_size: the size of the graphical model kernel size
        """
        self.records = lst
        self.image_dir = os.path.join(dataset_root, 'imgs')
        
        with open(os.path.join(dataset_root,'labels.json'),'r') as f:
            all_labels = json.load(f)

        self.all_labels = all_labels
        self.img_resol = img_resol
        self.heatmap_resol = heatmap_resol
        self.gm_kernel_size = gm_kernel_size
        self.directed_edges = [                  # start of schedule from leaves to root
                        [4,3],     # edge 0, [from_joint=4, to_joint=3]
                        [3,2],     # edge 1
                        [2,1],     # edge 2
                        [1,0],     # edge 3

                        [8,7],     # edge 4
                        [7,6],     # edge 5
                        [6,5],     # edge 6
                        [5,0],     # edge 7

                        [12,11],   # edge 8
                        [11,10],   # edge 9
                        [10,9],    # edge 10
                        [9,0],     # edge 11

                        [16,15],   # edge 12
                        [15,14],   # edge 13
                        [14,13],   # edge 14
                        [13,0],    # edge 15

                        [20,19],  # edge 16
                        [19,18],  # edge 17
                        [18,17],  # edge 18
                        [17,0],   # edge 19   # end of schedule from leaves to root

                        [0,1],    # edge 20  # start of schedule from root to leaves
                        [1,2],    # edge 21
                        [2,3],    # edge 22
                        [3,4],    # edge 23

                        [0,5],    # edge 24
                        [5,6],    # edge 25
                        [6,7],    # edge 26
                        [7,8],    # edge 27

                        [0,9],    # edge 28
                        [9,10],   # edge 29
                        [10,11],  # edge 30
                        [11,12],  # edge 31

                        [0,13],   # edge 32
                        [13,14],  # edge 33
                        [14,15],  # edge 34
                        [15,16],  # edge 35

                        [0,17],   # edge 36
                        [17,18],  # edge 37
                        [18,19],  # edge 38
                        [19,20],  # edge 39         # end of schedule from root to leaves 
                    ]

        print("the dataset contains {} instances!".format(len(lst)))


    def calculate_rela_positions(self, label):
        """ calculate relative positions of each pair of joints forming an edge
        """
        rela_positions = torch.zeros(len(self.directed_edges), 2)
        for edge_id in range(len(self.directed_edges)):
            edge = self.directed_edges[edge_id]
            from_joint = edge[0]
            to_joint = edge[1]
            #print('**************')
            #print(label[from_joint])
            #print(label[to_joint])
            rela_positions[edge_id] = label[to_joint] - label[from_joint]
            #print(rela_positions[edge_id])
            #print(torch.sqrt(torch.sum(rela_positions[edge_id]**2)))
            #print('**************')
        return rela_positions

    def generate_graphical_model_kernels(self, rela_positions):

        center = float(self.gm_kernel_size//2)
        coords = torch.tensor([[center, center]]) - rela_positions # note the minus sign here
        gm_kernels = self.generate_gaussian_groundtruth(coords, sigma=1,resol=self.gm_kernel_size)
        return gm_kernels



    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """
        x = x + 1e-16
        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps


    def __len__(self):
        return len(self.records)


    def __getitem__(self,idx):

        image_name = self.records[idx]

        # get image and label
        img = Image.open(os.path.join(self.image_dir, image_name))
        original_label = self.all_labels[image_name]

        # resize image and modify label correspondingly
        original_image_size = np.array(img.size)
        w, h = img.size                                     # 256 * 256 * 3 for example
        ratio_x = self.img_resol / float(w)
        ratio_y = self.img_resol / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.img_resol, self.img_resol))         # 368 * 368 * 3

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)

        label = torch.Tensor(original_label)
        label[:,0] = self.heatmap_resol/w*label[:,0]
        label[:,1] = self.heatmap_resol/h*label[:,1]        

        # rela positions
        rela_positions = self.calculate_rela_positions(label)
        gm_kernels = self.generate_graphical_model_kernels(rela_positions)

        # get heatmap
        heatmap = self.generate_gaussian_groundtruth(label)
        # normalization
        normalized_heatmap = self.normalize_prob(heatmap)

        sample = {
           'original_image_size': original_image_size,
           'image':img,
           'label': label,
           'original_label': torch.Tensor(original_label),
           'heatmap': heatmap,
           'normalized_heatmap': normalized_heatmap,
           'gm_kernels': gm_kernels,
           'image_name': image_name
           }

        return sample


class ROVITHandDataset(Dataset):
    """Hand dataset for CMU Panoptic data,
        which will also provide the groundtruth for parameters of graphical model.
        would return addtionally a gaussian kernel of size 40x45x45 (not normalized) 
        Used for training the adaptive graphical model NN.

        intermediate score map: not normalized
        final score map: normalized 
    """

    def __init__(self, lst, dataset_root, img_resol=368, heatmap_resol=46, gm_kernel_size=45):
        """
        Args:
            lst (list) : a list of image_names
            dataset_root (string) : the root dir which contains a folder named 'imgs' and a file named 'labels.json'
            img_resol: the resolution of the images returned by the getitem
            heatmap_resol: the resolution of the groundtruth heatmaps
            gm_kernel_size: the size of the graphical model kernel size
        """
        self.records = lst
        self.image_dir = os.path.join(dataset_root, 'imgs')
        
        with open(os.path.join(dataset_root,'labels.json'),'r') as f:
            all_labels = json.load(f)

        self.all_labels = all_labels
        self.img_resol = img_resol
        self.heatmap_resol = heatmap_resol
        self.gm_kernel_size = gm_kernel_size
        self.directed_edges = [                  # start of schedule from leaves to root
                        [4,3],     # edge 0, [from_joint=4, to_joint=3]
                        [3,2],     # edge 1
                        [2,1],     # edge 2
                        [1,0],     # edge 3

                        [8,7],     # edge 4
                        [7,6],     # edge 5
                        [6,5],     # edge 6
                        [5,0],     # edge 7

                        [12,11],   # edge 8
                        [11,10],   # edge 9
                        [10,9],    # edge 10
                        [9,0],     # edge 11

                        [16,15],   # edge 12
                        [15,14],   # edge 13
                        [14,13],   # edge 14
                        [13,0],    # edge 15

                        [20,19],  # edge 16
                        [19,18],  # edge 17
                        [18,17],  # edge 18
                        [17,0],   # edge 19   # end of schedule from leaves to root

                        [0,1],    # edge 20  # start of schedule from root to leaves
                        [1,2],    # edge 21
                        [2,3],    # edge 22
                        [3,4],    # edge 23

                        [0,5],    # edge 24
                        [5,6],    # edge 25
                        [6,7],    # edge 26
                        [7,8],    # edge 27

                        [0,9],    # edge 28
                        [9,10],   # edge 29
                        [10,11],  # edge 30
                        [11,12],  # edge 31

                        [0,13],   # edge 32
                        [13,14],  # edge 33
                        [14,15],  # edge 34
                        [15,16],  # edge 35

                        [0,17],   # edge 36
                        [17,18],  # edge 37
                        [18,19],  # edge 38
                        [19,20],  # edge 39         # end of schedule from root to leaves 
                    ]

        print("the dataset contains {} instances!".format(len(lst)))


    def calculate_rela_positions(self, label):
        """ calculate relative positions of each pair of joints forming an edge
        """
        rela_positions = torch.zeros(len(self.directed_edges), 2)
        for edge_id in range(len(self.directed_edges)):
            edge = self.directed_edges[edge_id]
            from_joint = edge[0]
            to_joint = edge[1]
            #print('**************')
            #print(label[from_joint])
            #print(label[to_joint])
            rela_positions[edge_id] = label[to_joint] - label[from_joint]
            #print(rela_positions[edge_id])
            #print(torch.sqrt(torch.sum(rela_positions[edge_id]**2)))
            #print('**************')
        return rela_positions

    def generate_graphical_model_kernels(self, rela_positions):

        center = float(self.gm_kernel_size//2)
        coords = torch.tensor([[center, center]]) - rela_positions # note the minus sign here
        gm_kernels = self.generate_gaussian_groundtruth(coords, sigma=1,resol=self.gm_kernel_size)
        return gm_kernels



    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """
        x = x + 1e-16
        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps


    def __len__(self):
        return len(self.records)


    def __getitem__(self,idx):

        image_name = self.records[idx]

        # get image and label
        img = Image.open(os.path.join(self.image_dir, image_name))
        original_label = self.all_labels[image_name]

        # resize image and modify label correspondingly
        original_image_size = np.array(img.size)
        w, h = img.size                                     # 256 * 256 * 3 for example
        ratio_x = self.img_resol / float(w)
        ratio_y = self.img_resol / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.img_resol, self.img_resol))         # 368 * 368 * 3

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)

        label = torch.Tensor(original_label)
        label[:,0] = self.heatmap_resol/w*label[:,0]
        label[:,1] = self.heatmap_resol/h*label[:,1]        

        # rela positions
        rela_positions = self.calculate_rela_positions(label)
        gm_kernels = self.generate_graphical_model_kernels(rela_positions)

        # get heatmap
        heatmap = self.generate_gaussian_groundtruth(label)
        # normalization
        normalized_heatmap = self.normalize_prob(heatmap)

        sample = {
           'original_image_size': original_image_size,
           'image':img,
           'label': label,
           'original_label': torch.Tensor(original_label),
           'heatmap': heatmap,
           'normalized_heatmap': normalized_heatmap,
           'gm_kernels': gm_kernels,
           'image_name': image_name
           }

        return sample




class MPIIHandDataset(Dataset):
    """Hand dataset for CMU Panoptic data,
        which will also provide the groundtruth for parameters of graphical model.
        would return addtionally a gaussian kernel of size 40x45x45 (not normalized) 
        Used for training the adaptive graphical model NN.

        intermediate score map: not normalized
        final score map: normalized 
    """

    def __init__(self, lst, dataset_root, img_resol=368, heatmap_resol=46, gm_kernel_size=45):
        """
        Args:
            lst (list) : a list of image_names
            dataset_root (string) : the root dir which contains a folder named 'imgs' and a file named 'labels.json'
            img_resol: the resolution of the images returned by the getitem
            heatmap_resol: the resolution of the groundtruth heatmaps
            gm_kernel_size: the size of the graphical model kernel size
        """
        self.records = lst
        self.image_dir = os.path.join(dataset_root, 'imgs')
        
        with open(os.path.join(dataset_root,'labels.json'),'r') as f:
            all_labels = json.load(f)

        self.all_labels = all_labels
        self.img_resol = img_resol
        self.heatmap_resol = heatmap_resol
        self.gm_kernel_size = gm_kernel_size
        self.directed_edges = [                  # start of schedule from leaves to root
                        [4,3],     # edge 0, [from_joint=4, to_joint=3]
                        [3,2],     # edge 1
                        [2,1],     # edge 2
                        [1,0],     # edge 3

                        [8,7],     # edge 4
                        [7,6],     # edge 5
                        [6,5],     # edge 6
                        [5,0],     # edge 7

                        [12,11],   # edge 8
                        [11,10],   # edge 9
                        [10,9],    # edge 10
                        [9,0],     # edge 11

                        [16,15],   # edge 12
                        [15,14],   # edge 13
                        [14,13],   # edge 14
                        [13,0],    # edge 15

                        [20,19],  # edge 16
                        [19,18],  # edge 17
                        [18,17],  # edge 18
                        [17,0],   # edge 19   # end of schedule from leaves to root

                        [0,1],    # edge 20  # start of schedule from root to leaves
                        [1,2],    # edge 21
                        [2,3],    # edge 22
                        [3,4],    # edge 23

                        [0,5],    # edge 24
                        [5,6],    # edge 25
                        [6,7],    # edge 26
                        [7,8],    # edge 27

                        [0,9],    # edge 28
                        [9,10],   # edge 29
                        [10,11],  # edge 30
                        [11,12],  # edge 31

                        [0,13],   # edge 32
                        [13,14],  # edge 33
                        [14,15],  # edge 34
                        [15,16],  # edge 35

                        [0,17],   # edge 36
                        [17,18],  # edge 37
                        [18,19],  # edge 38
                        [19,20],  # edge 39         # end of schedule from root to leaves 
                    ]

        print("the dataset contains {} instances!".format(len(lst)))


    def calculate_rela_positions(self, label):
        """ calculate relative positions of each pair of joints forming an edge
        """
        rela_positions = torch.zeros(len(self.directed_edges), 2)
        for edge_id in range(len(self.directed_edges)):
            edge = self.directed_edges[edge_id]
            from_joint = edge[0]
            to_joint = edge[1]
            #print('**************')
            #print(label[from_joint])
            #print(label[to_joint])
            rela_positions[edge_id] = label[to_joint] - label[from_joint]
            #print(rela_positions[edge_id])
            #print(torch.sqrt(torch.sum(rela_positions[edge_id]**2)))
            #print('**************')
        return rela_positions

    def generate_graphical_model_kernels(self, rela_positions):

        center = float(self.gm_kernel_size//2)
        coords = torch.tensor([[center, center]]) - rela_positions # note the minus sign here
        gm_kernels = self.generate_gaussian_groundtruth(coords, sigma=1,resol=self.gm_kernel_size)
        return gm_kernels



    def normalize_prob(self, x):
        """
        Normalize along last two dimensions
        """
        x = x + 1e-16
        return x/torch.sum(torch.sum(x,dim=-1,keepdim=True), dim=-2, keepdim=True)


    def generate_gaussian_groundtruth(self, labels, sigma=1, resol=46):
        """
        cpu version of generate_gaussian_groundtruth. 4x speed up.
        Implemented on 10/26/2018. Works fine.

        Args: 
            labels, torch.Tensor, 21*2, first column is x(horizontal),
                    second column is y(vertical, from top to bottom)

        Returns:
            heatmaps, torch.Tensor, 21*46*46   
        """
        #labels = torch.from_numpy(labels).float().cuda()
        labels = labels
        # construct 3-D tensor a, which stores (column(x), row(y)) for each pixel 
        a = torch.zeros((resol,resol,2))
        a[...,0] = a[...,0] + torch.Tensor(range(resol)).unsqueeze(0)
        a[...,1] = a[...,1] + torch.Tensor(range(resol)).unsqueeze(1)
        a = a.unsqueeze(0)
        labels = labels.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((a - labels)**2, dim=-1)
        heatmaps = torch.exp(-exponent / 2.0 / sigma / sigma)
        return heatmaps


    def __len__(self):
        return len(self.records)


    def __getitem__(self,idx):

        image_name = self.records[idx]

        # get image and label
        img = Image.open(os.path.join(self.image_dir, image_name))
        original_label = self.all_labels[image_name]

        # resize image and modify label correspondingly
        original_image_size = np.array(img.size)
        w, h = img.size                                     # 256 * 256 * 3 for example
        ratio_x = self.img_resol / float(w)
        ratio_y = self.img_resol / float(h)                    # 368 / 256 = 1.4375
        img = img.resize((self.img_resol, self.img_resol))         # 368 * 368 * 3

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)

        label = torch.Tensor(original_label)
        label[:,0] = self.heatmap_resol/w*label[:,0]
        label[:,1] = self.heatmap_resol/h*label[:,1]        

        # rela positions
        rela_positions = self.calculate_rela_positions(label)
        gm_kernels = self.generate_graphical_model_kernels(rela_positions)

        # get heatmap
        heatmap = self.generate_gaussian_groundtruth(label)
        # normalization
        normalized_heatmap = self.normalize_prob(heatmap)

        sample = {
           'original_image_size': original_image_size,
           'image':img,
           'label': label,
           'original_label': torch.Tensor(original_label),
           'heatmap': heatmap,
           'normalized_heatmap': normalized_heatmap,
           'gm_kernels': gm_kernels,
           'image_name': image_name
           }

        return sample






def main(dataset_name):
    '''
    Testing dataset module.
    '''
    if dataset_name == 'Dataset_Hand_1':
        partition_path = '../../data/internal/intermediate_1/partitions/split_both_hands.json'
        with open(partition_path, 'r') as f:
            partition = json.load(f)
        train_records = partition['train']
        print(len(train_records))



        # dataset = Dataset_Hand_1(list_records=train_records,
        #                             image_dir = '../../data/internal/intermediate_1/images',
        #                             label_dir='../../data/internal/intermediate_1/labels')
        dataset = Dataset_Hand_1(list_records=train_records,
                                    data_path = '../../data/internal/intermediate_1')
        data = dataset[0]
        image = data['image']
        label = data['label']
        print('*********************',image)
        print('*********************',label)

    elif dataset_name == 'CMUPanopticHandmaskDataset':

        partition_file = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/partitions/partition.json'
        image_dir = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/imgs'
        gdtt_mask_dir = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/gdtt_masks'
        with open(partition_file, 'r') as f:
            partition_dict = json.load(f)
        train_lst = partition_dict['train']
        dataset = CMUPanopticHandmaskDataset(train_lst, image_dir, gdtt_mask_dir)
        print(dataset[2])

    elif dataset_name =='CMUPanopticHandDataset':        
        partition_file = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/partitions/partition.json'
        dataset_root = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/'
        with open(partition_file, 'r') as f:
            partition_dict = json.load(f)
        train_lst = partition_dict['train']
        dataset = CMUPanopticHandDataset(train_lst, dataset_root)
        print(dataset[2]['heatmap'].shape)

    elif dataset_name =='CMUPanopticHandDataset_2':
        print('Testing ', dataset_name)        
        partition_file = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/partitions/partition.json'
        dataset_root = '/home/deyingk/handpose/data/external/cmu_panoptic_hands/intermediate_1/'
        with open(partition_file, 'r') as f:
            partition_dict = json.load(f)
        train_lst = partition_dict['train']
        dataset = CMUPanopticHandDataset_2(train_lst, dataset_root)
        print(dataset[2]['gm_kernels'].shape)
if __name__ == '__main__':
    main('CMUPanopticHandDataset_2')