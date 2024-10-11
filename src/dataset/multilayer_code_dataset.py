import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir, num_files, flip=False, uncond=False):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = flip
        self.uncond = uncond

        print("whether unconditional: ", self.uncond)
        print("whether flip: ", self.flip)

        self.aug_feature_dir = None
        self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable

        self.feature_files = [f"{i}.npy" for i in range(num_files)]
        self.label_files = [f"{i}.npy" for i in range(num_files)]

        # self.feature_files = [f"{i}.npy" for i in range(1000000)]
        # self.label_files = [f"{i}.npy" for i in range(1000000)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):

        feature_dir = self.feature_dir
        label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[0], size=(1,)).item()
            features = features[aug_idx,:]
        else:
            features = features[0,:]
        
        if self.uncond:
            # int zero
            labels = np.zeros(1).astype(np.int64)
        else:
            labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features).unsqueeze(0), torch.from_numpy(labels)

class ArxivImagenetDataset(Dataset):
    def __init__(self):
        self.feature_dir = "/cpfs01/user/cl424408/rq-vae-transformer-main/extracted_codes/ImageNet_Train/rq2048_16x16x8_imagenet_arxiv_withauxloss/in256-rqvae-16x16x8-arxiv-withaux/29072024_055601/epoch30_model/codes"
        self.label_dir = "/cpfs01/user/cl424408/rq-vae-transformer-main/extracted_codes/ImageNet_Train/rq2048_16x16x8_imagenet_arxiv_withauxloss/in256-rqvae-16x16x8-arxiv-withaux/29072024_055601/epoch30_model/labels"

        self.feature_dir_2 = "/cpfs01/user/cl424408/rq-vae-transformer-main/extracted_codes/ArxivFull_Train/rq2048_16x16x8_imagenet_arxiv_withauxloss/in256-rqvae-16x16x8-arxiv-withaux/29072024_055601/epoch30_model/codes"
        self.label_dir_2 = "/cpfs01/user/cl424408/rq-vae-transformer-main/extracted_codes/ArxivFull_Train/rq2048_16x16x8_imagenet_arxiv_withauxloss/in256-rqvae-16x16x8-arxiv-withaux/29072024_055601/epoch30_model/labels"

        self.num_files_1 = 1281167
        self.num_files_2 = 730209

        self.flip = True
        self.uncond = False

        print("whether unconditional: ", self.uncond)

        aug_feature_dir = self.feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = self.label_dir.replace('ten_crop/', 'ten_crop_105/')
        if "crop" in aug_feature_dir and os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None


        self.feature_files = [f"{i}.npy" for i in range(self.num_files_1)]
        self.label_files = [f"{i}.npy" for i in range(self.num_files_1)]

        self.feature_files_2 = [f"{i}.npy" for i in range(self.num_files_2)]
        self.label_files_2 = [f"{i}.npy" for i in range(self.num_files_2)]



    def __len__(self):
        return self.num_files_1 + self.num_files_2

    def __getitem__(self, idx):
        if idx < self.num_files_1:
            feature_dir = self.feature_dir
            label_dir = self.label_dir


            feature_file = self.feature_files[idx]
            label_file = self.label_files[idx]

        else:
            feature_dir = self.feature_dir_2
            label_dir = self.label_dir_2


            feature_file = self.feature_files_2[idx - self.num_files_1]
            label_file = self.label_files_2[idx - self.num_files_1]
        
        
                   

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[0], size=(1,)).item()
            features = features[aug_idx,:]

        
        if idx >= self.num_files_1:
            # int zero
            labels = np.array([1001]).astype(np.int64)
        else:
            labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features).unsqueeze(0), torch.from_numpy(labels)
        
def build_arxiv_imagenet_dataset(args):
    return ArxivImagenetDataset()

def build_multilayer_code_dataset(args):
    # arxiv_03m numfiles = 297122 07m: 730209
    return CustomDataset(args.feature_dir, args.label_dir, args.num_files, args.is_flip, args.uncond)