# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import Subset
import torchvision
from torchvision.datasets import ImageNet,ImageFolder
from torch.utils.data import ConcatDataset

import cv2
from PIL import Image
import numpy as np


from .lsun import LSUNClass
from .ffhq import FFHQ
from .transforms import create_transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split='val', is_eval=is_eval)

    root = config.dataset.get('root', None)

    if config.dataset.type == 'imagenet':
        print("imagenet")
        root = root if root else 'data/imagenet'

        # dataset = ImageFolder(args.data_path, transform=transform)

        #raw_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        raw_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        raw_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"

        # dataset_trn = ImageNet("/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val", split='val', transform=transforms_trn)
        # dataset_val = ImageNet("/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val", split='val', transform=transforms_val)
        dataset_trn = ImageFolder(raw_train_folder, transform=transforms_trn)
        dataset_val = ImageFolder(raw_val_folder, transform=transforms_val)
    
    elif config.dataset.type == 'PIL-512-32-2_4M':
        print("PIL-512-32-2_4M")
        arxiv_val_folder = "/cpfs01/user/cl424408/datasetwikipedia-main/PIL512-32-VAL"
        arxiv_train_folder = "/cpfs01/user/cl424408/datasetwikipedia-main/PIL512-32-2_4M"

        dataset_trn = ImageFolder(arxiv_train_folder, transform=transforms_trn)
        dataset_val = ImageFolder(arxiv_val_folder, transform=transforms_val)

        print("dataset_trn:", len(dataset_trn))
        print("dataset_val:", len(dataset_val))

    elif config.dataset.type == 'Imagenet-PIL2_4M':


        arxiv_train_folder = "/cpfs01/user/cl424408/datasetwikipedia-main/PIL512-32-2_4M"
        dataset_trn_arxiv = ImageFolder(arxiv_train_folder, transform=transforms_trn)

        im_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        im_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"

        # dataset_trn = ImageNet("/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val", split='val', transform=transforms_trn)
        # dataset_val = ImageNet("/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val", split='val', transform=transforms_val)
        dataset_trn_im = ImageFolder(im_train_folder, transform=transforms_trn)
        dataset_val_im = ImageFolder(im_val_folder, transform=transforms_val)

        dataset_trn = ConcatDataset([dataset_trn_arxiv,dataset_trn_im])
        dataset_val = ConcatDataset([dataset_val_im])

        print("total dataset_trn:", len(dataset_trn))
        print("total dataset_val:", len(dataset_val))



    
    
    elif config.dataset.type == 'arxiv':
        print("arxiv")
        arxiv_val_folder = "/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/val"
        arxiv_train_folder = "/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/train"

        dataset_trn = ImageFolder(arxiv_train_folder, transform=transforms_trn)
        dataset_val = ImageFolder(arxiv_val_folder, transform=transforms_val)

        print("dataset_trn:", len(dataset_trn))
        print("dataset_val:", len(dataset_val))
    


    elif config.dataset.type == 'arxiv-full':
        print("arxiv-full")
        arxiv_val_folder = "/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/val"
        arxiv_train_folder = "/cpfs01/user/cl424408/datasets/Z-PDF-Full-Images/train"

        dataset_trn = ImageFolder(arxiv_train_folder, transform=transforms_trn)
        dataset_val = ImageFolder(arxiv_val_folder, transform=transforms_val)

        print("dataset_trn:", len(dataset_trn))
        print("dataset_val:", len(dataset_val))

    elif config.dataset.type == 'arxiv-full-512':
        print("arxiv-full")
        arxiv_val_folder = "/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/val"
        arxiv_train_folder = "/cpfs01/user/cl424408/datasets/Z-PDF-Full-Images/train"

        dataset_trn = ImageFolder(arxiv_train_folder, transform=transforms_trn)
        dataset_val = ImageFolder(arxiv_val_folder, transform=transforms_val)

        print("dataset_trn:", len(dataset_trn))
        print("dataset_val:", len(dataset_val))

    elif config.dataset.type == 'arxiv-top-1024':
        print("arxiv-top-1024")
        arxiv_train_folder = "/cpfs01/user/cl424408/rq-vae-transformer-main/generate_pdf_images/pdf_1024_20w_top_train_images"
        arxiv_val_folder = "/cpfs01/user/cl424408/rq-vae-transformer-main/generate_pdf_images/pdf_1024_20w_top_validation_images"

        dataset_trn = ImageFolder(arxiv_train_folder, transform=transforms_trn)
        dataset_val = ImageFolder(arxiv_val_folder, transform=transforms_val)

        print("dataset_trn:", len(dataset_trn))
        print("dataset_val:", len(dataset_val))
    
    elif config.dataset.type == 'imagenet-arxiv':
        print("imagenet-arxiv")
        imagenet_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        arxiv_val_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/val"
        if is_eval:
            print("using val for both train and val")
            imagenet_train_folder = imagenet_val_folder
            arxiv_train_folder = arxiv_val_folder
        else:
            imagenet_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
            arxiv_train_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/train"

        dataset_trn_im = ImageFolder(imagenet_train_folder, transform=transforms_trn)
        print("dataset_trn_im:", len(dataset_trn_im))

        dataset_val_im = ImageFolder(imagenet_val_folder, transform=transforms_val)
        print("dataset_val_im:", len(dataset_val_im))

        dataset_trn_arxiv = ImageFolder(arxiv_train_folder, transform=transforms_trn)
        print("dataset_trn_arxiv:", len(dataset_trn_arxiv))

        dataset_val_arxiv = ImageFolder(arxiv_val_folder, transform=transforms_val)
        print("dataset_val_arxiv:", len(dataset_val_arxiv))

        dataset_trn = ConcatDataset([dataset_trn_im,dataset_trn_arxiv])
        dataset_val = ConcatDataset([dataset_val_im]) # only imagenet for val

        print("total dataset_trn:", len(dataset_trn))
        print("total dataset_val:", len(dataset_val))
    
    elif config.dataset.type == 'imagenet-syndogen-arxiv':
        print("imagenet-syndogen-arxiv")
        imagenet_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        syndog_en_val_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/valid"
        arxiv_val_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/val"
        if is_eval:
            print("using val for both train and val")
            imagenet_train_folder = imagenet_val_folder
            syndog_en_train_folder = syndog_en_val_folder
            arxiv_train_folder = arxiv_val_folder
        else:
            imagenet_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
            syndog_en_train_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/train"
            arxiv_train_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/train"

        dataset_trn_im = ImageFolder(imagenet_train_folder, transform=transforms_trn)
        print("dataset_trn_im:", len(dataset_trn_im))

        dataset_val_im = ImageFolder(imagenet_val_folder, transform=transforms_val)
        print("dataset_val_im:", len(dataset_val_im))

        dataset_trn_syn = ImageFolder(syndog_en_train_folder, transform=transforms_trn)
        print("dataset_trn_syn:", len(dataset_trn_syn))

        dataset_val_syn = ImageFolder(syndog_en_val_folder, transform=transforms_val)
        print("dataset_val_syn:", len(dataset_val_syn))

        dataset_trn_arxiv = ImageFolder(arxiv_train_folder, transform=transforms_trn)
        print("dataset_trn_arxiv:", len(dataset_trn_arxiv))

        dataset_val_arxiv = ImageFolder(arxiv_val_folder, transform=transforms_val)
        print("dataset_val_arxiv:", len(dataset_val_arxiv))

        dataset_trn = ConcatDataset([dataset_trn_im,dataset_trn_syn,dataset_trn_arxiv])
        dataset_val = ConcatDataset([dataset_val_im,dataset_val_syn,dataset_val_arxiv])

        print("total dataset_trn:", len(dataset_trn))
        print("total dataset_val:", len(dataset_val))
    
    elif config.dataset.type == 'imagenet-syndogen-arxiv-f32-VL':
        print("imagenet-syndogen-arxiv-f32-VL")
        imagenet_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        syndog_en_val_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/valid"
        arxiv_val_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/val"
        if is_eval:
            print("using val for both train and val")
            imagenet_train_folder = imagenet_val_folder
            syndog_en_train_folder = syndog_en_val_folder
            arxiv_train_folder = arxiv_val_folder
        else:
            imagenet_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
            syndog_en_train_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/train"
            arxiv_train_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/train"

        dataset_trn_im = DctImageFolder(imagenet_train_folder, transform=transforms_trn)
        print("dataset_trn_im:", len(dataset_trn_im))

        dataset_val_im = DctImageFolder(imagenet_val_folder, transform=transforms_val)
        print("dataset_val_im:", len(dataset_val_im))

        dataset_trn_syn = DctImageFolder(syndog_en_train_folder, transform=transforms_trn)
        print("dataset_trn_syn:", len(dataset_trn_syn))

        dataset_val_syn = DctImageFolder(syndog_en_val_folder, transform=transforms_val)
        print("dataset_val_syn:", len(dataset_val_syn))

        dataset_trn_arxiv = DctImageFolder(arxiv_train_folder, transform=transforms_trn)
        print("dataset_trn_arxiv:", len(dataset_trn_arxiv))

        dataset_val_arxiv = DctImageFolder(arxiv_val_folder, transform=transforms_val)
        print("dataset_val_arxiv:", len(dataset_val_arxiv))

        dataset_trn = ConcatDataset([dataset_trn_im,dataset_trn_syn,dataset_trn_arxiv])
        dataset_val = ConcatDataset([dataset_val_im,dataset_val_syn,dataset_val_arxiv])

        print("total dataset_trn:", len(dataset_trn))
        print("total dataset_val:", len(dataset_val))
        
    
    elif config.dataset.type == 'imagenet-syndogen-arxiv-f32-VL-Global':
        print("imagenet-syndogen-arxiv-f32-VL-Global-Effcient1.6")
        imagenet_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        syndog_en_val_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/valid"
        arxiv_val_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/val"
        if is_eval:
            print("using val for both train and val")
            imagenet_train_folder = imagenet_val_folder
            syndog_en_train_folder = syndog_en_val_folder
            arxiv_train_folder = arxiv_val_folder
        else:
            imagenet_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
            syndog_en_train_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/train"
            arxiv_train_folder="/cpfs01/user/cl424408/datasets/ZFC-PDF-Images/train"

        dataset_trn_im = GlobalDctImageFolder(imagenet_train_folder, transform=transforms_trn)
        print("dataset_trn_im:", len(dataset_trn_im))

        dataset_val_im = GlobalDctImageFolder(imagenet_val_folder, transform=transforms_val)
        print("dataset_val_im:", len(dataset_val_im))

        dataset_trn_syn = GlobalDctImageFolder(syndog_en_train_folder, transform=transforms_trn)
        print("dataset_trn_syn:", len(dataset_trn_syn))

        dataset_val_syn = GlobalDctImageFolder(syndog_en_val_folder, transform=transforms_val)
        print("dataset_val_syn:", len(dataset_val_syn))

        dataset_trn_arxiv = GlobalDctImageFolder(arxiv_train_folder, transform=transforms_trn)
        print("dataset_trn_arxiv:", len(dataset_trn_arxiv))

        dataset_val_arxiv = GlobalDctImageFolder(arxiv_val_folder, transform=transforms_val)
        print("dataset_val_arxiv:", len(dataset_val_arxiv))

        dataset_trn = ConcatDataset([dataset_trn_im,dataset_trn_syn,dataset_trn_arxiv])
        dataset_val = ConcatDataset([dataset_val_im,dataset_val_syn,dataset_val_arxiv])

        print("total dataset_trn:", len(dataset_trn))
        print("total dataset_val:", len(dataset_val))

    elif config.dataset.type == 'ffhq1024':
        print("ffhq1024")
        ffhq_val_folder = "/cpfs01/user/cl424408/datasets/FFHQ-IMGs/val"
        ffhq_train_folder = "/cpfs01/user/cl424408/datasets/FFHQ-IMGs/train"

        dataset_trn = ImageFolder(ffhq_train_folder, transform=transforms_trn)
        dataset_val = ImageFolder(ffhq_val_folder, transform=transforms_val)

    elif config.dataset.type == 'ffhq1024-f32-VL':
        print("ffhq1024-f32-VL")
        ffhq_val_folder = "/cpfs01/user/cl424408/datasets/FFHQ-IMGs/val"
        ffhq_train_folder = "/cpfs01/user/cl424408/datasets/FFHQ-IMGs/train"

        dataset_trn = DctImageFolder(ffhq_train_folder, transform=transforms_trn)
        dataset_val = DctImageFolder(ffhq_val_folder, transform=transforms_val)
    
    elif config.dataset.type == 'imagenet-syndogen':
        print("imagenet-syndogen")
        raw_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        syndog_en_val_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/valid"
        if is_eval:
            print("using val for both train and val")
            raw_train_folder = raw_val_folder
            syndog_en_train_folder = syndog_en_val_folder
        else:
            raw_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
            syndog_en_train_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/train"
        


        dataset_trn_im = ImageFolder(raw_train_folder, transform=transforms_trn)
        print("dataset_trn_im:", len(dataset_trn_im))

        dataset_val_im = ImageFolder(raw_val_folder, transform=transforms_val)
        print("dataset_val_im:", len(dataset_val_im))

        dataset_trn_syn = ImageFolder(syndog_en_train_folder, transform=transforms_trn)
        print("dataset_trn_syn:", len(dataset_trn_syn))
        
        dataset_val_syn = ImageFolder(syndog_en_val_folder, transform=transforms_val)
        print("dataset_val_syn:", len(dataset_val_syn))

        dataset_trn = ConcatDataset([dataset_trn_im,dataset_trn_syn])
        dataset_val = ConcatDataset([dataset_val_im,dataset_val_syn])
    
    elif config.dataset.type == 'imagenet256-f32-VL':
        # dataset = ImageFolder(args.data_path, transform=transform)
        raw_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        raw_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
        print("imagenet256-f32-VL")

        dataset_trn = DctImageFolder(raw_val_folder, transform=transforms_trn)
        dataset_val = DctImageFolder(raw_val_folder, transform=transforms_val)

    elif config.dataset.type == 'imagenet256-syndogen-f32-VL':
        # dataset = ImageFolder(args.data_path, transform=transform)
        print("imagenet256-f32-VL-with-syndog-en")

        raw_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        syndog_en_val_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/valid"
        
        if is_eval:
            print("using val for both train and val")
            raw_train_folder = raw_val_folder
            syndog_en_train_folder = syndog_en_val_folder
        else:
            raw_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
            syndog_en_train_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/train"
        
        dataset_trn_im = DctImageFolder(raw_train_folder, transform=transforms_trn)
        print("dataset_trn_im:", len(dataset_trn_im))

        dataset_val_im = DctImageFolder(raw_val_folder, transform=transforms_val)
        print("dataset_val_im:", len(dataset_val_im))

        dataset_trn_syn = DctImageFolder(syndog_en_train_folder, transform=transforms_trn)
        print("dataset_trn_syn:", len(dataset_trn_syn))
        
        dataset_val_syn = DctImageFolder(syndog_en_val_folder, transform=transforms_val)
        print("dataset_val_syn:", len(dataset_val_syn))

        dataset_trn = ConcatDataset([dataset_trn_im,dataset_trn_syn])
        dataset_val = ConcatDataset([dataset_val_im,dataset_val_syn])

    elif config.dataset.type == 'imagenet256-syndogen-f32-VL-Global':
        # dataset = ImageFolder(args.data_path, transform=transform)
        print("imagenet256-f32-VL-with-syndog-en using global efficient 1.6")

        raw_val_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val"
        syndog_en_val_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/valid"
        
        if is_eval:
            print("using val for both train and val")
            raw_train_folder = raw_val_folder
            syndog_en_train_folder = syndog_en_val_folder
        else:
            raw_train_folder = "/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/train/images"
            syndog_en_train_folder="/cpfs01/user/cl424408/datasets/synthdog_images_en/train"
        
        # import pdb
        # pdb.set_trace()

            
        

        # dataset_trn = ImageNet("/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val", split='val', transform=transforms_trn)
        # dataset_val = ImageNet("/cpfs01/user/cl424408/datasets/imagenet-1k/data/raw_images/raw_val", split='val', transform=transforms_val)
        dataset_trn_im = GlobalDctImageFolder(raw_train_folder, transform=transforms_trn)
        print("dataset_trn_im:", len(dataset_trn_im))

        dataset_val_im = GlobalDctImageFolder(raw_val_folder, transform=transforms_val)
        print("dataset_val_im:", len(dataset_val_im))

        dataset_trn_syn = GlobalDctImageFolder(syndog_en_train_folder, transform=transforms_trn)
        print("dataset_trn_syn:", len(dataset_trn_syn))
        
        dataset_val_syn = GlobalDctImageFolder(syndog_en_val_folder, transform=transforms_val)
        print("dataset_val_syn:", len(dataset_val_syn))

        dataset_trn = ConcatDataset([dataset_trn_im,dataset_trn_syn])
        dataset_val = ConcatDataset([dataset_val_im,dataset_val_syn])

    elif config.dataset.type == 'imagenet_u':
        root = root if root else 'data/imagenet'

        def target_transform(_):
            return 0
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn, target_transform=target_transform)
        dataset_val = ImageNet(root, split='val', transform=transforms_val, target_transform=target_transform)
    elif config.dataset.type == 'ffhq':
        root = root if root else 'data/ffhq'
        dataset_trn = FFHQ(root, split='train', transform=transforms_trn)
        dataset_val = FFHQ(root, split='val', transform=transforms_val)
    elif config.dataset.type in ['LSUN-cat', 'LSUN-church', 'LSUN-bedroom']:
        root = root if root else 'data/lsun'
        category_name = config.dataset.type.split('-')[-1]
        dataset_trn = LSUNClass(root, category_name=category_name, transform=transforms_trn)
        dataset_val = LSUNClass(root, category_name=category_name, transform=transforms_trn)
    else:
        raise ValueError('%s not supported...' % config.dataset.type)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')

    return dataset_trn, dataset_val


def block_dct(block):
    return cv2.dct(block.astype(np.float32))

def block_dct_image(image):
    # convert PIL image to cv2 grayscale image
    image = np.array(image.convert('L'))
    height, width = image.shape

    block_size = 8

    block_avg_dct = np.zeros((height//block_size, width//block_size))

    # Quantization matrix (example)
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            quantized_dct_block = np.round(block_dct(block) / Q)

            # compute the non-zero numbers in the quantized dct block
            non_zero = np.count_nonzero(quantized_dct_block)
            block_avg_dct[i//block_size, j//block_size] = non_zero
        
    return block_avg_dct

class DctImageFolder(ImageFolder):
    # 256*256 -> 8x8xN
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # first denormlize the image tensor and transform to PIL image


        img = torch.clamp(sample*0.5+0.5, 0, 1)
        img = Image.fromarray(np.uint8(img.numpy().transpose([1,2,0])*255))
        
        dct_nonzero_block = block_dct_image(img)

        # avg pooling the dct block by 4x4

        dct_nonzero_block = cv2.resize(dct_nonzero_block, (dct_nonzero_block.shape[1]//4, dct_nonzero_block.shape[0]//4), interpolation=cv2.INTER_LINEAR)
        
        # hack 256/512
        num_codes = torch.ceil(torch.tensor(dct_nonzero_block*1024/dct_nonzero_block.sum())+0.0001)
        
        return sample, target, num_codes #, img

class GlobalDctImageFolder(ImageFolder):
    # 256*256 -> 8x8xN
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # first denormlize the image tensor and transform to PIL image


        img = torch.clamp(sample*0.5+0.5, 0, 1)
        img = Image.fromarray(np.uint8(img.numpy().transpose([1,2,0])*255))
        
        dct_nonzero_block = block_dct_image(img)

        # avg pooling the dct block by 4x4

        dct_nonzero_block = cv2.resize(dct_nonzero_block, (dct_nonzero_block.shape[1]//4, dct_nonzero_block.shape[0]//4), interpolation=cv2.INTER_LINEAR)
        
        global_efficient = 1.6 # (1.6) for 1024 code (50000*1024*16)/514092267

        # hack 256/512
        num_codes = torch.ceil(torch.tensor(dct_nonzero_block*global_efficient)+0.0001)
        
        return sample, target, num_codes #, img