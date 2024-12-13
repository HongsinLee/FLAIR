# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from backbone.mobilenet_v2 import mobilenet_v2
from backbone.wideresnet import wrn3410
from PIL import Image
from torchvision.datasets import SVHN

from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val

class TSVHN(SVHN):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        if train == True : 
            self.split = "train"
        elif train == False :
            self.split = "test"
        
        super(TSVHN, self).__init__(root, self.split, transform, target_transform, download=True)
        self.targets = self.labels

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
  
        img, target = self.data[index], int(self.targets[index])
    
        # to return a PIL Image
        # img = Image.fromarray(img, mode='RGB')
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        original_img = img.copy()


        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]


        return img, target

class MySVHN(SVHN):
    """
    Overrides the SVHN dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        if train == True : 
            self.split = "train"
        elif train == False :
            self.split = "test"
        
        super(MySVHN, self).__init__(root, self.split, transform, target_transform, download=True)

        self.targets = self.labels


    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
     
        # to return a PIL Image
        # img = Image.fromarray(img, mode='RGB')
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)
   

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

 
        return img, target, not_aug_img


class SequentialSVHN(ContinualDataset):

    NAME = 'seq-SVHN'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
            # [transforms.RandomCrop(32, padding=4),
            #  transforms.RandomHorizontalFlip(),
             [transforms.ToTensor()
            # transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615))
                                  ])

    def get_data_loaders(self):
        transform_aug = []
        if self.args.aug == 'aua':
            transform_aug = [transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)]
        elif self.args.aug == 'ra':
            transform_aug = [transforms.RandAugment(2,8)]
        elif self.args.aug == 'none':
            transform_aug = []
        #transform = self.TRANSFORM
        transform = transforms.Compose(transform_aug + 
            [
            #     transforms.RandomCrop(32, padding=4),
            #  transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])


        test_transform = transforms.Compose(
            [transforms.ToTensor() 
            #self.get_normalization_transform()
            ])

        train_dataset = MySVHN(base_path() + 'SVHN', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TSVHN(base_path() + 'SVHN',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialSVHN.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(arch="None"):
        mean = torch.tensor((0.4914, 0.4822, 0.4465)).cuda().view(-1, 1, 1)
        std = torch.tensor((0.2470, 0.2435, 0.2615)).cuda().view(-1, 1, 1)
        if arch=="RES-18":
            return resnet18(mean, std, SequentialSVHN.N_CLASSES_PER_TASK * SequentialSVHN.N_TASKS)
        elif arch=="MN-V2":
            return mobilenet_v2(mean,std,SequentialSVHN.N_CLASSES_PER_TASK * SequentialSVHN.N_TASKS)
        elif arch=="WRN-34-10":
            return wrn3410(mean,std,SequentialSVHN.N_CLASSES_PER_TASK * SequentialSVHN.N_TASKS)
        else:
            raise NotImplementedError

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        #32 -- > 128
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialSVHN.get_batch_size()

    @staticmethod
    def get_robust_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [int(args.n_epochs * 0.48), int(args.n_epochs * 0.62), int(args.n_epochs * 0.80)], gamma=0.1, verbose=False)
        return scheduler