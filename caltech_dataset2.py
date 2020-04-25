from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def getTrainVal(self):
        curr = self.data
        random.shuffle(curr)
        limit = int((len(curr) * 2) / 3)
        train = range(0, limit)
        val = range(limit, len(curr))
        return train, val

    def retrieveTrainVal(self):
        samples = range(0,len(self.data))
        labels = []
        for i in samples:
            labels.append(self.data[i][1])
        train, val, y_train, y_val = train_test_split(samples,labels,test_size=0.5,random_state=42,stratify=labels)
        index_train = []
        index_val = []
        for i, el in enumerate(samples):
            if el in train:
                index_train.append(i)
            else:
                index_val.append(i)

        print("retrieveTrainVal")
        print(len(index_train))
        print(len(index_val))
        return index_train, index_val

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
        self.data = []
        self.classes = []
        # (split files are called 'train.txt' and 'test.txt')

        provData = []

        f = open("Caltech101/{}.txt".format(split), "r")
        for x in f:

            dire = "Caltech101/101_ObjectCategories/{}".format(x.rstrip())
            filename = x.split("/")
            if filename[0] != "BACKGROUND_Google":
                couple = []
                if filename[0] in self.classes:
                    img = pil_loader(dire)
                    index = len(self.classes) - 1
                    couple.append(img)
                    couple.append(index)
                    provData.append(couple)
                else:
                    self.classes.append(filename[0])
                    img = pil_loader(dire)
                    index = len(self.classes) - 1
                    couple.append(img)
                    couple.append(index)
                    provData.append(couple)

        f.close()
        print(len(provData))
        print(len(self.classes))
        print(self.classes)


        aug2 = transforms.Compose([
                                   transforms.RandomHorizontalFlip(p=0.5)

                                   # Normalizes tensor with mean and standard deviation
                                   ])
        aug3 = transforms.Compose([
                                   transforms.RandomRotation(45, resample=False, expand=False, center=None, fill=None)

                                   # Normalizes tensor with mean and standard deviation
                                   ])

        for i in range(0,len(provData)):
            couple = []
            couple.append(provData[i][0])
            couple.append(provData[i][1])
            self.data.append(couple)
            couple = []
            img = aug2(provData[i][0])
            couple.append(img)
            couple.append(provData[i][1])
            self.data.append(couple)
            couple = []
            img = aug2(provData[i][0])
            couple.append(img)
            couple.append(provData[i][1])
            self.data.append(couple)
            couple = []
            img = aug2(provData[i][0])
            couple.append(img)
            couple.append(provData[i][1])
            self.data.append(couple)
            couple = []
            img = aug2(provData[i][0])
            couple.append(img)
            couple.append(provData[i][1])
            self.data.append(couple)

        print(len(self.data))

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data[index][0], self.data[index][1]  # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data)  # Provide a way to get the length (number of elements) of the dataset
        return length


