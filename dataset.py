import torch.utils.data as data
from PIL import Image
import pandas as pd
from skimage import io
import os
import os.path
import numpy as np
from torchvision import datasets, transforms
from random import randint

class CARDataset(data.Dataset):

	def __init__(self, data_dir,flag = 'train', transform = None, mirror = False):

		self.data_dir = data_dir
		self.pairs = pd.read_csv(data_dir + flag + '.txt', header = None)
		self.transform = transform
		self.mirror = mirror

	def __getitem__(self, index):
		
		data = self.data_dir + self.pairs.ix[index,0]
		im_path = data.split(' ')[0]
		label = int(data.split(' ')[1])

		im = io.imread(im_path)
		if len(im.shape) < 3:
			im = np.repeat(im[:,:,np.newaxis],3,2)
		im = Image.fromarray(im,mode = 'RGB')

		ind = randint(0,5)
		if self.transform is not None:
			if len(self.transform) > 1:

				im = self.transform[ind](im)

			else:
				im = self.transform[0](im)

		return im,label

	def __len__(self):
		return len(self.pairs)


class FineGrainVerificationDataset(data.Dataset):
    def __init__(self, list, transform=None, mirror=False):

        self.pairs = pd.read_csv(list, header = None)
        self.transform = transform
        self.mirror = mirror
        self.list = list

    def __getitem__(self, index):
        im1_path = self.pairs.ix[index,0]
        im2_path = self.pairs.ix[index,2]
        # data_dir = '/media/ramdisk/data/'
        data_dir = '/media/ramdisk/cub_data/'        
        im1 = io.imread(data_dir + im1_path)
        im2 = io.imread(data_dir + im2_path)
        if len(im1.shape) < 3:
            im1 = np.repeat(im1[:, :, np.newaxis], 3, 2)
        if len(im2.shape) < 3:
            im2 = np.repeat(im2[:, :, np.newaxis], 3, 2)
        im1 = Image.fromarray(im1, mode='RGB')
        im2 = Image.fromarray(im2, mode='RGB')
        label = int(self.pairs.ix[index,1])
        

        ind1 = randint(0,4)
        ind2 = randint(0,4)

        if self.transform is not None:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        
        return im1,im2,label

    def __len__(self):
        return len(self.pairs)