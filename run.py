import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from dataset import *
import argparse
import torch.optim as optim
from model import *
# import torchsample as ts

parser = argparse.ArgumentParser(description='Bilinear model')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--modelname', type=str, default='bilinear_')
parser.add_argument('--pretrained', type=bool, default=True)
args = parser.parse_args()


rotation = transforms.Compose([transforms.Resize((224,224)),
                            transforms.RandomRotation(10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = (0.485, 0.456, 0.406),
							std = (0.29, 0.224, 0.225))
                            ])
h_crop = transforms.Compose([transforms.Resize((224+5,224)),
                            transforms.RandomCrop((224,224)),
                            transforms.ToTensor(),
							transforms.Normalize(mean = (0.485, 0.456, 0.406),
												std = (0.29, 0.224, 0.225))
							])
w_crop = transforms.Compose([transforms.Resize((224,224+5)),
                            transforms.RandomCrop((224,224)),
                            transforms.ToTensor(),
							transforms.Normalize(mean = (0.485, 0.456, 0.406),
												std = (0.29, 0.224, 0.225))
                            ])
crop = transforms.Compose([transforms.Resize((224+15,224+15)),
                            transforms.RandomCrop((224,224)),
                            transforms.ToTensor(),
							transforms.Normalize(mean = (0.485, 0.456, 0.406),
												std = (0.29, 0.224, 0.225))
                            ])
c_crop = transforms.Compose([transforms.CenterCrop(size = 224),
							transforms.ToTensor(),
							transforms.Normalize(mean = (0.485, 0.456, 0.406),
												std = (0.29, 0.224, 0.225))
							])
normal = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
							transforms.Normalize(mean = (0.485, 0.456, 0.406),
												std = (0.29, 0.224, 0.225))
                            ])

train_transforms = [rotation,h_crop,w_crop,crop,c_crop,normal]
# # train_transforms = [normal]
test_transforms = [normal]
data_dir = '/media/ramdisk/cars_data/'
train_loader = torch.utils.data.DataLoader(
    CARDataset(data_dir,'train_c_first100',
                transform= train_transforms),batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
	CARDataset(data_dir,'test_c_first100',
				transform = test_transforms), batch_size = args.batch_size, shuffle = False, num_workers = 4)

# verification_loader = torch.utils.data.DataLoader(
#     FineGrainVerificationDataset('/media/ramdisk/cub_data/test_pairwise_balanced_short.txt',
#                     transform= normal),
#     batch_size=32, shuffle=True)

net = BiLinearModel()
net.features = torch.nn.DataParallel(net.features).cuda()
net.loadweight_from('models/bilinear_stage1/bilinear__0.7116_epoch-149.pth')
train(net,train_loader, test_loader,lr=args.lr, num_epochs=args.epochs, train_log = 'logs/bilinear_stage2.txt', model_name = 'models/bilinear_stage2/')
# verification_test(net,verification_loader)