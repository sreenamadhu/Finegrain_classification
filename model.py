import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from time import strftime, localtime
import numpy as np
import math
from torchvision.models.vgg import model_urls

class BiLinearModel(nn.Module):
	def __init__(self):

		super(BiLinearModel, self).__init__()

		model_urls['vgg16'] = model_urls['vgg16'].replace('https://','http://')
		self.features = torchvision.models.vgg16(pretrained=True).features
		self.features = torch.nn.Sequential(*list(self.features.children())
											[:-1])
		self.fc = torch.nn.Linear(512**2, 200)


	def forward(self, x):

		batch_size = x.size()[0]
		x1 = self.features(x)

		x1 = x1.view(batch_size,-1,14*14)
		x = torch.bmm(x1,torch.transpose(x1,1,2)) / (14*14)

		assert x.size() == (batch_size,512,512)
		x = x.view(batch_size,-1)
		x = torch.sqrt(x+1e-5)
		x = torch.nn.functional.normalize(x)
		x = self.fc(x)
		assert x.size() == (batch_size,200)
		return x

	def loadweight_from(self, pretrain_path):
		pretrained_dict = torch.load(pretrain_path)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.load_state_dict(model_dict)

	def cls_loss(self, x, y):
		loss = F.nll_loss(F.log_softmax(x, dim=1), y)
		return loss

def train(model,trainloader,validloader,lr = 0.001, num_epochs=10,train_log = 'logs/bilinear_basic.txt',model_name = 'models/bilinear_basic/'):
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		model.cuda()

	# for name,x in model.named_parameters():
	# 	if 'features' in name:
	# 		x.requires_grad= False
	optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = lr)
	fd = open(train_log, 'w+')
	for epoch in range(num_epochs):

		model.train()
		torch.set_grad_enabled(True)
		train_loss = 0.0
		total = 0
		correct = 0

		for batch_idx, (image, label) in enumerate(trainloader):


			image = image.float()
			if use_cuda:
				image, label = image.cuda(), label.cuda()
			optimizer.zero_grad()
			image, label = Variable(image), Variable(label)
			feat = model.forward(image)
			loss = model.cls_loss(feat, label)
			loss.backward()
			optimizer.step()
			train_loss += loss
			total += label.size(0)
			_, predicted = torch.max(feat.data, 1)
			correct += (predicted == label).sum().item()
			# print("    #Iter %3d: Training Loss: %.3f" % (
			# 	batch_idx, loss.data[0]))
			# # print(total)
		train_acc = correct/float(total)


		# validate
		model.eval()
		torch.set_grad_enabled(False)
		valid_loss = 0.0
		total = 0
		correct = 0
		tp = 0
		tn = 0
		pos = 0
		neg = 0
		for batch_idx, (image, label) in enumerate(validloader):
			image = image.float()
			if use_cuda:
				image, label = image.cuda(), label.cuda()
			image, label = Variable(image), Variable(label)
			feat = model.forward(image)
			valid_loss += model.cls_loss(feat, label)
			# compute the accuracy
			total += label.size(0)

			_, predicted = torch.max(feat.data, 1)
			pos += label.sum()
			neg += (1 - label).sum()
			correct += (predicted == label).sum().item()
			tp += (predicted * label).sum().item()
			tn += ((1 - predicted) * (1 - label)).sum().item()
		valid_acc = correct / float(total)
		tpr = float(tp) / float(pos)
		tnr = float(tn) / float(neg)
		print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
		print("#Epoch {}: Train Loss: {:.4f},Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid tpr: {:.4f}, Valid tnr: {:.4f}".
			   format(epoch, train_loss / len(trainloader.dataset),train_acc, valid_loss / len(validloader.dataset), valid_acc, tpr, tnr))
		fd.write('#Epoch {}: Train Loss: {:.4f},Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid tpr: {:.4f}, Valid tnr: {:.4f} \n'.
			   format(epoch, train_loss / len(trainloader.dataset),train_acc, valid_loss / len(validloader.dataset), valid_acc, tpr, tnr))
		torch.save(model.state_dict(),model_name +'{}_{:.4f}_epoch-{}.pth'.format('bilinear_', valid_acc, epoch))

def verification_test(model,validloader):
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		model.cuda()
	model.eval()
	torch.set_grad_enabled(False)
	total = 0
	matches = 0
	nonmatches = 0
	for batch_idx,(im1,im2,label) in enumerate(validloader):
		im1,im2 = im1.float(),im2.float()
		if use_cuda:
			im1,im2,label = im1.cuda(),im2.cuda(),label.cuda()
		im1,im2 = Variable(im1),Variable(im2)
		feat1 = model.forward(im1)
		feat2 = model.forward(im2)
		total += label.size(0)

		_,predicted1 = torch.max(feat1.data,1)
		_,predicted2 = torch.max(feat2.data,1)
		# print((predicted1 == predicted2))
		# print(label)
		label = label.byte()
		matches+= ((predicted1 == predicted2)*label).sum().item()

		nonmatches += ((predicted1 != predicted2)*(1-label)).sum().item()
		print('{}/{}'.format(batch_idx,int(len(validloader.dataset)/16.0)))

	correct = matches + nonmatches
	valid_acc = correct/float(total)
	print('Accuracy : {}'.format(valid_acc))