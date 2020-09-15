# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:38:59 2020

@author: putta
"""

import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms
import struct
import glob
import cv2
import matplotlib.pyplot as plt
import sys
import numpy
from PIL import Image
numpy.set_printoptions(threshold=sys.maxsize)
from torch import device 


use_cuda = True


# Data
'''param alpha_t: Scaling factor for spike influence on trace
    :param tau_t: Trace decay time constant
    :param dt: Duration of single timestep
'''
class PuttatidaMNIST(nn.Module):
	def __init__(self):
		super(PuttatidaMNIST, self).__init__()
        
    # Input

		self.conv1 = snn.Convolution(6, 30, 5, 0.8, 0.05) #4 in, 20 out, 34 kernal (out),
		self.conv1_t = 15 #threshold
		self.k1 = 5 #k-winner takes all
		self.r1 = 3 #inhibition window

		self.conv2 = snn.Convolution(30, 250, 3, 0.8, 0.05)
		self.conv2_t = 10 
		self.k2 = 8
		self.r2 = 1

		self.conv3 = snn.Convolution(250, 200, 5, 0.8, 0.05)

		self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        #self.anti_stdp1 = snn.STDP(self.conv1, (-0.004, 0.0005))
		self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        #self.anti_stdp2 = snn.STDP(self.conv1, (-0.004, 0.0005))
		self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
		self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
		self.max_ap = Parameter(torch.Tensor([0.15]))

		self.decision_map = []
		for i in range(10):
			self.decision_map.extend([i]*20)

		self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
		self.spk_cnt1 = 0
		self.spk_cnt2 = 0

 
	def forward(self, input, max_layer):
		input = sf.pad(input.float(), (2,2,2,2), 0)
		if self.training:
			pot = self.conv1(input) #potential
			spk, pot = sf.fire(pot, self.conv1_t, True) #returns spikewave tensor
			if max_layer == 1: #layer1
				self.spk_cnt1 += 1
				if self.spk_cnt1 >= 500: 
					self.spk_cnt1 = 0 #reset to 0
                    
					ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2 #learning rate (positive)
					ap = torch.min(ap, self.max_ap)
					an = ap * -0.75 #learning rate (negative)
					self.stdp1.update_all_learning_rate(ap.item(), an.item())
				pot = sf.pointwise_inhibition(pot) #basically a standaard inhibition, returns tensor of inhibition potential
				spk = pot.sign() #return tensor with the sign of input
				winners = sf.get_k_winners(pot, self.k1, self.r1, spk) #return a list of winners
				self.ctx["input_spikes"] = input
				self.ctx["potentials"] = pot
				self.ctx["output_spikes"] = spk
				self.ctx["winners"] = winners
				return spk, pot
			spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)) #perform pooling and return spike input
			pot = self.conv2(spk_in) #return potential tensors
			spk, pot = sf.fire(pot, self.conv2_t, True)
			if max_layer == 2:
				self.spk_cnt2 += 1
				if self.spk_cnt2 >= 500:
					self.spk_cnt2 = 0
					ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
					ap = torch.min(ap, self.max_ap)
					an = ap * -0.75
					self.stdp2.update_all_learning_rate(ap.item(), an.item())
				pot = sf.pointwise_inhibition(pot)
				spk = pot.sign()
				winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
				self.ctx["input_spikes"] = spk_in
				self.ctx["potentials"] = pot
				self.ctx["output_spikes"] = spk
				self.ctx["winners"] = winners
				return spk, pot
			spk_in = sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2))
			pot = self.conv3(spk_in)
			spk = sf.fire(pot)
			winners = sf.get_k_winners(pot, 1, 0, spk)
			self.ctx["input_spikes"] = spk_in
			self.ctx["potentials"] = pot
			self.ctx["output_spikes"] = spk
			self.ctx["winners"] = winners
			output = -1
			if len(winners) != 0:
				output = self.decision_map[winners[0][0]]
			return output
		else: #test
			pot = self.conv1(input)
			spk, pot = sf.fire(pot, self.conv1_t, True) #take in potential tensor of convolved input
			if max_layer == 1:
				return spk, pot
			pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
			spk, pot = sf.fire(pot, self.conv2_t, True)
			if max_layer == 2:
				return spk, pot
			pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
			spk = sf.fire(pot)
			winners = sf.get_k_winners(pot, 1, 0, spk)
			output = -1
			if len(winners) != 0:
				output = self.decision_map[winners[0][0]]
			return output
	
	def stdp(self, layer_idx):
		if layer_idx == 1:
			self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
		if layer_idx == 2:
			self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

	def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
		self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
		self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

	def reward(self):
		self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

	def punish(self):
		self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])


    
def train_unsupervise(network, data, layer_idx):
	network.train()
	for i in range(len(data)):
		data_in = data[i]
		if use_cuda:
			data_in = data_in.cuda()
		network(data_in, layer_idx)
		network.stdp(layer_idx)

def train_rl(network, data, target):
	network.train()
	perf = np.array([0,0,0]) # correct, wrong, silence
	for i in range(len(data)):
		data_in = data[i]
		target_in = target[i]
		if use_cuda:
			data_in = data_in.cuda()
			target_in = target_in.cuda()
		d = network(data_in, 3)
		if d != -1:
			if d == target_in:
				perf[0]+=1
				network.reward()
			else:
				perf[1]+=1
				network.punish()
		else:
			perf[2]+=1
	return perf/len(data)

def test(network, data, target):
	network.eval()
	perf = np.array([0,0,0]) # correct, wrong, silence
	for i in range(len(data)):
		data_in = data[i]
		target_in = target[i]
		if use_cuda:
			data_in = data_in.cuda()
			target_in = target_in.cuda()
		d = network(data_in, 3)
		if d != -1:
			if d == target_in:
				perf[0]+=1
			else:
				perf[1]+=1
		else:
			perf[2]+=1
	return perf/len(data) #show performance values of network

def time_dim(input):  
	return input.unsqueeze(0) #add time dimension

class S1C1Transform:
    def __init__(self, filter, timesteps = 15):
        #self.gs = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        self.filter = filter #gabor filer
        self.temporal_transform = utils.Intensity2Latency(number_of_spike_bins = 15, to_spike = True) #pixel to rank coding
        self.cnt=0
    def __call__(self, image):
        if self.cnt%1000 == 0:
            print(self.cnt)
        self.cnt+=1
       #image = self.gs(image)
        image = self.to_tensor(image)*255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image,8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()

kernels = [	utils.GaborKernel(window_size = 3, orientation = 0+22.5),
			utils.GaborKernel(3, 30+22.5),
            utils.GaborKernel(3, 60+22.5),
            utils.GaborKernel(3, 90+22.5),
			utils.GaborKernel(3, 120+22.5),
			utils.GaborKernel(3, 150+22.5)]

filter = utils.Filter(kernels, use_abs = True)
s1c1 = S1C1Transform(filter)

#filter = utils.Filter(kernels, padding = 6, thresholds = 50)
#s1c1 = S1C1Transform(filter)

data_root = "data"
data_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
data_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
train_dataloader = DataLoader(data_train, batch_size=1000, shuffle=False)
test_dataloader = DataLoader(data_test, batch_size=len(data_test), shuffle=False)


'''
#visualization
ex = enumerate(train_dataloader)
batch_idx, (data_exam, data_label) = next(ex)
plt.style.use('seaborn-white')
plt_idx = 0

sw = data_exam[5]
for f in range(2):
	for t in range(5):
		plt_idx += 1
		ax = plt.subplot(5,5, plt_idx)
		plt.setp(ax, xticklabels=[])
		plt.setp(ax, yticklabels=[])
		if t == 0:
			ax.set_ylabel('Feature ' + str(f))
		plt.imshow(sw[t][f],cmap='gray')

		if f == 3:
			ax = plt.subplot(5, 5, plt_idx + 5)
			plt.setp(ax, xticklabels=[])
			plt.setp(ax, yticklabels=[])
			if t == 0:
				ax.set_ylabel('Sum')
			ax.set_xlabel('t = ' + str(t))
			plt.imshow(sw[t].sum(dim=0).numpy(),cmap='gray')
plt.show()

'''
puttatida = PuttatidaMNIST()
if use_cuda:
	puttatida.cuda()
# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1_mnistP2.net"):
	puttatida.load_state_dict(torch.load("saved_l1_mnistP2.net"))
else:
	for epoch in range(2):
		print("Epoch", epoch)
		iter = 0
		for data,targets in train_dataloader:
			print("Iteration", iter)
			train_unsupervise(puttatida, data, 1)
			print("Done!")
			iter+=1
	torch.save(puttatida.state_dict(), "saved_l1_mnistP2.net")
# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2_mnistP2.net"):
	puttatida.load_state_dict(torch.load("saved_l2_mnistP2.net"))
else:
	for epoch in range(4):
		print("Epoch", epoch)
		iter = 0  
		for data,targets in train_dataloader:
			print("Iteration", iter)
			train_unsupervise(puttatida, data, 2)
			print("Done!")
			iter+=1
	torch.save(puttatida.state_dict(), "saved_l2_mnistP2.net")

# initial adaptive learning rates
apr = puttatida.stdp3.learning_rate[0][0].item() #reward learning rate (positive)
anr = puttatida.stdp3.learning_rate[0][1].item() #reward learning rate (negative)
app = puttatida.anti_stdp3.learning_rate[0][1].item() #punishment learning rate (positive)
anp = puttatida.anti_stdp3.learning_rate[0][0].item() #punishment leanring rate (negative)

adaptive_min = 0
adaptive_int = 1
apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
'''
#C2 Visualisation


feature = torch.stack([k() for k in kernels])
cstride = (1,1)
# C1 Features #
feature,cstride = vis.get_deep_feature(feature, cstride, (3, 3), (2, 2))
# S2 Features #
feature,cstride = vis.get_deep_feature(feature, cstride, (5,5), (2,2), puttatida.conv1.weight)

feature_idx = 0
for r in range(4):
    for c in range(5):
        ax = plt.subplot(4, 5, feature_idx+1)
        plt.xticks([])
        plt.yticks([])
        plt.setp(ax, xticklabels=[])
        plt.setp(ax, yticklabels=[])
        plt.imshow(feature[feature_idx].numpy(),cmap='gray')
        feature_idx += 1
plt.show()
'''


# Training The Third Layer
print("Training the third layer")
for epoch in range(10):
	print("Epoch #:", epoch)
	perf_train = np.array([0.0,0.0,0.0])
	for data,targets in train_dataloader:
		perf_train_batch = train_rl(puttatida, data, targets)
		print(perf_train_batch)
		#update adaptive learning rates
		apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
		anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
		app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
		anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
		puttatida.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
		perf_train += perf_train_batch
	perf_train /= len(train_dataloader)
	if best_train[0] <= perf_train[0]:
		best_train = np.append(perf_train, epoch)
	print("Current Train:", perf_train)
	print("   Best Train:", best_train)

	for data,targets in test_dataloader:
		perf_test = test(puttatida, data, targets)
		if best_test[0] <= perf_test[0]:
			best_test = np.append(perf_test, epoch)
			torch.save(puttatida.state_dict(), "saved_mnistP2.net")
		print(" Current Test:", perf_test)
		print("    Best Test:", best_test)
        
  
