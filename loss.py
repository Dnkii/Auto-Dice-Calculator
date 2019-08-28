import torch
import torch.nn as nn
from sklearn import preprocessing
import numpy as np
# import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(threshold=np.inf)
class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()

	def	forward(self, input, target):
		N = target.size(0)
		# print(N)
		smooth = 1
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
		input_np = torch.detach(input_flat).cpu().numpy()
		input_np = preprocessing.Binarizer(threshold=0.5).transform(input_np)
		input_flat = torch.from_numpy(input_np)
		input_flat = input_flat.to(device)
		# input_flat = torch.detach(input_flat).cpu().numpy()
		# target_flat = torch.detach(target_flat).cpu().numpy()
		# print(input_flat)
		# print(target_flat)
		# os.system("pause")
		# print(input_flat.size())


		intersection = input_flat * target_flat
		# if target_flat.sum()==0:
		# 	loss = input_flat.sum()/input_flat.sum()
		# else:
		# loss = 2 * (intersection.sum()) / (input_flat.sum() + target_flat.sum())
		loss1 = 2 * intersection.sum()
		loss2 = input_flat.sum() + target_flat.sum()
		# loss = 1 - loss/ N
		# loss = 2 * (intersection.sum(1)) / (input_flat.sum(1) + target_flat.sum(1)+smooth)
		# loss = 1 - loss/ N

		# loss =1-(intersection.sum(1)+smooth) / (target_flat.sum(1)+smooth)
		
		# print(intersection.sum(1))
		# print(input_flat.sum(1))
		# print(target_flat.sum(1))
		# os.system("pause")
		return loss1,loss2
