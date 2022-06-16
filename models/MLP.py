from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F
import torch
from config import DefaultConfig

class MLP(nn.Module):

	def __init__(self,input_size,middle_size,output_size):

		super(MLP, self).__init__()

		self.input_size = input_size
		self.middle_size = middle_size
		self.output_size = output_size
		self.module = nn.Sequential(
			
			nn.Linear(self.input_size, self.middle_size),
			nn.Dropout(0.5),
			nn.ReLU(inplace=True),
			nn.Linear(self.middle_size, self.output_size),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.module(x)
		return x
