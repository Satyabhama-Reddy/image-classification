import torch
import os

# Class to help make predictions on Ensembled DenseNet and Resnet models.
class Ensembled(object):
	def __init__(self, resnet, densenet, resnet_ratio=0.5):
		self.densenet = densenet
		self.resnet = resnet
		self.resnet_ratio = resnet_ratio

	def predict_prob(self, x_test, resnet_checkpoint_num = None, densenet_checkpoint_num = None):
		densenet_prob = self.densenet.predict_prob(x_test, densenet_checkpoint_num)
		resnet_prob = self.resnet.predict_prob(x_test, resnet_checkpoint_num)
		ensembled_prob = torch.add(self.resnet_ratio*resnet_prob,(1-self.resnet_ratio)*densenet_prob)
		return ensembled_prob

	def evaluate(self, x, y, resnet_checkpoint_num = None, densenet_checkpoint_num = None):

		y_generated = self.predict_prob(x, resnet_checkpoint_num, densenet_checkpoint_num)
		preds = torch.argmax(y_generated, dim=1)
		y = torch.tensor(y, device=self.densenet.configs['device'])

		print('Ensembled Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))
