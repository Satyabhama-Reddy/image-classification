### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import DenseNet
from ImageUtils import parse_record, parse_record_test
from tqdm import tqdm
import statistics as st

# from torchsummary import summary

"""This script defines the training, validation and testing process for Densenet model.
"""

class MyModel(object):

	def __init__(self, configs):
		self.configs = configs
		self.network = DenseNet(configs)
		self.model_setup()

	def model_setup(self):
		self.network.to(self.configs['device'])
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.configs['optimizer']['learning_rate'], momentum=self.configs['optimizer']['momentum'], weight_decay=self.configs['optimizer']['weight_decay'])
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.configs['lr_scheduler']['step_size'], gamma=self.configs['lr_scheduler']['gamma'])

	def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

		# summary(self.network,(3,32,32), batch_size= configs['batch_size'], device = self.configs['device'])
		batch_size = configs['batch_size']
		epochs = configs['num_epochs']
		num_samples = x_train.shape[0]
		num_batches = num_samples // batch_size

		# Initializing validation data
		x_valid = np.array([parse_record(i, training=False) for i in x_valid])
		x_valids = torch.split(torch.tensor(x_valid, device=self.configs['device'], dtype=torch.float), batch_size)
		y_valid = torch.tensor(y_valid, device=self.configs['device'])
		print('### Training... ###')
		quarter_epochs = epochs / 4
		for epoch in tqdm(range(1, epochs+1)):
			self.network.train()
			start_time = time.time()
			shuffle_index = np.random.permutation(num_samples)
			curr_x_train = x_train[shuffle_index]
			curr_y_train = y_train[shuffle_index]

			for i in tqdm(range(num_batches),position=0, leave=True):

				# blur first quarter of the epochs:
				blur = True if (configs['blurring'] and epoch < quarter_epochs) else False
				x_batch = curr_x_train[i * batch_size:(i + 1) * batch_size, :]
				x_batch = np.array([parse_record(x, training=True, blur=blur) for x in x_batch])
				x_batch = torch.tensor(x_batch, device=self.configs['device'], dtype=torch.float)

				y_batch = curr_y_train[i * batch_size:(i + 1) * batch_size]
				y_batch = torch.tensor(y_batch, device=self.configs['device'])
				y_generated = self.network(x_batch)

				loss = self.cross_entropy_loss(y_generated, y_batch)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

			duration = time.time() - start_time
			print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

			if epoch % configs['checkpoint_interval'] == 0:
				self.save(epoch)
				if(loss < 0.08):
					self.validate(x_valids, y_valid)

	def validate(self, xs, y):
		# print('### Validation ###')
		self.network.eval()
		preds = []
		with torch.no_grad():
			for i in range(len(xs)):
				y_generated = self.network(xs[i])
				preds.append(torch.argmax(y_generated, dim=1))
			preds = torch.cat(preds)
		print('Validation Accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))

	def evaluate(self, x, y, checkpoint_num_list = None):
		self.network.eval()
		print('### Test or Validation ###')
		for checkpoint_num in checkpoint_num_list:
			checkpointfile = os.path.join(self.configs['save_models_dir'], 'model-%d.ckpt' % (checkpoint_num))
			self.load(checkpointfile)

			ys_generated = self.predict_prob(x, None)
			preds = torch.argmax(ys_generated, dim=1)

			y = torch.tensor(y, device=self.configs['device'])

			print('Densenet Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))

	def predict_prob(self, x, checkpoint_num = None, batch_size=32):
		if checkpoint_num:
			checkpointfile = os.path.join(self.configs['save_models_dir'], 'model-%d.ckpt' % (checkpoint_num))
			self.load(checkpointfile)
		self.network.eval()
		ys_generated = []
		with torch.no_grad():
			inputs = np.array([parse_record(i, training=False) for i in x])
			inputs = torch.split(torch.tensor(inputs, device=self.configs['device'], dtype=torch.float), batch_size)
			for i in tqdm(range(len(inputs))):
				ys_generated.append(self.network(inputs[i]))
			ys_generated = torch.cat(ys_generated)
			ys_generated = nn.functional.softmax(ys_generated, dim=1)
		return ys_generated

	def save(self, epoch):
		checkpoint_path = os.path.join(self.configs['save_models_dir'], 'model-%d.ckpt'%(epoch))
		os.makedirs(self.configs['save_models_dir'], exist_ok=True)
		torch.save(self.network.state_dict(), checkpoint_path)
		print("Checkpoint has been created.")

	def load(self, checkpoint_name):
		ckpt = torch.load(checkpoint_name, map_location=self.configs['device'])
		self.network.load_state_dict(ckpt, strict=True)
		print("Restored model parameters from {}".format(checkpoint_name))

