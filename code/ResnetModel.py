### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import numpy as np
from ResnetNetwork import ResNet
from ImageUtils import parse_record, parse_record_test
from tqdm import tqdm
import statistics as st

# from torchsummary import summary

"""This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
	def __init__(self, config):
		super(Cifar, self).__init__()
		self.config = config
		self.network = ResNet(
			self.config["resnet_version"],
			self.config["resnet_size"],
			self.config["num_classes"],
			self.config["first_num_filters"],
		)
		### YOUR CODE HERE
		# define cross entropy loss and optimizer
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.1, momentum=0.9, weight_decay=self.config["weight_decay"])
		self.network.to(self.config['device'])

		### YOUR CODE HERE
	
	def train(self, x_train, y_train, max_epoch):
		self.network.train()
		# Determine how many batches in an epoch
		num_samples = x_train.shape[0]
		num_batches = num_samples // self.config["batch_size"]
		print('### Training... ###')
		for epoch in range(1, max_epoch+1):
			start_time = time.time()
			# Shuffle
			shuffle_index = np.random.permutation(num_samples)
			curr_x_train = x_train[shuffle_index]
			curr_y_train = y_train[shuffle_index]

			### YOUR CODE HERE
			# Set the learning rate for this epoch
			# Usage example: divide the initial learning rate by 10 after several epochs
			if epoch % 30 == 0 or epoch % 60 == 0:
				self.optimizer.param_groups[0]['lr'] /= 10
			### YOUR CODE HERE
			
			for i in range(num_batches):
				### YOUR CODE HERE
				# Construct the current batch.
				# Don't forget to use "parse_record" to perform data preprocessing.
				# Don't forget L2 weight decay
				x_batch = curr_x_train[i * self.config["batch_size"]:(i + 1) * self.config["batch_size"], :]
				x_batch = np.array([parse_record(x, training=True) for x in x_batch])
				x_batch = torch.tensor(x_batch, device=self.config['device'], dtype=torch.float)

				y_batch = curr_y_train[i * self.config["batch_size"]:(i + 1) * self.config["batch_size"]]
				y_batch = torch.tensor(y_batch, device=self.config['device'])

				y_generated = self.network(x_batch)

				loss = self.cross_entropy_loss(y_generated, y_batch)
				### YOUR CODE HERE
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
			
			duration = time.time() - start_time
			print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

			if epoch % self.config["save_interval"] == 0:
				self.save(epoch)


	def evaluate(self, x, y, checkpoint_num_list):
		self.network.eval()
		print('### Test or Validation ###')
		for checkpoint_num in checkpoint_num_list:
			checkpointfile = os.path.join(self.config["modeldir"], 'model-%d.ckpt' % (checkpoint_num))
			self.load(checkpointfile)

			ys_generated = self.predict_prob(x, None)
			preds = torch.argmax(ys_generated, dim=1)

			y = torch.tensor(y, device=self.config['device'])
			print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))

	def test_or_validate_tta(self, x, y, checkpoint_num_list):
		self.network.eval()
		print('### Test or Validation ###')
		for checkpoint_num in checkpoint_num_list:
			checkpointfile = os.path.join(self.config["modeldir"], 'model-%d.ckpt' % (checkpoint_num))
			self.load(checkpointfile)

			preds = []
			with torch.no_grad():
				for i in tqdm(range(x.shape[0])):
					### YOUR CODE HERE
					outputs = []
					inputs = parse_record_test(x[i], False)
					for input in inputs:
						input = np.expand_dims(input, axis=0)
						input = torch.FloatTensor(input)
						outputs.append(self.network.forward(input))
					temp_preds = []
					for output in outputs:
						_, predict = torch.max(output, 1)
						temp_preds.append(predict)
					predict = st.mode(temp_preds)
					preds.append(predict)

			y = torch.tensor(y, device=self.config['device'])
			preds = torch.tensor(preds, device=self.config['device'])
			print('Resnet Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
	
	def save(self, epoch):
		checkpoint_path = os.path.join(self.config["modeldir"], 'model-%d.ckpt'%(epoch))
		os.makedirs(self.config["modeldir"], exist_ok=True)
		torch.save(self.network.state_dict(), checkpoint_path)
		print("Checkpoint has been created.")
	
	def load(self, checkpoint_name):
		ckpt = torch.load(checkpoint_name, map_location="cpu")
		self.network.load_state_dict(ckpt, strict=True)
		print("Restored model parameters from {}".format(checkpoint_name))

	def predict_prob(self, x, checkpoint_num = None, batch_size = 32):
		if checkpoint_num:
			checkpointfile = os.path.join(self.config["modeldir"], 'model-%d.ckpt' % (checkpoint_num))
			self.load(checkpointfile)
		self.network.eval()
		ys_generated = []
		with torch.no_grad():
			inputs = np.array([parse_record(i, training=False) for i in x])
			inputs = torch.split(torch.tensor(inputs, device=self.config['device'], dtype=torch.float), batch_size)
			for i in tqdm(range(len(inputs))):
				ys_generated.append(self.network(inputs[i]))
			ys_generated = torch.cat(ys_generated)
			ys_generated = nn.functional.softmax(ys_generated, dim=1)

		return ys_generated

### END CODE HERE