### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from ResnetModel import Cifar
from Ensembled import Ensembled
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs, resnet_configs
from ImageUtils import visualize


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", default="../cifar-10-batches-py/", help="path to the data")
parser.add_argument("--save_dir", default="../predictions",help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	resnet = Cifar(resnet_configs)
	densenet = MyModel(model_configs)
	ensembled = Ensembled(resnet, densenet)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

		densenet.train(x_train_new, y_train_new, training_configs, x_valid, y_valid)
		densenet.evaluate(x_test, y_test, [training_configs["num_epochs"]])
		resnet.train(x_train_new, y_train_new, resnet_configs["num_epochs"])
		resnet.evaluate(x_test, y_test, [resnet_configs["num_epochs"]])

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)

		# resnet.test_or_validate_tta(x_test, y_test, [200])	#Performs test time augmentations
		densenet.evaluate(x_test, y_test, [200])
		resnet.evaluate(x_test, y_test, [200])
		ensembled.evaluate(x_test, y_test, 200, 200)

	elif args.mode == 'predict':
		# Loading private testing dataset
		x_test = load_testing_images(args.data_dir)
		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset 

		# predictions = resnet.predict_prob(x_test, 200).numpy()
		# predictions = densenet.predict_prob(x_test, 200).numpy()

		predictions = ensembled.predict_prob(x_test, 200, 200).numpy()
		np.save(args.save_dir, predictions)
		print("Prediction file created at", args.save_dir)
		

### END CODE HERE

