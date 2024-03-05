Steps to run the code:

All the configurations are set in Configure.py. Please set the "device" parameter in both densenet and resnet configurations based on the device being used(cpu/cuda). Default is cpu.

cd code;

To train and evaluate the models, run:
python3 main.py train <path to cifar-10-batches-py>

To test the models, on cifar-10 test images, run :
python3 main.py test <path to cifar-10-batches-py>

To predict the probabilities on private test set, run :
python3 main.py predict <path to directory containing the test file private_test_images_2022.npy>

The predictions generated are based on the ensembled model.
main.py can be modified to experiment with ResNet and DenseNet models individually.

Note: The batch size for testing is 32 by default. Can be updated in predict_prob fuunctions if needed.

Modules used:
pytorch - 1.9.0
numpy - 1.23.4
opencv-python - 4.6.0.66    #Used for augmentations. Can be commented out in ImageUtils.py if causing issues.