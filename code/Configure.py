# Below configures are examples,
# you can modify them as you wish.

## YOUR CODE HERE

#DenseNet
model_configs = {
    "name": 'MyModel',
    "save_models_dir": '../saved_models/densenet/',
        "optimizer": {
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "momentum": 0.9,
    },
    "lr_scheduler": {
        "step_size": 10,
        "gamma": 0.1
    },
    "device": "cpu",
    "growth_rate": 32,
    "num_blocks": [6, 12, 24, 16],
    "reduction": 0.5,
    "drop_rate": 0.2,
    "num_classes": 10
}

#DenseNet
training_configs = {
    "batch_size": 256,
    "num_epochs": 200,
    "checkpoint_interval": 10,
    "blurring": False
}

#ResNet
resnet_configs = {
    "device": "cpu",
    "resnet_version" : 1,
    "num_epochs" : 200,
    "resnet_size": 18,
    "batch_size":128,
    "num_classes":10,
    "save_interval":10,
    "first_num_filters":16,
    "weight_decay": 0.0002,
    "modeldir" : '../saved_models/resnet/',
}

### END CODE HERE

