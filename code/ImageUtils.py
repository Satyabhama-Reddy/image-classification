import numpy as np
from matplotlib import pyplot as plt
import cv2

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training, blur=False):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE

    ### END CODE HERE
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training, blur)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training, blur=False):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        # Resize the image to add four extra pixels on each side.
        image = np.pad(image, [(4,), (4,), (0,)], mode='constant')

        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        image = image[x:x + 32, y:y + 32, :]

        # Randomly flip the image horizontally.
        if np.random.randint(0, 2):
            image = np.fliplr(image)

        # Experimented blurring for first 25% of the epochs
        if blur:
            image = cv2.GaussianBlur(image, (3, 3), 2)

    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image)
    sd = np.std(image)
    image = (image - mean) / (sd + 1e-05)  # to avoid division by 0
    ### END CODE HERE

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # image = image.reshape((32, 32, 3))
    depth_major = image.reshape((3, 32, 32))
    image = np.transpose(depth_major, [1, 2, 0])
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

def parse_record_test(record, training):
    """Parse a record to an image and perform data preprocessing to generate multiple versions of the image.
    Used in test time augmentation in ResNet. Explored this approach, but dropped eventually.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: list of arrays of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    ### END CODE HERE

    resized_img = np.pad(image, ((1, 1), (1, 1), (0, 0)), constant_values=(0))

    images = []
    for i in range(3):
        for j in range(3):
            image = resized_img[i:i + 32, j:j + 32, ]
            image = (image - np.mean(image)) / np.std(image)
            image = np.transpose(image, [2, 0, 1])
            images.append(image)

    # Other explored augmentations
    # blurred = cv2.GaussianBlur(images[5], (3, 3), 2)
    # blurred = (blurred - np.mean(blurred)) / np.std(blurred)
    # intensity_mat = np.ones(images[5].shape)*60
    # bright = cv2.add(images[5], intensity_mat)
    # dark = cv2.subtract(images[5], intensity_mat)
    
    # x = np.random.randint(0, 32)
    # y = np.random.randint(0, 32)
    # bicubic_img = cv2.resize(image, None, fy=2, fz=2, interpolation=cv2.INTER_CUBIC)

    # images = np.array([images[5], blurred, bright, dark])

    return images