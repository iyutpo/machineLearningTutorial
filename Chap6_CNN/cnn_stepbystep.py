import os
import zipfile
from PIL import Image
import numpy as np

base_dir = 'tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()



imagetest = np.zeros((2, 1024, 1024, 3))
for i in range(2):
  image = np.array(Image.open(os.path.join(train_cats_dir, train_cat_fnames[i]) )) / 255.0
#  imagetest[i, int((imagetest.shape[0] - image.shape[0]) / 2):image.shape[0] + int((imagetest.shape[0] - image.shape[0]) / 2), int((imagetest.shape[1] - image.shape[1]) / 2) :image.shape[1] + int((imagetest.shape[1] - image.shape[1]) / 2), :] = image
  imagetest[i, :image.shape[0], :image.shape[1], :] = image

from cnnfunc_stepbystep import *
## First convolution extracts 16 filters that are 3x3
## Convolution is followed by max-pooling layer with a 2x2 window
W1 = np.ones((3, 3, 3, 16))
b1 = np.ones((1, 1, 1, 16))
hparameters = {'stride': 1, 'pad': 0}
Z1, cache11 = conv_forward(imagetest, W1, b1, hparameters)

hparameters = {'stride': 1, 'f': 2}
A1, cache12 = pool_forward(Z1, hparameters, mode = "max")

## Second convolution extracts 32 filters that are 3x3
## Convolution is followed by max-pooling layer with a 2x2 window
hparameters = {'stride': 1, 'pad': 0}
W2 = np.ones((3, 3, 16, 32))
b2 = np.ones((1, 1, 1, 32))
Z2, cache21 = conv_forward(A1, W2, b2, hparameters)
hparameters = {'stride': 1, 'f': 2}
A2, cache22 = pool_forward(Z2, hparameters, mode = "max")

## Third convolution extracts 64 filters that are 3x3
## Convolution is followed by max-pooling layer with a 2x2 window
hparameters = {'stride': 1, 'pad': 0}
W3 = np.ones((3, 3, 32, 64))
b3 = np.ones((1, 1, 1, 64))
Z3, cache31 = conv_forward(A2, W3, b3, hparameters)
hparameters = {'stride': 1, 'f': 2}
A3, cache32 = pool_forward(Z3, hparameters, mode = "max")


## Flatten feature map to a 1-dim tensor so we can add fully connected layers
## Create a fully connected layer with ReLU activation and 512 hidden units

## Create output layer with a single node and sigmoid activation



## Let's run our image through our network, thus obtaining all
## intermediate representations for this image.
#successive_feature_maps = visualization_model.predict(x)
#
## These are the names of the layers, so can have them as part of our plot
#layer_names = [layer.name for layer in model.layers]
#
## Now let's display our representations
#for layer_name, feature_map in zip(layer_names, successive_feature_maps):
#  if len(feature_map.shape) == 4:
#    # Just do this for the conv / maxpool layers, not the fully-connected layers
#    n_features = feature_map.shape[-1]  # number of features in feature map
#    # The feature map has shape (1, size, size, n_features)
#    size = feature_map.shape[1]
#    # We will tile our images in this matrix
#    display_grid = np.zeros((size, size * n_features))
#    for i in range(n_features):
#      # Postprocess the feature to make it visually palatable
#      x = feature_map[0, :, :, i]
#      x -= x.mean()
#      x /= (1 + x.std())
#      x *= 64
#      x += 128
#      x = np.clip(x, 0, 255).astype('uint8')
#      # We'll tile each filter into this big horizontal grid
#      display_grid[:, i * size : (i + 1) * size] = x
#    # Display the grid
#    scale = 20. / n_features
#    plt.figure(figsize=(scale * n_features, scale))
#    plt.title(layer_name)
#    plt.grid(False)
#    plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
#plt.show()
#
#
## Retrieve a list of accuracy results on training and validation data
## sets for each training epoch
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#
## Retrieve a list of list results on training and validation data
## sets for each training epoch
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
## Get number of epochs
#epochs = range(len(acc))
#
## Plot training and validation accuracy per epoch
#plt.plot(epochs, acc)
#plt.plot(epochs, val_acc)
#plt.title('Training and validation accuracy')
#
#plt.figure()
#
## Plot training and validation loss per epoch
#plt.plot(epochs, loss)
#plt.plot(epochs, val_loss)
#plt.title('Training and validation loss')
#plt.show()





