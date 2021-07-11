import os
import zipfile
from PIL import Image
import numpy as np

base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()



image = np.array(Image.open(os.path.join(train_cats_dir, train_cat_fnames[1]) ))
imagetest = np.zeros((1024, 1024, 3))
imagetest[:image.shape[0], :image.shape[1], :] = image
print(imagetest)




#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
## All images will be rescaled by 1./255
#train_datagen = ImageDataGenerator(rescale=1./255)
#val_datagen = ImageDataGenerator(rescale=1./255)
#
## Flow training images in batches of 20 using train_datagen generator
#train_generator = train_datagen.flow_from_directory(
#        train_dir,  # This is the source directory for training images
#        target_size=(150, 150),  # All images will be resized to 150x150
#        batch_size=20,
#        # Since we use binary_crossentropy loss, we need binary labels
#        class_mode='binary')
#
## Flow validation images in batches of 20 using val_datagen generator
#validation_generator = val_datagen.flow_from_directory(
#        validation_dir,
#        target_size=(150, 150),
#        batch_size=20,
#        class_mode='binary')
#
#print(train_generator)





##import matplotlib.pyplot as plt
##import matplotlib.image as mpimg
### Parameters for our graph; we'll output images in a 4x4 configuration
##nrows = 4
##ncols = 4
##
### Index for iterating over images
##pic_index = 0
##
##fig = plt.gcf()
##fig.set_size_inches(ncols * 4, nrows * 4)
##
##pic_index += 8
##next_cat_pix = [os.path.join(train_cats_dir, fname) 
##                for fname in train_cat_fnames[pic_index-8:pic_index]]
##next_dog_pix = [os.path.join(train_dogs_dir, fname) 
##                for fname in train_dog_fnames[pic_index-8:pic_index]]
##
##for i, img_path in enumerate(next_cat_pix+next_dog_pix):
##  # Set up subplot; subplot indices start at 1
##  sp = plt.subplot(nrows, ncols, i + 1)
##  sp.axis('Off') # Don't show axes (or gridlines)
##
##  img = mpimg.imread(img_path)
##  plt.imshow(img)
##
##plt.show()
#
#from tensorflow.keras import layers
#from tensorflow.keras import Model
#
## Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
## the three color channels: R, G, and B
#img_input = layers.Input(shape=(150, 150, 3))
#
## First convolution extracts 16 filters that are 3x3
## Convolution is followed by max-pooling layer with a 2x2 window
#x = layers.Conv2D(16, 3, activation='relu')(img_input)
#x = layers.MaxPooling2D(2)(x)
#
## Second convolution extracts 32 filters that are 3x3
## Convolution is followed by max-pooling layer with a 2x2 window
#x = layers.Conv2D(32, 3, activation='relu')(x)
#x = layers.MaxPooling2D(2)(x)
#
## Third convolution extracts 64 filters that are 3x3
## Convolution is followed by max-pooling layer with a 2x2 window
#x = layers.Conv2D(64, 3, activation='relu')(x)
#x = layers.MaxPooling2D(2)(x)
#
#
## Flatten feature map to a 1-dim tensor so we can add fully connected layers
#x = layers.Flatten()(x)
#
## Create a fully connected layer with ReLU activation and 512 hidden units
#x = layers.Dense(512, activation='relu')(x)
#
## Create output layer with a single node and sigmoid activation
#output = layers.Dense(1, activation='sigmoid')(x)
#
## Create model:
## input = input feature map
## output = input feature map + stacked convolution/maxpooling layers + fully 
## connected layer + sigmoid output layer
#model = Model(img_input, output)
#
#model.summary()
#
#
#from tensorflow.keras.optimizers import RMSprop
#
#model.compile(loss='binary_crossentropy',
#              optimizer=RMSprop(lr=0.001),
#              metrics=['acc'])
#

#import numpy as np
#import random
#from tensorflow.keras.preprocessing.image import img_to_array, load_img
#
## Let's define a new Model that will take an image as input, and will output
## intermediate representations for all layers in the previous model after
## the first.
#successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
#
## Let's prepare a random input image of a cat or dog from the training set.
#cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
#dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
#img_path = random.choice(cat_img_files + dog_img_files)
#
#img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
#x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
#x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
#
## Rescale by 1/255
#x /= 255
#
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





