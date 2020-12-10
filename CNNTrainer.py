import sys, os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
base_path = './drive/MyDrive/Aspardisid'

K.clear_session()
training_data = base_path + '/Images/Training'
validation_data = base_path + '/Images/Validation'
weight_model_path = base_path + '/Model/weights.h5'
model_path = base_path + '/Model/model.h5'

epochs = 50
width, height = 256, 256
batch_size = 8
steps = 248
validation_steps = 64
conv_filters1 = 32
conv_filters2 = 64
filter_size1 = (3, 3)
filter_size2 = (2, 2)
pool_size = (2, 2)
class_amount = 2
learning_rate = 0.0004

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

training_generator = training_datagen.flow_from_directory(
    training_data,
    target_size = (height, width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data,
    target_size = (height, width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

cnn = Sequential()

cnn.add(Convolution2D(conv_filters1, filter_size1, padding = 'same', input_shape = (width, height, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = pool_size))

cnn.add(Convolution2D(conv_filters2, filter_size2, padding = 'same'))
cnn.add(MaxPooling2D(pool_size = pool_size))

cnn.add(Flatten())
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.2))
cnn.add(Dense(class_amount, activation = 'softmax'))

cnn.compile(
    loss = 'categorical_crossentropy',
    optimizer = Adam(lr = learning_rate),
    metrics = ['accuracy']
)

cnn.fit(
    training_generator,
    steps_per_epoch = int(steps / batch_size),
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = int(validation_steps / batch_size)
)

cnn.save(model_path)
cnn.save_weights(weight_model_path)