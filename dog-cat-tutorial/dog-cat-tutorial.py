from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from matplotlib import pyplot
from numpy import loadtxt 

def VGG16():
    model = Sequential()

    model.add(Conv2D(input_shape=(150, 150,3),filters = 64,kernel_size = 3,padding = 'same', activation= 'relu')) #the ouput will have the same size of the input (padding = same and stride =1)
    model.add(Conv2D(filters = 64,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Conv2D(filters = 128,kernel_size = 3,padding = 'same', activation= 'relu'))
    model.add(Conv2D(filters = 128,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Conv2D(filters = 256,kernel_size = 3,padding = 'same', activation= 'relu'))
    model.add(Conv2D(filters = 256,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(Conv2D(filters = 256,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Conv2D(filters = 512,kernel_size = 3,padding = 'same', activation= 'relu'))
    model.add(Conv2D(filters = 512,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(Conv2D(filters = 512,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Conv2D(filters = 512,kernel_size = 3,padding = 'same', activation= 'relu'))
    model.add(Conv2D(filters = 512,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(Conv2D(filters = 512,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dense(4096, activation = 'relu'))
    #model.add(Dense(1, activation = 'softmax'))
    model.add(Dense(1, activation = 'sigmoid'))

    
    #to train the model we are going to use a binary cross entropy loss function, and gradient descen (adam)
    #model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

    from tensorflow.keras.optimizers import Adam
    opt = Adam(lr=0.001)
    model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def simpler_VGG16():
    model = Sequential()

    model.add(Conv2D(input_shape=(150, 150,3),filters = 64,kernel_size = 3,padding = 'same', activation= 'relu')) #the ouput will have the same size of the input (padding = same and stride =1)
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))
   
    model.add(Conv2D(filters = 128,kernel_size = 3,padding = 'same', activation= 'relu'))
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Conv2D(filters = 256,kernel_size = 3,padding = 'same', activation= 'relu'))
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))
    
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation = 'sigmoid'))


    from tensorflow.keras.optimizers import Adam
    opt = Adam(lr=0.001)
    model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def simplerv2_VGG16():
    model = Sequential()

    model.add(Conv2D(input_shape=(150, 150,3),filters = 32,kernel_size = 3,padding = 'same', activation= 'relu')) 
    model.add(MaxPool2D(pool_size = 2,strides = 2,padding = 'valid'))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation = 'sigmoid'))


    from tensorflow.keras.optimizers import Adam
    opt = Adam(lr=0.001)
    model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def tutorial_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

    return model


dataset = '/datasets/dogVscats_xs'
train_data_dir = dataset+'/train'
validation_data_dir = dataset+'/validate'


modelName = "/models/cat-dog-model-simpleVGG16-v2_model.h5"
lossPlotImgPath = "/models/cat-dog-model-simpleVGG16-v2_model.png"

# dimensions of our images.
img_width, img_height = 150, 150

epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = simplerv2_VGG16()
model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

model.save_weights(modelName)

pyplot.title('training')
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['accuracy'], label='train_acc')
pyplot.plot(history.history['val_accuracy'], label='val_acc')
pyplot.plot(history.history['val_loss'], label='val_los')
pyplot.legend()
pyplot.savefig(lossPlotImgPath)