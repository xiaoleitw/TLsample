# -*- coding: utf-8 -*-
# create by Xiaolei
# @ 6/26/18
import sys
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.utils import plot_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from plotlog import plot_history
# keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


PAR_DIR="./sample/"
#here we define some parameters, i.e. path
img_height,img_width = 224, 224
img_size = (img_width, img_height )
batch_size = 16
num_train = 200
num_valid = 200
num_test = 100
num_epoch = 50
FREEZE_LAYERS = 5
CLASS_INDEX = None

# check keras backend: tensorflow or theano.
print('backend is '+str( K.backend()))


# here we prepare some optimizers for future use
myrmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
myadam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
mysgd = keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=False)
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

if K.image_data_format() == 'channels_first':
    input_img_shape = (3, img_width, img_height)
else:
    input_img_shape = (img_width, img_height,3)



def try_image_augmentation(testimg='train/cats/cat.111.jpg'):  # visualize some image augmentation, check this as True when you want
    img = load_img(PAR_DIR+testimg)  # PIL image
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=PAR_DIR+'preview', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # we do only want 20 augmented samples

    return i



def preprocess_input_local(x): # before predict some image ...
    dim_ordering = 'tf' if K.backend() == 'tensorflow' else 'th'

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def TrainedModel_run():

    model = ResNet50(weights='imagenet')
    img_path = PAR_DIR + 'preview/cat_0_3538.jpeg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=5)[0])



def simple_cnn_model(): # a simple model in keras

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_img_shape))
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

    model.compile(loss='binary_crossentropy',
                  optimizer=myrmsprop,
                  metrics=['accuracy'])

    return model



def train_simple_cnn(): # train a model, calling inside

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    # kindly notice that rescale is a must
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
            PAR_DIR+'train',  # this is the target directory
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            PAR_DIR+'valid',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary')

    model = simple_cnn_model()
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_valid // batch_size)

    model.save('train_simple_cnn.h5')  # always save
    return history, model



def train_model(model, MODEL_WEIGHTS_FILE='trans_vgg16_m1.h5'):
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        PAR_DIR + 'train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        PAR_DIR + 'valid',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=10),
                 CSVLogger(MODEL_WEIGHTS_FILE+'.csv', separator=',', append=False),
                 EarlyStopping(monitor='val_loss', min_delta=0, patience=7)]

    # fine-tune the model
    history=model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        callbacks=callbacks,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_valid // batch_size)

    model.save('trans_vgg16.h5')

    return history, model



def train_from_file(feature_train_file='resnet_features_train.npy', feature_valid_file='resnet_features_validation.npy'):
    train_data = np.load(open(feature_train_file))
    # the features were saved in order, so recreating the labels is easy
    train_labels = np.array([0] * int(num_train/2) + [1] * int(num_train/2))

    validation_data = np.load(open(feature_valid_file))
    validation_labels = np.array([0] * int(num_valid/2) + [1] * int(num_valid/2))
    print("Train Feature Shape: ", train_data.shape)
    print("Test Feature Shape: ", validation_data.shape)


    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=myrmsprop,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    history=model.fit(train_data, train_labels,
              epochs=num_epoch,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights('transfer_fc_model.h5')
    return history, model


def build_top(model, top_model_weights_path):
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)
    return model



def make_base_model(model):
    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=mysgd,
                  metrics=['accuracy'])
    return model


def make_resnet_features():
    generator = datagen.flow_from_directory(
            PAR_DIR+'train',
            target_size=img_size,
            batch_size=batch_size,
            class_mode=None,  # this means our generator will only yield batches of data, no labels
            shuffle=False)
    resnet_model = ResNet50(include_top=False, weights='imagenet')
    print('ResNet Model Loaded ... ')
    bottleneck_features_train = resnet_model.predict_generator(generator, num_train)
    # save the output as a Numpy array
    # this part may take hours
    np.save(open('resnet_features_train.npy', 'wb'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            PAR_DIR+'valid',
            target_size=img_size,
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    # this part may take hours
    bottleneck_features_validation = resnet_model.predict_generator(generator, num_valid)
    np.save(open('resnet_features_validation.npy', 'wb'), bottleneck_features_validation)
    print("Feature Got from ResNet ...")
    return True


def make_vgg16_tuned_model():
    from keras.layers import Input, Flatten, Dense
    from keras.models import Model

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    LAYER_TRAIN = 0
    LAYER_NOTRAIN = 0
    for layer in model_vgg16_conv.layers[:FREEZE_LAYERS]:
        layer.trainable = False
        LAYER_NOTRAIN += 1
    for layer in model_vgg16_conv.layers[FREEZE_LAYERS:]:
        layer.trainable = True
        LAYER_TRAIN += 1

    print('Number of trainable Layers: '+ str(LAYER_TRAIN))
    print('Number of non-trainable Layers: ' + str(LAYER_NOTRAIN) )


    input = Input(shape=input_img_shape)
    output_vgg16_conv = model_vgg16_conv(input)
    x = Flatten()(output_vgg16_conv)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input=input, output=x)

    model.compile(loss='binary_crossentropy',
                  optimizer=myrmsprop,
                  metrics=['accuracy'])

    print(model.summary())
    return model

def make_vgg16_model():
    from keras.layers import Input, Flatten, Dense
    from keras.models import Model

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=input_img_shape)
    output_vgg16_conv = model_vgg16_conv(input)
    x = Flatten()(output_vgg16_conv)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input=input, output=x)

    model.compile(loss='binary_crossentropy',
                  optimizer=myrmsprop,
                  metrics=['accuracy'])

    print(model.summary())
    return model


if __name__ == '__main__':
    step_to_run = int(sys.argv[1])
    
    # Step 0:
    if step_to_run == 1:
        try_image_augmentation()
        
    # Step 1:
    if step_to_run == 2:
        history0, model0= train_simple_cnn()
        plot_history(history0, model0, 'CNN-sample-history.png')
        plot_model(model0, to_file='CNN-model.png', show_shapes=True)

    # Step 2:
    if step_to_run == 3:
        model1 = make_vgg16_model()
        history1, model1 = train_model(model1)
        plot_history(history1, model1, 'VGG-Trans-sample-history.png')
        plot_model(model1, to_file='VGG-Trans-model.png', show_shapes=True)


    if step_to_run == 4:
        from keras.models import load_model
        temp_model = load_model('trans_vgg16.h5')
        img_path = PAR_DIR+'/preview/cat_0_3116.jpeg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print('Input image shape:', x.shape)
        preds = temp_model.predict(x)
        print('Predicted:', preds)

