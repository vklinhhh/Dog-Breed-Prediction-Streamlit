# Load libraries

import cv2, random, time, shutil, csv
import os
import numpy as np
import pickle
import copy
import keras
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models,utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils

# keras libraries

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.utils import to_categorical
from keras import backend as K

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input


#####

current_path = os.getcwd()

# FEATURE EXTRACTION OF VALIDAION AND TESTING ARRAYS
def get_valfeatures(model_name, data_preprocessor, data):
    '''
    Same as above except not image augmentations applied.
    Used for feature extraction of validation and testing.
    '''

    dataset = tf.data.Dataset.from_tensor_slices(data)

    ds = dataset.batch(64)

    input_size = data.shape[1:]
    #Prepare pipeline.
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)

    base_model = model_name(weights='imagenet', include_top=False,
                                input_shape=input_size)(preprocessor)

    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    #Extract feature.
    feature_maps = feature_extractor.predict(ds, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps

# RETURNING CONCATENATED FEATURES USING MODELS AND PREPROCESSORS
def get_concat_features(feat_func, models, preprocs, array):

    print(f"Beggining extraction with {feat_func.__name__}\n")
    feats_list = []

    for i in range(len(models)):
        
        print(f"\nStarting feature extraction with {models[i].__name__} using {preprocs[i].__name__}\n")
        # applying the above function and storing in list
        feats_list.append(feat_func(models[i], preprocs[i], array))

    # features concatenating
    final_feats = np.concatenate(feats_list, axis=-1)

    return final_feats

# DEFINING models and preprocessors imports 
inception_preprocessor = preprocess_input
xception_preprocessor = preprocess_input
nasnet_preprocessor = preprocess_input
inc_resnet_preprocessor = preprocess_input
modellist = [InceptionV3,  InceptionResNetV2, Xception, NASNetLarge]
preprocs = [inception_preprocessor,  inc_resnet_preprocessor, 
            xception_preprocessor, nasnet_preprocessor]

# Load the list of model names
with open(os.path.join(current_path, 'static/model_names.txt'), 'r') as file:
    model_names = file.read().splitlines()

# Load the models
trained_models = []
for model_name in model_names:
    model_path = os.path.join(current_path, 'static/models', model_name)
    model = load_model(model_path)
    trained_models.append(model)
    
# Load class_names from file
with open(os.path.join(current_path,'static/class_names.pkl'), 'rb') as f:
    class_names = pickle.load(f)


def predictor(image_path):
    X = []
    orig_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    res_image = cv2.resize(orig_image, (331, 331))
    X.append(res_image)
    X_test = np.array(X)
    test_features = get_concat_features(get_valfeatures, modellist, preprocs, X_test)
    
    y_pred_norm = trained_models[0].predict(test_features, batch_size=128) / 3
    for dnn in trained_models[1:]:
        y_pred_norm += dnn.predict(test_features, batch_size=128) / 3
    predicted_probs = y_pred_norm[0]

    # Get the predicted breed label
    predicted_label_index = np.argmax(predicted_probs)
    predicted_label = class_names[predicted_label_index]

    return predicted_label