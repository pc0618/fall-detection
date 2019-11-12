#reference https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow

from __future__ import print_function
from numpy.random import seed
seed(1)
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import scipy.io as sio
import cv2
import glob
import gc

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
		 	  Activation, Dense, Dropout, ZeroPadding2D)
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.layers.advanced_activations import ELU

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# CHANGE THESE VARIABLES ---
data_folder = '../URFD_opticalflow_test_4/'
mean_file = '../flow_mean.mat'
vgg_16_weights = '../weights.h5'
save_features = True
save_plots = True

# Set to 'True' if you want to restore a previous trained models
# Training is skipped and test is done
use_checkpoint = False # Set to True or False
# --------------------------

best_model_path = 'models/'
plots_folder = 'plots/'
checkpoint_path = 'models/fold_'

features_file = '5_test_features_urfd_tf.h5'
labels_file = '5_test_labels_urfd_tf.h5'
features_key = 'features'
labels_key = 'labels'

L = 10
num_features = 4096

        
def plot_training_info(case, metrics, save, history):
    '''
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png' 
	will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    '''
    plt.ioff()
    if 'accuracy' in metrics:     
        fig = plt.figure()
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save == True:
            plt.savefig(case + 'accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # summarize history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'val'], loc='upper left')
        if save == True:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)
 
def generator(list1, lits2):
    '''
    Auxiliar generator: returns the ith element of both given list with
	 each call to next() 
    '''
    for x,y in zip(list1,lits2):
        yield x, y
          
def saveFeatures(feature_extractor,
		 features_file,
		 labels_file,
		 features_key, 
		 labels_key):
    '''
    Function to load the optical flow stacks, do a feed-forward through the
	 feature extractor (VGG16) and
    store the output feature vectors in the file 'features_file' and the 
	labels in 'labels_file'.
    Input:
    * feature_extractor: model VGG16 until the fc6 layer.
    * features_file: path to the hdf5 file where the extracted features are
	 going to be stored
    * labels_file: path to the hdf5 file where the labels of the features
	 are going to be stored
    * features_key: name of the key for the hdf5 file to store the features
    * labels_key: name of the key for the hdf5 file to store the labels
    '''
    
    class0 = 'Falls'
    class1 = 'NotFalls'     

    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']

    # Fill the folders and classes arrays with all the paths to the data
    folders, classes = [], []
    fall_videos = [f for f in os.listdir(data_folder + class0) 
			if os.path.isdir(os.path.join(data_folder + class0, f))]
    fall_videos.sort()
    for fall_video in fall_videos:
        x_images = glob.glob(data_folder + class0 + '/' + fall_video
				 + '/flow_x*.jpg')
        if int(len(x_images)) >= 10:
            folders.append(data_folder + class0 + '/' + fall_video)
            classes.append(0)

    not_fall_videos = [f for f in os.listdir(data_folder + class1) 
			if os.path.isdir(os.path.join(data_folder + class1, f))]
    not_fall_videos.sort()
    for not_fall_video in not_fall_videos:
        x_images = glob.glob(data_folder + class1 + '/' + not_fall_video
				 + '/flow_x*.jpg')
        if int(len(x_images)) >= 10:
            folders.append(data_folder + class1 + '/' + not_fall_video)
            classes.append(1)

    # Total amount of stacks, with sliding window = num_images-L+1
    nb_total_stacks = 0
    for folder in folders:
        x_images = glob.glob(folder + '/flow_x*.jpg')
        nb_total_stacks += len(x_images)-L+1
    
    # File to store the extracted features and datasets to store them
    # IMPORTANT NOTE: 'w' mode totally erases previous data
    h5features = h5py.File(features_file,'w')
    h5labels = h5py.File(labels_file,'w')
    dataset_features = h5features.create_dataset(features_key,
			 shape=(nb_total_stacks, num_features),
			 dtype='float64')
    dataset_labels = h5labels.create_dataset(labels_key,
			 shape=(nb_total_stacks, 1),
			 dtype='float64')  
    cont = 0
    
    for folder, label in zip(folders, classes):
        x_images = glob.glob(folder + '/flow_x*.jpg')
        x_images.sort()
        y_images = glob.glob(folder + '/flow_y*.jpg')
        y_images.sort()
        nb_stacks = len(x_images)-L+1
        # Here nb_stacks optical flow stacks will be stored
        flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
        gen = generator(x_images,y_images)
        for i in range(len(x_images)):
            flow_x_file, flow_y_file = next(gen)
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            # Assign an image i to the jth stack in the kth position, but also
	    # in the j+1th stack in the k+1th position and so on	
	    # (for sliding window) 
            for s in list(reversed(range(min(10,i+1)))):
                if i-s < nb_stacks:
                    flow[:,:,2*s,  i-s] = img_x
                    flow[:,:,2*s+1,i-s] = img_y
            del img_x,img_y
            gc.collect()
            
        # Subtract mean
        flow = flow - np.tile(flow_mean[...,np.newaxis],
			      (1, 1, 1, flow.shape[3]))
        flow = np.transpose(flow, (3, 0, 1, 2)) 
        predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
        truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
        # Process each stack: do the feed-forward pass and store
	# in the hdf5 file the output
        for i in range(flow.shape[0]):
            prediction = feature_extractor.predict(
					np.expand_dims(flow[i, ...],0))
            predictions[i, ...] = prediction
            truth[i] = label
        dataset_features[cont:cont+flow.shape[0],:] = predictions
        dataset_labels[cont:cont+flow.shape[0],:] = truth
        cont += flow.shape[0]
    h5features.close()
    h5labels.close()
    
def test_video(feature_extractor, video_path, ground_truth):
    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']
    
    x_images = glob.glob(video_path + '/flow_x*.jpg')
    x_images.sort()
    y_images = glob.glob(video_path + '/flow_y*.jpg')
    y_images.sort()
    nb_stacks = len(x_images)-L+1
    # Here nb_stacks optical flow stacks will be stored
    flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
    gen = generator(x_images,y_images)
    for i in range(len(x_images)):
        flow_x_file, flow_y_file = gen.next()
        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
        # Assign an image i to the jth stack in the kth position, but also
	# in the j+1th stack in the k+1th position and so on
	# (for sliding window) 
        for s in list(reversed(range(min(10,i+1)))):
            if i-s < nb_stacks:
                flow[:,:,2*s,  i-s] = img_x
                flow[:,:,2*s+1,i-s] = img_y
        del img_x,img_y
        gc.collect()
    flow = flow - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
    flow = np.transpose(flow, (3, 0, 1, 2)) 
    predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
    truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
    # Process each stack: do the feed-forward pass
    for i in range(flow.shape[0]):
        prediction = feature_extractor.predict(np.expand_dims(flow[i, ...],0))
        predictions[i, ...] = prediction
        truth[i] = ground_truth
    return predictions, truth
            
def main():
    # ========================================================================
    # VGG-16 ARCHITECTURE
    # ========================================================================
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(num_features, name='fc6', kernel_initializer='glorot_uniform'))
    
    # ========================================================================
    # WEIGHT INITIALIZATION
    # ========================================================================
    layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
		   'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
		   'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    h5 = h5py.File(vgg_16_weights, 'r')
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Copy the weights stored in the 'vgg_16_weights' file to the
    # feature extractor part of the VGG16
    for layer in layerscaffe[:-3]:
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        #w2 = np.transpose(np.asarray(w2), (0,1,2,3))
        #w2 = w2[:, :, ::-1, ::-1]
        w2 = np.transpose(np.asarray(w2), (2,3,1,0))
        w2 = w2[::-1, ::-1, :, :]
        b2 = np.asarray(b2)
        #layer_dict[layer].W.set_value(w2)
        #layer_dict[layer].b.set_value(b2)
        layer_dict[layer].set_weights((w2, b2))
    #sys.exit()
    # Copy the weights of the first fully-connected layer (fc6)
    layer = layerscaffe[-3]
    w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
    w2 = np.transpose(np.asarray(w2), (1,0))
    b2 = np.asarray(b2)
    #layer_dict[layer].W.set_value(w2)
    #layer_dict[layer].b.set_value(b2)
    layer_dict[layer].set_weights((w2, b2))

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    if save_features:
        saveFeatures(model, features_file,
		    labels_file, features_key,
		    labels_key)
        
if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    main()
