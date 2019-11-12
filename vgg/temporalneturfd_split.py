#reference: https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow

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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# CHANGE THESE VARIABLES ---
vgg_16_weights = '../weights.h5'
save_plots = True

# Set to 'True' if you want to restore a previous trained models
# Training is skipped and test is done
use_checkpoint = False # Set to True or False
# --------------------------

best_model_path = 'models/'
plots_folder = 'plots/'
checkpoint_path = 'models/fold_'

train_features_file = 'train_features_urfd_tf.h5'
train_labels_file = 'train_labels_urfd_tf.h5'
test_features_file = 'test_features_urfd_tf.h5'
test_labels_file = 'test_labels_urfd_tf.h5'
features_key = 'features'
labels_key = 'labels'

L = 10
num_features = 4096
batch_norm = True
learning_rate = 0.0001
mini_batch_size = 0
weight_0 = 0.08
epochs = 1000

# Name of the experiment
exp = 'urfd_lr{}_batchs{}_batchnorm{}_w0_{}'.format(learning_rate,
					       mini_batch_size,
					       batch_norm,
					       weight_0)
        
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
    # TRAINING
    # ========================================================================    
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
		epsilon=1e-08)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
		  metrics=['accuracy'])
    do_training = True   
    compute_metrics = True
    threshold = 0.5
    
    if do_training:
        fold_number = 1
        test_h5features = h5py.File(test_features_file, 'r')
        test_h5labels = h5py.File(test_labels_file, 'r')
        train_h5features = h5py.File(train_features_file, 'r')
        train_h5labels = h5py.File(train_labels_file, 'r')
        
	# train/test sets

        X = np.array(train_h5features[features_key][:])
        _y = np.array(train_h5labels[labels_key][:])
        X2 = np.array(test_h5features[features_key][:])
        _y2 = np.array(test_h5labels[labels_key][:])

        # Create a validation subset from the training set
        val_size = 100
        zeroes = np.asarray(np.where(_y==0)[0])
        ones = np.asarray(np.where(_y==1)[0])
        
        zeroes.sort()
        ones.sort()
    
        trainval_split_0 = StratifiedShuffleSplit(n_splits=1,
    					   test_size=val_size//2,
    				 	   random_state=7)
        indices_0 = trainval_split_0.split(X[zeroes,...],
    				     np.argmax(_y[zeroes,...], 1))
        trainval_split_1 = StratifiedShuffleSplit(n_splits=1,
    					   test_size=val_size//2,
    				 	   random_state=7)
        indices_1 = trainval_split_1.split(X[ones,...],
    				     np.argmax(_y[ones,...], 1))
        train_indices_0, val_indices_0 = next(indices_0)
        train_indices_1, val_indices_1 = next(indices_1)
    
        X_train = np.concatenate([X[zeroes,...][train_indices_0,...],
    			      X[ones,...][train_indices_1,...]],axis=0)
        y_train = np.concatenate([_y[zeroes,...][train_indices_0,...],
    			      _y[ones,...][train_indices_1,...]],axis=0)
        X_val = np.concatenate([X[zeroes,...][val_indices_0,...],
    			      X[ones,...][val_indices_1,...]],axis=0)
        y_val = np.concatenate([_y[zeroes,...][val_indices_0,...],
    			      _y[ones,...][val_indices_1,...]],axis=0)
       

        # Balance the number of positive and negative samples so that
        # there is the same amount of each of them
        all0 = np.asarray(np.where(y_train==0)[0])
        all1 = np.asarray(np.where(y_train==1)[0])  
        
        if len(all0) < len(all1):
            all1 = np.random.choice(all1, len(all0), replace=False)
        else:
            all0 = np.random.choice(all0, len(all1), replace=False)
        allin = np.concatenate((all0.flatten(),all1.flatten()))
        allin.sort()
        X_train = X_train[allin,...]
        y_train = y_train[allin]
        
        all0 = np.asarray(np.where(y_train==0)[0])
        all1 = np.asarray(np.where(y_train==1)[0])
        
        # ==================== CLASSIFIER ========================
        extracted_features = Input(shape=(num_features,),
    			       dtype='float32', name='input')
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99,
    			       epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)
       
        x = Dropout(0.9)(x)
        x = Dense(4096, name='fc2', kernel_initializer='glorot_uniform')(x)
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(x)
        x = Dropout(0.8)(x)
        x = Dense(1, name='predictions',
    		kernel_initializer='glorot_uniform')(x)
        x = Activation('sigmoid')(x)
        
        classifier = Model(input=extracted_features,
    		       output=x, name='classifier')
        fold_best_model_path = best_model_path + 'urfd_fold_{}.h5'.format(
    							fold_number)
        classifier.compile(optimizer=adam, loss='binary_crossentropy',
    		       metrics=['accuracy'])
    
        if not use_checkpoint:
    	# ==================== TRAINING ========================     
    	# weighting of each class: only the fall class gets
    	# a different weight
            class_weight = {0: weight_0, 1: 1}
            
            # callback definition
            metric = 'val_loss'
            e = EarlyStopping(monitor=metric, min_delta=0, patience=100,
            		  mode='auto')
            c = ModelCheckpoint(fold_best_model_path, monitor=metric,
            		    save_best_only=True,
            		    save_weights_only=False, mode='auto')
            callbacks = [e, c]
            
            # Batch training
            if mini_batch_size == 0:
            	history = classifier.fit(X_train, y_train, 
            				validation_data=(X_val,y_val),
            				batch_size=X_train.shape[0],
            				nb_epoch=epochs,
            				shuffle='batch',
            				class_weight=class_weight,
            				callbacks=callbacks)
            else:
            	history = classifier.fit(X_train, y_train, 
            				validation_data=(X_val,y_val),
            				batch_size=mini_batch_size,
            				nb_epoch=epochs,
            				shuffle='batch',
            				class_weight=class_weight,
            				callbacks=callbacks)
            
            plot_training_info(plots_folder + exp, ['accuracy', 'loss'],
            		   save_plots, history.history)
            
            classifier = load_model(fold_best_model_path)
            
            # Use full training set (training+validation)
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)
            
            if mini_batch_size == 0:
            	history = classifier.fit(X_train, y_train, 
            				batch_size=X_train.shape[0],
            				nb_epoch=1,
            				shuffle='batch',
            				class_weight=class_weight)
            else:
            	history = classifier.fit(X_train, y_train, 
            				batch_size=mini_batch_size,
            				nb_epoch=1,
            				shuffle='batch',
            				class_weight=class_weight)
            
            classifier.save(fold_best_model_path)
        # ==================== EVALUATION ========================     
        
        # Load best model
        print('Model loaded from checkpoint')
        classifier = load_model(fold_best_model_path)
    
        if compute_metrics:
           predicted = classifier.predict(np.asarray(X2))
           for i in range(len(predicted)):
               if predicted[i] < threshold:
                   predicted[i] = 0
               else:
                   predicted[i] = 1
           # Array of predictions 0/1
           predicted = np.asarray(predicted).astype(int)   
           # Compute metrics and print them
           cm = confusion_matrix(_y2, predicted,labels=[0,1])
           tp = cm[0][0]
           fn = cm[0][1]
           fp = cm[1][0]
           tn = cm[1][1]
           tpr = tp/float(tp+fn)
           fpr = fp/float(fp+tn)
           fnr = fn/float(fn+tp)
           tnr = tn/float(tn+fp)
           precision = tp/float(tp+fp)
           recall = tp/float(tp+fn)
           specificity = tn/float(tn+fp)
           f1 = 2*float(precision*recall)/float(precision+recall)
           accuracy = accuracy_score(_y2, predicted)
           
           print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
           print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(
    						tpr,tnr,fpr,fnr))   
           print('Sensitivity/Recall: {}'.format(recall))
           print('Specificity: {}'.format(specificity))
           print('Precision: {}'.format(precision))
           print('F1-measure: {}'.format(f1))
           print('Accuracy: {}'.format(accuracy))
           
if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    main()
