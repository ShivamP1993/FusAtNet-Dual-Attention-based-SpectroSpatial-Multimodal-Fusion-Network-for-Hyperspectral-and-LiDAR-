# Import all the necessary libraries and classes

import numpy as np
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras.layers import Input
from keras.layers import Conv2D, Reshape, BatchNormalization
from keras.layers import Concatenate
from keras.layers import Multiply, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
from sklearn.metrics import confusion_matrix
from keras import regularizers

# Function to perform one hot encoding of the class labels 

def my_ohc(lab_arr):
    lab_arr_unique =  np.unique(lab_arr)
    r,c = lab_arr.shape
    r_u  = lab_arr_unique.shape
    
    one_hot_enc = np.zeros((r,r_u[0]), dtype = 'float')
    
    for i in range(r):
        for j in range(r_u[0]):
            if lab_arr[i,0] == lab_arr_unique[j]:
                one_hot_enc[i,j] = 1
    
    return one_hot_enc

# Function that takes the confusion matrix as input and
# calculates the overall accuracy, producer's accuracy, user's accuracy,
# Cohen's kappa coefficient and syandard deviation of 
# Cohen's kappa coefficient

def accuracies(cm):
  import numpy as np
  num_class = np.shape(cm)[0]
  n = np.sum(cm)

  P = cm/n
  ovr_acc = np.trace(P)

  p_plus_j = np.sum(P, axis = 0)
  p_i_plus = np.sum(P, axis = 1)

  usr_acc = np.diagonal(P)/p_i_plus
  prod_acc = np.diagonal(P)/p_plus_j

  theta1 = np.trace(P)
  theta2 = np.sum(p_plus_j*p_i_plus)
  theta3 = np.sum(np.diagonal(P)*(p_plus_j + p_i_plus))
  theta4 = 0
  for i in range(num_class):
    for j in range(num_class):
      theta4 = theta4+P[i,j]*(p_plus_j[i]+p_i_plus[j])**2

  kappa = (theta1-theta2)/(1-theta2)

  t1 = theta1*(1-theta1)/(1-theta2)**2
  t2 = 2*(1-theta1)*(2*theta1*theta2-theta3)/(1-theta2)**3
  t3 = ((1-theta1)**2)*(theta4 - 4*theta2**2)/(1-theta2)**4

  s_sqr = (t1+t2+t3)/n

  return ovr_acc, usr_acc, prod_acc, kappa, s_sqr

# This is the main feature extractor for the hyperpsectral images. 
# The input is a hyperspectral patch. It consists of 6 sets of 
# convolutional, relu and batch normalization operations 

def hs(x):
  
  conv1 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                            activation='relu', use_bias=True,  
                            kernel_regularizer=regularizers.l2(0.01), name = 'conv31')(x)

  conv1 = BatchNormalization()(conv1)

  conv2 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'conv32')(conv1)

  conv2 = BatchNormalization()(conv2) 
  

  conv3 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'conv33')(conv2) 

  conv3 = BatchNormalization()(conv3)
                           
  conv4 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'conv34')(conv3) 

  conv4 = BatchNormalization()(conv4)


  conv5 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'conv35')(conv4) 

  conv5 = BatchNormalization()(conv5)

  conv6 = Conv2D(1024, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'conv36')(conv5) 

  conv6 = BatchNormalization()(conv6)

  return conv6

# This is the spectral attention mask for hyperspecral images.
# The input are hyperspectral patches and output is an attention vector 
# It consists of 6 convolutional, relu and batch normalization operations. 
# There is a residual block and 2D maxpool operation after second and fourth 
# convolution layers. Last convolution layer is followed by a maxpool and 
# Global average pool operation.

def mask_spec(x):
  
  conv1 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                            activation='relu', use_bias=True,  
                            kernel_regularizer=regularizers.l2(0.01), name = 'convc1')(x)

  conv1 = BatchNormalization(name = 'BNc1')(conv1)

  conv2 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01),  name = 'convc2')(conv1)

  conv2 = BatchNormalization(name = 'BNc2')(conv2) 

  res1 = Concatenate(axis = 3)([conv1, conv2])
  
  mp1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(res1)

  conv3 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convc3')(mp1) 

  conv3 = BatchNormalization(name = 'BNc3')(conv3)
                           
  conv4 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convc4')(conv3) 

  conv4 = BatchNormalization(name = 'BNc4')(conv4)
  res2 = Concatenate(axis = 3)([conv3, conv4])
  
  mp2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(res2)

  conv5 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convc5')(mp2) 

  conv5 = BatchNormalization(name = 'BNc5')(conv5)

  conv6 = Conv2D(1024, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convc6')(conv5) 

  conv6 = BatchNormalization(name = 'BNc6')(conv6)

  mp3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv6)
  gap1 = GlobalAveragePooling2D()(mp3)

  return gap1

# This is the spatial attention mask for hyperspecral images.
# The input are lidar patches and output is an attention tensor 
# It consists of 6 convolutional, relu and batch normalization operations. 
# There is a residual block after second and fourth 
# convolution layers. 

def mask_spat(x):
  
  conv1 = Conv2D(128, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                            activation='relu', use_bias=True,  
                            kernel_regularizer=regularizers.l2(0.01), name = 'convt1')(x)

  conv1 = BatchNormalization(name = 'BNt1')(conv1)

  conv2 = Conv2D(128, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01),  name = 'convt2')(conv1)

  conv2 = BatchNormalization(name = 'BNt2')(conv2) 

  res1 = Concatenate(axis = 3)([conv1, conv2])

  conv3 = Conv2D(128, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convt3')(res1) 

  conv3 = BatchNormalization(name = 'BNt3')(conv3)
                           
  conv4 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convt4')(conv3) 

  conv4 = BatchNormalization(name = 'BNt4')(conv4)

  res2 = Concatenate(axis = 3)([conv3, conv4])

  conv5 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convt5')(res2) 

  conv5 = BatchNormalization(name = 'BNt5')(conv5)

  conv6 = Conv2D(1024, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convt6')(conv5) 

  conv6 = BatchNormalization(name = 'BNt6')(conv6)

  return conv6

# It is a part of modality attention module.  
# The input are highlighted spectral and spatial attention features from above modules. 
# It consists of 6 convolutional, relu and batch normalization operations. 

def main2(x):
  
  conv1 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                            activation='relu', use_bias=True,  
                            kernel_regularizer=regularizers.l2(0.01), name = 'convm31')(x)

  conv1 = BatchNormalization()(conv1)

  conv2 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convm32')(conv1)

  conv2 = BatchNormalization()(conv2) 

  conv3 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convm33')(conv2) 

  conv3 = BatchNormalization()(conv3)
                           
  conv4 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convm34')(conv3) 

  conv4 = BatchNormalization()(conv4)
  

  conv5 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convm35')(conv4) 

  conv5 = BatchNormalization()(conv5)

  conv6 = Conv2D(1024, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convm36')(conv5) 

  conv6 = BatchNormalization()(conv6)

  return conv6

# This is the attention layer for maodality attention module. 
# The input are highlighted spectral and spatial attention features from above modules. 
# It consists of 6 convolutional, relu and batch normalization operations. 
# There is a residual block after second and fourth 
# convolution layers. 

def att2(x):
  
  conv1 = Conv2D(128, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                            activation='relu', use_bias=True,  
                            kernel_regularizer=regularizers.l2(0.01), name = 'convatt1')(x)

  conv1 = BatchNormalization(name = 'BN2t1')(conv1)

  conv2 = Conv2D(128, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convatt2')(conv1)

  conv2 = BatchNormalization(name = 'BN2t2')(conv2) 

  res1 = Concatenate(axis = 3)([conv1, conv2])

  conv3 = Conv2D(128, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convatt3')(res1) 

  conv3 = BatchNormalization(name = 'BN2t3')(conv3)
                           
  conv4 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convatt4')(conv3) 

  conv4 = BatchNormalization(name = 'BN2t4')(conv4)

  res2 = Concatenate(axis = 3)([conv3, conv4])

  conv5 = Conv2D(256, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convatt5')(res2) 

  conv5 = BatchNormalization(name = 'BN2t5')(conv5)

  conv6 = Conv2D(1024, (3,3), strides=(1, 1), padding='same', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convatt6')(conv5) 

  conv6 = BatchNormalization(name = 'BN2t6')(conv6)

  return conv6

# This is a classifier function. It is a CNN with 6 layers
# (convolution + RelU + Batch Normalization). Inputs are 
# Attention assisted enhanced features from modality attention module 
# and number of classes

def clf(x, num_classes):
  
  conv1 = Conv2D(256, (3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                            activation='relu', use_bias=True,  
                            kernel_regularizer=regularizers.l2(0.01), name = 'convcl1')(x)

  conv1 = BatchNormalization(name = 'BNcl1')(conv1)
  

  conv2 = Conv2D(256, (3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convcl2')(conv1)

  conv2 = BatchNormalization(name = 'BNcl2')(conv2) 
  
  
  conv3 = Conv2D(256, (3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convcl3')(conv2) 

  conv3 = BatchNormalization(name = 'BNcl3')(conv3)
  
                           
  conv4 = Conv2D(256, (3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convcl4')(conv3) 

  conv4 = BatchNormalization(name = 'BNcl4')(conv4)
  
  conv5 = Conv2D(1024, (3,3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='relu', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convcl5')(conv4) 

  conv5 = BatchNormalization(name = 'BNcl5')(conv5)

  conv6 = Conv2D(num_classes, (1,1), strides=(1, 1), padding='valid', dilation_rate=(1, 1), 
                              activation='softmax', use_bias=True,  
                              kernel_regularizer=regularizers.l2(0.01), name = 'convcl6')(conv5) 
  

  return Reshape([num_classes])(conv6)

# reading training and test data

train_patches = np.load('/data/train_patches.npy')
test_patches = np.load('/data/test_patches.npy')

train_labels = np.load('/data/train_labels.npy')
test_labels = np.load('/data/test_labels.npy')

# Separating HSI and lidar patches from training
# validation and testing data

#num_hsi_bands = 144
#num_lidar_bands = 1

train_hsi = train_patches[:,:,:,0:144]
train_lidar = np.expand_dims(train_patches[:,:,:,144], axis = 3) # Expanding dimension to preserve shape
                                                           #since only one band is present

test_hsi = test_patches[:,:,:,0:144]
test_lidar = np.expand_dims(test_patches[:,:,:,144], axis = 3)

## Training module

K.clear_session()
g = tf.Graph()

k = 0        #k is created to temporarily store the maximum validation accuracy for each epoch

with g.as_default():

  x1 = Input(shape=(11,11,144), name='inputA')     #num_hsi_bands = 144

  x2 = Input(shape=(11,11,1), name='inputB')      #num_lidar_bands = 1

  feats_new = hs(x1)                              # Main feature extraction

  # Generating spectral attention mask and spectrally highlighting HSI features
  spec = Multiply()([feats_new, mask_spec(x1)])   
  
  # Generating spatial attention mask  and spatially highlighting HSI features
  spat = Multiply()([feats_new, mask_spat(x2)]) 

  # Concatenationg highlighted features and input features  
  conc = Concatenate(axis = 3)([x1,x2,spec,spat]) 

  feats2 = main2(conc)                            # Modality features extraction
  mask2 = att2(conc)                              # Modality attention mask

  # Highlighting modality features using modality attention mask

  at_feats = Multiply()([feats2, mask2])  

  clsf = clf(at_feats, 15)                        # Classifier with number of classes = 15

  # Initialising model
  model_att = Model([x1,x2], clsf, name = 'att_clf')

  # Adam with Nesterov Momentum optimizer
  optim = keras.optimizers.Nadam(0.00002, beta_1=0.9, beta_2=0.999)
  
  # Compiling the model
  model_att.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

  for epoch in range(20):  # Number of epochs = 1000
    
    model_att.fit(x = [train_hsi, train_lidar], 
                  y = my_ohc(np.expand_dims(train_labels, axis = 1)),
                  epochs=1, batch_size = 64, verbose = 1)
    
    preds2 = model_att.predict([test_hsi, test_lidar], verbose = 1)

    conf = confusion_matrix(test_labels, np.argmax(preds2,1)) # Test set predictions
    ovr_acc, _, _, _, _ = accuracies(conf)

    if ovr_acc>=k:

      # Saving model for maximum accuracy     
      model_att.save('/models/model')
      k = ovr_acc
      ep = epoch
    print('acc_max_val = ', np.round(100*k,2), '% at epoch', ep) # Maximum test accuracy


# Evaluating the model on test set

K.clear_session()
g = tf.Graph()

with g.as_default():

  # Loading saved model
  model = keras.models.load_model('/models/model')

  preds_final = model.predict([test_hsi, test_lidar], verbose = 1)
  conf_final = confusion_matrix(test_labels, np.argmax(preds_final,1))
  ovr_acc_final, usr_acc, prod_acc, kappa, s_sqr = accuracies(conf_final)

print('Test accuracy is ', np.round(100*ovr_acc_final,2), '%') # Final test accuracy

