# Import relevant libraries

import scipy.io as sio
import numpy as np
import tqdm

# The code takes the entire hsi/lidar image as input for 'X' and grounttruth file as input for 'y'
# and the patchsize as for 'windowSize'.
# The output are the patches centered around the groundtruth pixel, the corresponding groundtruth label and the
# pixel location of the patch.

def make_patches(X, y, windowSize):

  shapeX = np.shape(X)

  margin = int((windowSize-1)/2)
  newX = np.zeros([shapeX[0]+2*margin,shapeX[1]+2*margin,shapeX[2]])

  newX[margin:shapeX[0]+margin:,margin:shapeX[1]+margin,:] = X

  index = np.empty([0,3], dtype = 'int')

  cou = 0
  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    for i in range(shapeX[0]):
      for j in range(shapeX[1]):
        if y[i,j] == k:
          index = np.append(index,np.expand_dims(np.array([k,i,j]),0),0)
          #print(cou)
          cou = cou+1

  patchesX = np.empty([index.shape[0],2*margin+1,2*margin+1,shapeX[2]], dtype = 'float32')
  patchesY = np.empty([index.shape[0]],dtype = 'uint8')

  for i in range(index.shape[0]):
    p = index[i,1]
    q = index[i,2]
    patchesX[i,:,:,:] = newX[p:p+windowSize,q:q+windowSize,:]
    patchesY[i] = index[i,0]

  return patchesX, patchesY, index

# Reading data
data = sio.loadmat('/data/houston2013.mat')

# Concatenating HSI and LiDAR bands from the data and removing spurious pixels
feats = np.concatenate([data['hsi'], np.expand_dims(data['lidar'], axis = 2)], axis = 2)

# Normalising the bands using min-max normalization 

feats_norm = np.empty([349,1905,145], dtype = 'float32')
for i in tqdm.tqdm(range(145)):
  feats_norm[:,:,i] = feats[:,:,i]-np.min(feats[:,:,i])
  feats_norm[:,:,i] = feats_norm[:,:,i]/np.max(feats_norm[:,:,i])

## REading train and test groundtruth images

train = data['train']
test = data['test']

# Create train patches
train_patches, train_labels, index_train = make_patches(feats_norm, train, 11)

# Create test patches
test_patches, test_labels, index_test = make_patches(feats_norm, test, 11)

# Data augmentation by rotating patches by 90, 180 and 270 degrees

tr90 = np.empty([2832,11,11,145], dtype = 'float32')
tr180 = np.empty([2832,11,11,145], dtype = 'float32')
tr270 = np.empty([2832,11,11,145], dtype = 'float32')

for i in tqdm.tqdm(range(2832)):
  tr90[i,:,:,:] = np.rot90(train_patches[i,:,:,:])
  tr180[i,:,:,:] = np.rot90(tr90[i,:,:,:])
  tr270[i,:,:,:] = np.rot90(tr180[i,:,:,:])

train_patches = np.concatenate([train_patches, tr90, tr180, tr270], axis = 0)
train_labels = np.concatenate([train_labels,train_labels,train_labels,train_labels], axis = 0)

# Save the train patches/ test patches along with the labels

np.save('/data/train_patches',train_patches)
np.save('/data/test_patches',test_patches)
np.save('/data/train_labels',train_labels)
np.save('data/test_labels',test_labels)

# Save the normalised HSI and LiDAR images

np.save('/data/Houston/hsi',feats_norm[:,:,0:144])
np.save('/data/Houston/lidar',feats_norm[:,:,144])
