import tensorflow as tf
import os
import random
import numpy as np
#import skimage
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
from utils import DataGenerator
from unet import *
#from unet import *



#fpath = '/home/wenzhang/multi unet/'
fpath = '/home/wzhang/AttributesandUnet_testsforPaper/'

os.environ['CUDA_VISIBLE_DEVICES']='2'#训练的时候用这个GPU.

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':1,#一次训练所选取的样本数
          'dim':(128,128,128),
          'n_channels':93,
          'shuffle': True}
  seismPath = fpath+"data_for_train/Normalized_fake_400/"
  faultPath = fpath+"data_for_train/fx/"

  train_ID=[]#定义一个新的列表
  valid_ID=[]
  c = 0
  for sfile in os.listdir(seismPath+'nx/'):
    if sfile.endswith(".dat"):
      if(c<380):
        train_ID.append(sfile)

      else:
        valid_ID.append(sfile)
      c = c+1

  train_generator = DataGenerator(dpath=seismPath,fpath=faultPath,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPath,fpath=faultPath,
                                  data_IDs=valid_ID,**params)

  model = unet(input_size1 = (None,None,None,1),input_size2 = (None,None,None,23),input_size3 = (None,None,None,23),input_size4 = (None,None,None,23),input_size5 = (None,None,None,23))
  #model = unet(input_size=((None,None,None,19)))
  #model = unet(input_size=((None,None,None,1)))
  model.compile(optimizer=Adam(lr=1e-4), loss="mse",metrics = ['accuracy'])
  #model.compile(optimizer = Adam(lr = 1e-5), loss = cross_entropy_balanced, metrics = ['accuracy'])
  model.summary()

  # checkpoint
  filepath=fpath+"test21/check21/fseg-{epoch:02d}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
        verbose=1, save_best_only=False, mode='max')
  #logging = TrainValTensorBoard()
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                patience=2, min_lr=1e-8)
  callbacks_list = [checkpoint, reduce_lr]
  print("data prepared, ready to train!")
  # Fit the model
  history=model.fit_generator(generator=train_generator,
  validation_data=valid_generator,epochs=130,callbacks=callbacks_list,verbose=1)
  #model.save(fpath+'check25_nx_out_19_in_simpleunet_removecon1conca/fseg.hdf5')
  showHistory(history)

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  fig = plt.figure(figsize=(10,6))

  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()
  plt.savefig('history for accuracy21.png')
  plt.close()

  # summarize history for loss
  fig = plt.figure(figsize=(10,6))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()
  plt.savefig('history for loss21.png')
  plt.close()

  with open('loss21.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['loss']))
  with open('val_loss21.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_loss']))

  with open('acc21.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['acc']))
  with open('val_acc21.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_acc']))  



if __name__ == '__main__':
    main()
