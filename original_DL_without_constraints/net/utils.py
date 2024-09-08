import numpy as np
import keras
import random
from keras.utils import to_categorical
import os

class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,dpath,fpath,data_IDs, batch_size=1, dim=(128,128,128), 
             n_channels=5, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  '''def  __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    A, B, C, D, Y = self.__data_generation_with_additional_infomation(data_IDs_temp)

    return [A, B, C, D], Y'''

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    A,Y = self.__data_generation(data_IDs_temp)

    return A, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  '''def __data_generation_with_additional_infomation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    a = 2 #data augumentation

    A = np.zeros((a*self.batch_size, 128, 128, 128, 1),dtype=np.single)
    B = np.zeros((a*self.batch_size, 128, 128, 128, 19),dtype=np.single)
    C = np.zeros((a*self.batch_size, 64, 64,64, 19),dtype=np.single)
    D = np.zeros((a*self.batch_size, 32,32,32,19),dtype=np.single)
    Y = np.zeros((a*self.batch_size, 128, 128, 128, 1),dtype=np.single)
    name0 = ['nx']
    names1 = ["Grad_fx","Grad_fy","Grad_fz","ilSlope","clSlope","DGSemb",
             "DGVar","CVar","planarity","envelope","swPosCur","swNegCur","lwPosCur","lwNegCur",
             "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
    names2 = ['Grad_fx_2subsampled','Grad_fy_2subsampled','Grad_fz_2subsampled',
            'ilSlope_2subsampled','clSlope_2subsampled','DGSemb_2subsampled',
             'DGVar_2subsampled','CVar_2subsampled','planarity_2subsampled',
             'envelope_2subsampled','swPosCur_2subsampled','swNegCur_2subsampled',
             'lwPosCur_2subsampled','lwNegCur_2subsampled',
             'ctrGLCM_2subsampled','entrGLCM_2subsampled','enerGLCM_2subsampled',
             'dissimiGLCM_2subsampled','homoGLCM_2subsampled']
    names3 = ['Grad_fx_4subsampled','Grad_fy_4subsampled','Grad_fz_4subsampled',
            'ilSlope_4subsampled','clSlope_4subsampled','DGSemb_4subsampled',
             'DGVar_4subsampled','CVar_4subsampled','planarity_4subsampled',
             'envelope_4subsampled','swPosCur_4subsampled','swNegCur_4subsampled',
             'lwPosCur_4subsampled','lwNegCur_4subsampled',
             'ctrGLCM_4subsampled','entrGLCM_4subsampled','enerGLCM_4subsampled',
             'dissimiGLCM_4subsampled','homoGLCM_4subsampled']                 
    #Names = ['gx2_v1', 'gx2_v2', 'gx3_v1', 'gx3_v2']
    names1 = ["DGSemb","planarity"]
    names2 = ['DGSemb_2subsampled','planarity_2subsampled']
    names3 = ['DGSemb_4subsampled','planarity_4subsampled']  
    names1 = ['Grad_fz','DGSemb','DGVar','CVar',
             'planarity','swPosCur','swNegCur',
             'lwPosCur','lwNegCur']
    names2 = ['Grad_fz_2subsampled','DGSemb_2subsampled','DGVar_2subsampled','CVar_2subsampled',
             'planarity_2subsampled','swPosCur_2subsampled','swNegCur_2subsampled',
             'lwPosCur_2subsampled','lwNegCur_2subsampled']
    names3 = ['Grad_fz_4subsampled','DGSemb_4subsampled','DGVar_4subsampled','CVar_4subsampled',
             'planarity_4subsampled','swPosCur_4subsampled','swNegCur_4subsampled',
             'lwPosCur_4subsampled','lwNegCur_4subsampled'] 
    names1 = ["Grad_fx","Grad_fy","ilSlope","clSlope","envelope",
             "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
    names2 = ['Grad_fx_2subsampled','Grad_fy_2subsampled',
            'ilSlope_2subsampled','clSlope_2subsampled',
             'envelope_2subsampled','ctrGLCM_2subsampled',
             'entrGLCM_2subsampled','enerGLCM_2subsampled',
             'dissimiGLCM_2subsampled','homoGLCM_2subsampled']
    names3 = ['Grad_fx_4subsampled','Grad_fy_4subsampled',
            'ilSlope_4subsampled','clSlope_4subsampled',
             'envelope_4subsampled','ctrGLCM_4subsampled',
             'entrGLCM_4subsampled','enerGLCM_4subsampled',
             'dissimiGLCM_4subsampled','homoGLCM_4subsampled']    
    for k in range(self.batch_size):
      c = k*a
      fx  = np.fromfile(self.dpath+'fx/'+str(data_IDs_temp[k]),dtype=np.single)
      fx = np.reshape(fx,(128,128,128))
      fx = 2*np.clip(fx,0,1)
      fx = np.transpose(fx)
      Y[c+0,:,:,:,0] = fx
      i = random.randint(0,3)
      Y[c+1,:,:,:,0] = np.rot90(fx,i,(1,2))

      for j,name in enumerate(name0):
        gx = np.fromfile(self.dpath+name+'/'+str(data_IDs_temp[k]),dtype=np.single)
        gx = np.reshape(gx,(128,128,128))
        gx = dataprocess(gx)
        A[c+0,:,:,:,j] = gx
        A[c+1,:,:,:,j] = np.rot90(gx,i,(1,2))

      for j,name in enumerate(names1):
        gx = np.fromfile(self.dpath+name+'/'+str(data_IDs_temp[k]),dtype=np.single)
        gx = np.reshape(gx,(128,128,128))
        gx = dataprocess(gx)
        B[c+0,:,:,:,j] = gx
        B[c+1,:,:,:,j] = np.rot90(gx,i,(1,2))

      for j,name in enumerate(names2):
        gx = np.fromfile(self.dpath+'2_subsampled/'+name+'/'+str(data_IDs_temp[k]),dtype=np.single)
        gx = np.reshape(gx,(64,64,64))
        gx = dataprocess(gx)
        C[c+0,:,:,:,j] = gx
        C[c+1,:,:,:,j] = np.rot90(gx,i,(1,2))

      for j,name in enumerate(names3):
        gx = np.fromfile(self.dpath+'4_subsampled/'+name+'/'+str(data_IDs_temp[k]),dtype=np.single)
        gx = np.reshape(gx,(32,32,32))
        gx = dataprocess(gx)
        D[c+0,:,:,:,j] = gx
        D[c+1,:,:,:,j] = np.rot90(gx,i,(1,2))

    return A, B, C, D, Y'''

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    a = 2 #data augumentation

    A = np.zeros((a*self.batch_size,128,128,128, self.n_channels),dtype=np.single)
    Y = np.zeros((a*self.batch_size,128,128,128,1),dtype=np.single)

    '''names = ["nx","Grad_fx","Grad_fy","Grad_fz","ilSlope","clSlope","DGSemb",
             "DGVar","CVar","planarity","envelope","swPosCur","swNegCur","lwPosCur","lwNegCur",
             "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]'''
    #names = ['nx','DGSemb','planarity']
    names = ['nx']      
    #names = ['DGSemb']  
    '''names = ["Grad_fx","Grad_fy","Grad_fz","ilSlope","clSlope","DGSemb",
             "DGVar","CVar","planarity","envelope","swPosCur","swNegCur","lwPosCur","lwNegCur",
             "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]'''
    for k in range(self.batch_size):
      c = k*a
      fx  = np.fromfile(self.dpath+'fx/'+str(data_IDs_temp[k]),dtype=np.single)
      fx = np.reshape(fx,(128,128,128))
      fx = 2*np.clip(fx,0,1)
      fx = np.transpose(fx)
      Y[c+0,:,:,:,0] = fx
      i = random.randint(0,3)
      Y[c+1,:,:,:,0] = np.rot90(fx,i,(1,2))
      for j,name in enumerate(names):
        gx = np.fromfile(self.dpath+name+'/'+str(data_IDs_temp[k]),dtype=np.single)
        gx = np.reshape(gx,(128,128,128))
        gx = dataprocess(gx)
        A[c+0,:,:,:,j] = gx
        gx = np.rot90(gx,i,(1,2))
        A[c+1,:,:,:,j] = gx
    return A,Y


def dataprocess(x):
  # gm = np.mean(x)
  # gs = np.std(x)
  # gx = x-gm
  # gx = gx/gs
  gx = np.transpose(x)
  return gx
