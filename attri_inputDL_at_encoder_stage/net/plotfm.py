from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
#import cv2
import numpy as np
import math
import skimage
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model
#from __future__ import absolute_import, division
#from deform_conv.layers import DeformableConv
import tensorflow as tf
#from keras import backend
from keras.layers import *
from keras.models import load_model
#from skimage.measure import compare_psnr
from unet import cross_entropy_balanced
import os
from unet import *
from unet import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
path = '/home/wenzhang/multi unet/'
fpath = '/home/wenzhang/multi unet/data/3rd/'
pngDir = './png/'

#md = 40 #,65,70 #trained models at different epoch

def main():
  #goVisual()
  goPlotfm()

def goVisual():
    global modelv
    modelv = load_model(path+'check2/fseg-02.hdf5')
    modelv.summary()
    plot_model(modelv, to_file="model_19+1.png", show_shapes=True)

def goPlotfm():
    model = unet(input_size=(128,128,128,20))
    #model.load_weights('model/fsegv2-'+str(md)+'.hdf5')
    model = load_model(path+'check2/fseg-130.hdf5')
    n1,n2,n3,n4 = 128,128,128,20
    gm = np.zeros((n1,n2,n3,n4))
    names = ["nx","Grad_fx","Grad_fy","Grad_fz","ilSlope","clSlope","DGSemb",
             "DGVar","CVar","planarity","envelope","swPosCur","swNegCur","lwPosCur","lwNegCur",
             "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
    fname = "0.dat"
    for k, name in enumerate(names):
      gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
      gm[:,:,:,k] = gx
    #gm=np.reshape(gx,(20,128,128,128,1))
    gm=np.reshape(gm,(1,128,128,128,20))


    # 第一个 model.layers[0],不修改,表示输入数据；
    # 第二个model.layers[ ],修改为需要输出的层数的编号[]
    layer_1 = K.function([model.layers[0].input], [model.layers[2].output])

    # 只修改input_image
    f1 = layer_1([gm])[0]

    # 第一层卷积后的特征图展示，输出是（1,66,66,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    for i in range(32):
                show_img = f1[ :, 50, :, :, i]
                show_img.shape = [128,128]
                fig = plt.figure(figsize=(10,10))
                #plt.subplot(4, 4, i + 1)
                # plt.imshow(show_img, cmap='black')
                plt.imshow(show_img, cmap='gray')
                plt.axis('off')
                #plt.savefig('unet+分频特征图 layer35的'+str(i)+'.png')
                plt.show()

def loadData(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  #gmin,gmax=np.min(gx)/5,np.max(gx)/5
  gm,gs = np.mean(gx),np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx
                
if __name__ == '__main__':

    main()