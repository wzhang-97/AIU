import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from vis.utils import utils
from vis.input_modifiers import Jitter

path = '/home/wzhang/AttributesandUnet_testsforPaper/data_for_prediction/bgp_kerry_f3d_attri/histo_to_3_epoch85/'
path_tr = '/home/wzhang/AttributesandUnet_testsforPaper/data_for_train/Normalized_fake_400/'
path_pr = '/home/wzhang/AttributesandUnet_testsforPaper/data_for_prediction/bgp_kerry_f3d_attri/'

os.environ['CUDA_VISIBLE_DEVICES']='2'
# n1,n2,n3=600,601,501 #dfb_cut
# n1,n2,n3=700,500,600 #hz08_cut
# n1,n2,n3=350,530,340 #volve
# n1,n2,n3=240,800,825 #new
def main():
  loadModel()
  # gof3()
  goboxing()
  # gobgp()
  # gokerry()
  # gofakedata()
  # gobgp_subsampled()
  # godfb()
  # gohz08()
  # govolve()
  # gonew()

def loadModel():
  global model
  model = load_model(path+"../../../test6/check6/fseg-85.hdf5")
  plot_model(model, to_file='model_summary.png', show_shapes=True)

def goboxing(): 
  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=926,566,534 #boxing
  n11,n22,n33=463,566,534# boxing_2subsampled
  n4=24
  m1,m2,m3=256,256,256
  m11,m22,m33=128,256,256
  gm = np.zeros((n1,n2,n3,n4))
  gm2 = np.zeros((n11,n22,n33,n4))

  fpath = path_pr
  fname1 = "boxing.dat"
  fname2 = "boxing_2subsampled.dat"

  # for k, name in enumerate(names):
  #   gx = loadData(n1,n2,n3,fpath+name+"/",fname1) 
  #   gm[:,:,:,k] = gx
  # fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  # fx = np.transpose(fx)
  # fx.tofile(path+"test6_boxing_18226.dat",format="%4")

  for k, name in enumerate(names):
    gx = loadData(n11,n22,n33,fpath+name+"/",fname2) 
    gm2[:,:,:,k] = gx
  fx = goPredictSubs(m11,m22,m33,gm2) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_boxing_z_2_subsampled_18226.dat",format="%4")

def godfb(): 
  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=600,601,501 #dfb_cut
  n4=24
  m1,m2,m3=128,256,256
  gm = np.zeros((n1,n2,n3,n4))

  fpath = path_pr
  fname = "dfb.dat"

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gm[:,:,:,k] = gx 
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_dfb_cut.dat",format="%4")

def gohz08(): 
  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=700,500,600 #hz08_cut
  n4=24
  m1,m2,m3=128,256,256
  gm = np.zeros((n1,n2,n3,n4))

  fpath = path_pr
  fname = "hz08.dat"

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gm[:,:,:,k] = gx 
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_hz08_cut.dat",format="%4")

def govolve(): 
  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=350,530,340 #volve
  n4=24
  m1,m2,m3=128,256,256
  gm = np.zeros((n1,n2,n3,n4))

  fpath = path_pr
  fname = "volve.dat"

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gm[:,:,:,k] = gx 
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_volve.dat",format="%4")

def gonew(): 
  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=240,800,825 #new
  n4=24
  m1,m2,m3=128,256,256
  gm = np.zeros((n1,n2,n3,n4))

  fpath = path_pr
  fname = "new.dat"

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gm[:,:,:,k] = gx 
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_new.dat",format="%4")

def gobgp_subsampled(): 
  print("gobgp_subsampled")

  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=700,481,281# bgp_cut
  n11,n22,n33=350,481,281# bgp_cut
  n4=24
  m1,m2,m3=128,256,256
  m11,m22,m33=256,256,256

  gm = np.zeros((n1,n2,n3,n4))
  gm2 = np.zeros((n11,n22,n33,n4))

  fpath = path_pr
  fname1 = "bgp_cut_z_2subsampled.dat"
  fname2 = "bgp_cut_z_4subsampled.dat"

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname1) 
    gm[:,:,:,k] = gx
  fx = goPredictSubs(m11,m22,m33,gm) #make a prediction 
  fx = np.transpose(fx)
  
  fx.tofile(path+"test6_bgp_cut_z_2_subsampled_6223.dat",format="%4")

  for k, name in enumerate(names):
    gx = loadData(n11,n22,n33,fpath+name+"/",fname2) 
    gm2[:,:,:,k] = gx
  fx = goPredictSubs(m1,m2,m3,gm2) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_bgp_cut_z_4_subsampled_6223.dat",format="%4")

def gof3(): 
  print("gof3")

  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=388,901,593 # f3_1
  n4=24
  m1,m2,m3=128,256,256
  gm = np.zeros((n1,n2,n3,n4))

  fpath = path_pr
  fname = "f3d_cut.dat"

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gm[:,:,:,k] = gx 
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_f3d_cut.dat",format="%4")

def gobgp(): 
  print("gobgp")

  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=1400,481,281# bgp_cut
  n4=24
  # m1,m2,m3=128,256,256
  m1,m2,m3=256,256,256
  gm = np.zeros((n1,n2,n3,n4))

  fpath = path_pr
  fname1 = "bgp_cut6223.dat"
  fname2 = "bgp_cut18226.dat"
  fname3 = "bgp_cut24228.dat"
  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname1) 
    gm[:,:,:,k] = gx
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_bgp_cut6223.dat",format="%4")

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname2) 
    gm[:,:,:,k] = gx
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_bgp_cut18226.dat",format="%4")

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname3) 
    gm[:,:,:,k] = gx
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_bgp_cut24228.dat",format="%4")

def gokerry(): 
  print("gokerry")

  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1,n2,n3=650,575,170 # kerry cut
  n4=24
  m1,m2,m3=128,128,128
  gm = np.zeros((n1,n2,n3,n4))

  fpath = path_pr
  fname = "kerry_cut.dat"

  for k, name in enumerate(names):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gm[:,:,:,k] = gx
  fx = goPredictSubs(m1,m2,m3,gm) #make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test6_kerry_cut.dat",format="%4")

def gofakedata(): 
  print("gofakedata")

  names = ["nx","ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
            "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
            "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]
  n1, n2, n3 = 128, 128, 128  # fake
  n4 = 24
  m1, m2, m3 = 128, 128, 128
  gm = np.zeros((n1, n2, n3, n4))

  fpath = path_tr
  start_i = 380
  end_i = 400

  for i in range(start_i, end_i):
    fname = str(i) + '.dat'

    for k, name in enumerate(names):
      gx = loadData(n1, n2, n3, fpath + name + "/", fname) 
      gm[:, :, :, k] = gx

    fx = goPredictSubs(m1, m2, m3, gm)  # 进行预测
    fx = np.transpose(fx)
    output_file = path + "test6_fake" + str(i) + ".dat"
    fx.tofile(output_file, format="%4")

#m1,m2,m3:the dimensions of a subset
#each needs be divisible by 16,
#choose large dimensions if your CPU/GPU memory allows
def goPredictSubs(m1,m2,m3,gx): 
  n1,n2,n3,n4=gx.shape 
  p1,p2,p3=16,16,16 #overlap
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  c1=1+int(np.ceil(float(n1-m1)/(m1-p1)))
  c2=1+int(np.ceil(float(n2-m2)/(m2-p2)))
  c3=1+int(np.ceil(float(n3-m3)/(m3-p3)))
  for k3 in range(c3):
    for k2 in range(c2):
      for k1 in range(c1):
        gk = np.zeros((m1,m2,m3,n4),dtype=np.single)
        ###########
        b1,b2,b3 = k1*(m1-p1),k2*(m2-p2),k3*(m3-p3)
        t1,t2,t3 = b1+p1,b2+p2,b3+p3
        e1,e2,e3 = b1+m1,b2+m2,b3+m3
        e1 = min(e1,n1)
        e2 = min(e2,n2)
        e3 = min(e3,n3)
        b1,b2,b3 = e1-m1,e2-m2,e3-m3
        gk[:,:,:,:] = gx[b1:e1,b2:e2,b3:e3,:]
        ###########
        for k in range(n4):
          gk[:,:,:,k]=dataprocess(gk[:,:,:,k])
        gk = np.reshape(gk,(1,m1,m2,m3,n4))
        fk = model.predict(gk,verbose=1) #fault prediction
        ###########
        bt1,bt2,bt3=t1-b1,t2-b2,t3-b3
        t1 = min(int(bt1/2),b1)
        t2 = min(int(bt2/2),b2)
        t3 = min(int(bt3/2),b3)
        fx[b1+t1:e1,b2+t2:e2,b3+t3:e3] = fk[0,t1:,t2:,t3:,0]
        ###########
  #set the bounds
  fx[-1,:,:]=fx[-2,:,:]
  fx[:,-1,:]=fx[:,-2,:]
  fx[:,:,-1]=fx[:,:,-2]
  return fx

def dataprocess(gx):
  # gm = np.mean(x)
  # gs = np.std(x)
  # gx = x-gm
  # gx = gx/gs 
  vmin_s, vmax_s = -3, 3
  # 线性拉伸到0-255范围内
  vmin, vmax = np.min(gx), np.max(gx)
  gx = (gx - vmin) * 255.0 / (vmax - vmin)
  # 计算直方图
  hist, bins = np.histogram(gx, bins=256, range=(0, 255))
  # 计算直方图均衡化
  cdf = hist.cumsum()
  cdf = 255 * cdf / cdf[-1]
  gx_eq = np.interp(gx.flatten(), bins[:-1], cdf).reshape(gx.shape)
  # 将灰度图像值域映射回数据值域范围
  gx_eq = gx_eq / 255.0 * (vmax_s - vmin_s) + vmin_s
  gx_eq.astype(np.single)
  return gx_eq

def loadData(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  # gm,gs = np.mean(gx),np.std(gx)
  # gx = gx-gm
  # gx = gx/gs
  gx[np.isnan(gx)] = 0
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

def loadDatax(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

def sigmoid(x):
    s=1.0/(1.0+np.exp(-x))
    return s

def plot2d(gx,fx,fp,at=1,png=None):
  fig = plt.figure(figsize=(15,5))
  #fig = plt.figure()
  ax = fig.add_subplot(131)
  ax.imshow(gx,vmin=-2,vmax=2,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(132)
  ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(133)
  ax.imshow(fp,vmin=0,vmax=1.0,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  if png:
    plt.savefig(pngDir+png+'.png')
  #cbar = plt.colorbar()
  #cbar.set_label('Fault probability')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
    main()
