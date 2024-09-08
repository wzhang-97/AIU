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

os.environ['CUDA_VISIBLE_DEVICES']='1'
# n1,n2,n3=600,601,501 #dfb_cut
# n1,n2,n3=700,500,600 #hz08_cut
# n1,n2,n3=350,530,340 #volve
# n1,n2,n3=240,800,825 #new

def main():
  loadModel()
  # gof3()
  # gobgp()
  # gokerry()
  gofakedata()
  # gobgp_subsampled()
  # godfb()
  # gohz08()
  # govolve()
  # gonew()

def loadModel():
  global model
  model = load_model(path+"../../../test21/check21/fseg-85.hdf5")
  plot_model(model, to_file='model_summary.png', show_shapes=True)
def godfb(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]         

  n1,n2,n3=600,601,501 #dfb_cut
  n4=23
  m1,m2,m3=128,256,256
  gms = np.zeros((n1,n2,n3,1))
  gma = np.zeros((n1,n2,n3,n4))
  fpath = path_pr
  fname = "dfb.dat"

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_dfb_cut.dat",format="%4")

def gohz08(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]         

  n1,n2,n3=700,500,600 #hz08_cut
  n4=23
  m1,m2,m3=128,256,256
  gms = np.zeros((n1,n2,n3,1))
  gma = np.zeros((n1,n2,n3,n4))
  fpath = path_pr
  fname = "hz08.dat"

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_hz08_cut.dat",format="%4")

def govolve(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]         

  n1,n2,n3=350,530,340 #volve
  n4=23
  m1,m2,m3=128,256,256
  gms = np.zeros((n1,n2,n3,1))
  gma = np.zeros((n1,n2,n3,n4))
  fpath = path_pr
  fname = "volve.dat"

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_volve.dat",format="%4")

def gonew(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]         

  n1,n2,n3=240,800,825 #new
  n4=23
  m1,m2,m3=128,256,256
  gms = np.zeros((n1,n2,n3,1))
  gma = np.zeros((n1,n2,n3,n4))
  fpath = path_pr
  fname = "new.dat"

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_new.dat",format="%4")
  
def gobgp_subsampled(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]    
  n1,n2,n3=700,481,281# bgp_cut
  n11,n22,n33=350,481,281# bgp_cut
  n4=23
  m1,m2,m3=128,256,256

  fpath = path_pr
  fname1 = "bgp_cut_z_2subsampled.dat"
  fname2 = "bgp_cut_z_4subsampled.dat"

  gms1 = np.zeros((n1,n2,n3,1))
  gma1 = np.zeros((n1,n2,n3,n4))
  gx1 = loadData(n1,n2,n3,fpath+name0+"/",fname1) 
  gms1[:,:,:,0] = gx1
  for k, name in enumerate(names1):
    gx1 = loadData(n1,n2,n3,fpath+name+"/",fname1) 
    gma1[:,:,:,k] = gx1
  fx = goPredictSubs2(m1,m2,m3,gms1,gma1)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_bgp_cut_z_2_subsampled_6223.dat",format="%4")

  gms2 = np.zeros((n11,n22,n33,1))
  gma2 = np.zeros((n11,n22,n33,n4))
  gx2 = loadData(n11,n22,n33,fpath+name0+"/",fname2) 
  gms2[:,:,:,0] = gx2
  for k, name in enumerate(names1):
    gx2 = loadData(n11,n22,n33,fpath+name+"/",fname2) 
    gma2[:,:,:,k] = gx2
  fx = goPredictSubs2(m1,m2,m3,gms2,gma2)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_bgp_cut_z_4_subsampled_6223.dat",format="%4")

def gof3(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]         

  n1,n2,n3=388,901,593 # f3_1
  n4=23

  m1,m2,m3=128,256,256

  gms = np.zeros((n1,n2,n3,1))
  gma = np.zeros((n1,n2,n3,n4))

  fpath = path_pr

  fname = "f3d_cut.dat"

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_f3d_cut.dat",format="%4")

def gobgp(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]         

  n1,n2,n3=1400,481,281# bgp_cut
  
  n4=23

  m1,m2,m3=128,256,256

  gms = np.zeros((n1,n2,n3,1))
  gma = np.zeros((n1,n2,n3,n4))

  fpath = path_pr

  fname1 = "bgp_cut6223.dat"
  fname2 = "bgp_cut18226.dat"
  fname3 = "bgp_cut24228.dat"

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname1) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname1) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_bgp_cut6223.dat",format="%4")

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname2) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname2) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_bgp_cut18226.dat",format="%4")

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname3) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname3) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_bgp_cut24228.dat",format="%4")

def gokerry(): 
  name0 = 'nx'
  names1 = ["ilSlope","clSlope","planarity","envelope","Grad_fz","CVar","DGVar","DGSemb",
          "lwPosCur","lwNegCur","lwMaxCur","lwMinCur","lwCurvedness",
          "swPosCur","swNegCur","swMaxCur","swMinCur","swCurvedness",
            "ctrGLCM","entrGLCM","enerGLCM","dissimiGLCM","homoGLCM"]         

  n1,n2,n3=650,575,170 # kerry cut
  
  n4=23

  m1,m2,m3=128,128,128

  gms = np.zeros((n1,n2,n3,1))
  gma = np.zeros((n1,n2,n3,n4))

  fpath = path_pr

  fname = "kerry_cut.dat"

  gx = loadData(n1,n2,n3,fpath+name0+"/",fname) 
  gms[:,:,:,0] = gx
  for k, name in enumerate(names1):
    gx = loadData(n1,n2,n3,fpath+name+"/",fname) 
    gma[:,:,:,k] = gx
  fx = goPredictSubs2(m1,m2,m3,gms,gma)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(path+"test21_kerry_cut.dat",format="%4")

def gofakedata(): 
  name0 = 'nx'
  names1 = ["ilSlope", "clSlope", "planarity", "envelope", "Grad_fz", "CVar", "DGVar", "DGSemb",
            "lwPosCur", "lwNegCur", "lwMaxCur", "lwMinCur", "lwCurvedness",
            "swPosCur", "swNegCur", "swMaxCur", "swMinCur", "swCurvedness",
            "ctrGLCM", "entrGLCM", "enerGLCM", "dissimiGLCM", "homoGLCM"]

  n1, n2, n3 = 128, 128, 128  # fake
  n4 = 23

  m1, m2, m3 = 128, 128, 128

  fpath = path_tr
  start_i = 380
  end_i = 400

  gms = np.zeros((n1, n2, n3, 1))
  gma = np.zeros((n1, n2, n3, n4))

  for i in range(start_i, end_i):
    fname = str(i) + '.dat'

    gx = loadData(n1, n2, n3, fpath + name0 + "/", fname) 
    gms[:,:,:,0] = gx
    for k, name in enumerate(names1):
      gx = loadData(n1, n2, n3, fpath + name + "/", fname) 
      gma[:,:,:,k] = gx
    fx = goPredictSubs2(m1, m2, m3, gms, gma)  # 进行预测 
    fx = np.transpose(fx)
    output_file = path + "test21_fake" + str(i) + ".dat"
    fx.tofile(output_file, format="%4")


def goPredictSubs2(m1,m2,m3,gms,gma): 
  n1,n2,n3,n4=gms.shape 
  nn1,nn2,nn3,nn4=gma.shape 
  p1,p2,p3=16,16,16 #overlap
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  c1=1+int(np.ceil(float(n1-m1)/(m1-p1)))
  c2=1+int(np.ceil(float(n2-m2)/(m2-p2)))
  c3=1+int(np.ceil(float(n3-m3)/(m3-p3)))
  for k3 in range(c3):
    for k2 in range(c2):
      for k1 in range(c1):
        gks = np.zeros((m1,m2,m3,n4),dtype=np.single)
        gka = np.zeros((m1,m2,m3,nn4),dtype=np.single)
        b1,b2,b3 = k1*(m1-p1),k2*(m2-p2),k3*(m3-p3)
        t1,t2,t3 = b1+p1,b2+p2,b3+p3
        e1,e2,e3 = b1+m1,b2+m2,b3+m3
        e1 = min(e1,n1)
        e2 = min(e2,n2)
        e3 = min(e3,n3)
        b1,b2,b3 = e1-m1,e2-m2,e3-m3

        gks[:,:,:,:] = gms[b1:e1,b2:e2,b3:e3,:]
        for ks in range(n4):
          gks[:,:,:,ks]=dataprocess(gks[:,:,:,ks])
        gks = np.reshape(gks,(1,m1,m2,m3,n4))

        gka[:,:,:,:] = gma[b1:e1,b2:e2,b3:e3,:]
        for ka in range(nn4):
          gka[:,:,:,ka]=dataprocess(gka[:,:,:,ka])

        gka2 = gka[::2,::2,::2,::1]
        gka4 = gka[::4,::4,::4,::1]
        gka8 = gka[::8,::8,::8,::1]
        gka16 = gka[::16,::16,::16,::1]
        l1,l2,l3,l4=gka2.shape 
        ll1,ll2,ll3,ll4=gka4.shape 
        lll1,lll2,lll3,lll4=gka8.shape 
        llll1,llll2,llll3,llll4=gka16.shape 
        gka = np.reshape(gka,(1,m1,m2,m3,nn4))
        gka2 = np.reshape(gka2,(1,l1,l2,l3,l4))
        gka4 = np.reshape(gka4,(1,ll1,ll2,ll3,ll4))
        gka8 = np.reshape(gka8,(1,lll1,lll2,lll3,lll4))
        gka16 = np.reshape(gka16,(1,llll1,llll2,llll3,llll4))
        fk = model.predict([gks,gka2,gka4,gka8,gka16],verbose=1) #fault prediction

        bt1,bt2,bt3=t1-b1,t2-b2,t3-b3
        t1 = min(int(bt1/2),b1)
        t2 = min(int(bt2/2),b2)
        t3 = min(int(bt3/2),b3)
        fx[b1+t1:e1,b2+t2:e2,b3+t3:e3] = fk[0,t1:,t2:,t3:,0]
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

if __name__ == '__main__':
    main()
