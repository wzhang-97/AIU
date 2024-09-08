"""
Jython utilities for cig.
Author: Xinming Wu, USTC
Version: 2020.03.28
"""
from common import *

#############################################################################
# Internal constants

_datdir = "/media/xinwu/disk-2/cig-cnpc/data/"
_pngdir = "/media/xinwu/disk-2/cig-cnpc/png/"
_datdir = "/media/xinwu/disk-2/cig-sl/data/"
_pngdir = "/media/xinwu/disk-2/cig-sl/png/"
#_pngdir = "../png/"
#_datdir = "../data/"

#############################################################################
# Setup

def setupForSubset(name):
  """
  Setup for a specified directory includes:
    seismic directory
    samplings s1,s2
  Example: setupForSubset("pnz")
  """
  global pngDir
  global seismicDir
  global s1,s2,s3
  global n1,n2,n3
  if name=="f3d":
    """ gather """
    print "setupForSubset: gather"
    pngDir = _pngdir+"f3d/"
    seismicDir = _datdir+"prediction/f3d/"
    n1,n2,n3 = 128,384,512
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="Z3":
    print "setupForSubset: Z3"
    pngDir = _pngdir+"z3/"
    seismicDir = _datdir+"prediction/z3/"
    n1,n2,n3 = 251,901,1592
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="NZ":
    print "setupForSubset: NZ"
    pngDir = _pngdir+"nz/"
    seismicDir = _datdir+"prediction/nz/"
    n1,n2,n3 = 1501,441,601
    n1,n2,n3 = 750,441,601
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="D2":
    print "setupForSubset: D2"
    pngDir = _pngdir+"d2/"
    seismicDir = _datdir+"prediction/d2/"
    n1,n2,n3 = 126,771,991
    n1,n2,n3 = 251,771,991
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="cnpc":
    print "setupForSubset: cnpc"
    pngDir = _pngdir
    seismicDir = _datdir+"seis/"
    n1,n2,n3 = 1500,968,611
    n1,n2,n3 = 3000,968,611
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)

  elif name=="xin":
    """ xin """
    print "setupForSubset: xin"
    pngDir = _pngdir+"xin/"
    seismicDir = _datdir+"prediction/xin/"
    n1,n2,n3 = 1856,231,311
    n1,n2,n3 = 2800,231,311
    n1,n2,n3 = 1400,231,311
    #n1,n2,n3 = 933,231,311
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="dy":
    """ pttep """
    print "setupForSubset: dy"
    pngDir = _pngdir+"dy/"
    seismicDir = _datdir+"prediction/dy/"
    n1,n2,n3 = 626,350,400
    n1,n2,n3 = 1251,701,801
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="bgp":
    """ bgp """
    print "setupForSubset: bgp"
    pngDir = _pngdir+"bgp/"
    seismicDir = _datdir+"prediction/bgp/"
    n1,n2,n3 = 1250,240,140
    n1,n2,n3 = 1250,481,281
    n1,n2,n3 = 2501,481,281
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="pttep":
    """ pttep """
    print "setupForSubset: pttep"
    pngDir = _pngdir+"pttep/"
    seismicDir = _datdir+"prediction/pttep/"
    n1,n2,n3 = 1250,240,140
    n1,n2,n3 = 1250,481,281
    n1,n2,n3 = 750,1193,863
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="crf":
    """ crf """
    print "setupForSubset: crf"
    pngDir = _pngdir+"crf/"
    seismicDir = _datdir+"prediction/crf/"
    n1,n2,n3 = 601,3675,825
    d1,d2,d3 = 1,1,1 # (s,km/s)
    f1,f2,f3 = 0,0,0
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="train":
    print "setupForSubset: train"
    pngDir = _pngdir+"train/"
    seismicDir = _datdir+"train/"
    n1,n2,n3 = 128,128,128
    d1,d2,d3 = 1.0,1.0,1.0 
    f1,f2,f3 = 0.0,0.0,0.0 # = 0.000,0.000,0.000
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  elif name=="validation":
    print "setupForSubset: validation"
    pngDir = _pngdir+"validation/"
    seismicDir = _datdir+"validation/"
    n1,n2,n3 = 128,128,128
    d1,d2,d3 = 1.0,1.0,1.0 
    f1,f2,f3 = 0.0,0.0,0.0 # = 0.000,0.000,0.000
    s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
  else:
    print "unrecognized subset:",name
    System.exit

def getSamplings():
  return s1,s2,s3

def getDataShape():
  return n1,n2,n3

def getSeismicDir():
  return seismicDir

def getPngDir():
  return pngDir

#############################################################################
# read/write images
def readImageChannels(basename):
  """ 
  Reads three channels of a color image
  """
  fileName = seismicDir+basename+".jpg"
  il = ImageLoader()
  image = il.readThreeChannels(fileName)
  return image
def readColorImage(basename):
  """ 
  Reads three channels of a color image
  """
  fileName = seismicDir+basename+".jpg"
  il = ImageLoader()
  image = il.readColorImage(fileName)
  return image

def readImage2D(n1,n2,basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def readImageX(m1,m2,m3,basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(m1,m2,m3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImage(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImage3DB(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName,ByteOrder.BIG_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def readImageL(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2,n3)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readInts(image)
  ais.close()
  return image

def readImage2DL(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1,n2)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image


def readImage1D(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1)
  ais = ArrayInputStream(fileName)
  ais.readFloats(image)
  ais.close()
  return image

def readImage1L(basename):
  """ 
  Reads an image from a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  image = zerofloat(n1)
  ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  ais.readFloats(image)
  ais.close()
  return image

def writeImage(basename,image):
  """ 
  Writes an image to a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  #aos.writeBytes(image)
  aos.close()
  return image

def writeImagex(fname,image):
  """ 
  Writes an image to a file with specified basename
  """
  fileName = fname+".dat"
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  #aos.writeBytes(image)
  aos.close()
  return image


def writeImageL(basename,image):
  """ 
  Writes an image to a file with specified basename
  """
  fileName = seismicDir+basename+".dat"
  print fileName
  aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
  aos.writeFloats(image)
  aos.close()
  return image

#############################################################################
# read/write fault skins

def skinName(basename,index):
  return basename+("%05i"%(index))
def skinIndex(basename,fileName):
  assert fileName.startswith(basename)
  i = len(basename)
  return int(fileName[i:i+5])

def listAllSkinFiles(basename):
  """ Lists all skins with specified basename, sorted by index. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  return fileNames

def removeAllSkinFiles(basename):
  """ Removes all skins with specified basename. """
  fileNames = listAllSkinFiles(basename)
  for fileName in fileNames:
    File(seismicDir+fileName).delete()

def readSkin(basename,index):
  """ Reads one skin with specified basename and index. """
  return FaultSkin.readFromFile(seismicDir+skinName(basename,index)+".dat")

def getSkinFileNames(basename):
  """ Reads all skins with specified basename. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  return fileNames

def readSkins(basename):
  """ Reads all skins with specified basename. """
  fileNames = []
  for fileName in File(seismicDir).list():
    if fileName.startswith(basename):
      fileNames.append(fileName)
  fileNames.sort()
  skins = []
  for iskin,fileName in enumerate(fileNames):
    index = skinIndex(basename,fileName)
    skin = readSkin(basename,index)
    skins.append(skin)
  return skins

def writeSkin(basename,index,skin):
  """ Writes one skin with specified basename and index. """
  FaultSkin.writeToFile(seismicDir+skinName(basename,index)+".dat",skin)

def writeSkins(basename,skins):
  """ Writes all skins with specified basename. """
  for index,skin in enumerate(skins):
    writeSkin(basename,index,skin)

from org.python.util import PythonObjectInputStream
def readObject(name):
  fis = FileInputStream(seismicDir+name+".dat")
  ois = PythonObjectInputStream(fis)
  obj = ois.readObject()
  ois.close()
  return obj
def writeObject(name,obj):
  fos = FileOutputStream(seismicDir+name+".dat")
  oos = ObjectOutputStream(fos)
  oos.writeObject(obj)
  oos.close()
