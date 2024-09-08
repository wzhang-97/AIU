#############################################################################
"""
Demo of demostrating fault results
Author: Xinming Wu, USTC
Version: 2020.03.26
"""
from datUtils import * 
#############################################################################

plotOnly = True
plotOnly = False

def main(args):
  if args[1]=="f3Full":
    goF3Full()
  elif args[1]=="xin":
    goXin()
  elif args[1]=="nz":
    #goNz()
    goInterp()
  elif args[1]=="z3":
    goZ3()
  elif args[1]=="d2":
    goD2()
  elif args[1]=="dy":
    goDy()
    #goInterpX()
  elif args[1]=="bgp":
    goBgp()
    #goInterp()
  elif args[1]=="pttep":
    goPttep()
    #goInterp()
  elif args[1]=="crf":
    goCrf()
    #goInterp()
  elif args[1]=="cnpc":
    goCnpc()
    #goInterpCnpc()
  else:
    print "demo not found"

def goPttep():
  gxfile = "sx" # input seismic image
  fpfile = "fpx" # fault probability by cnn
  setupForSubset("pttep")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")
def goCnpc():
  gxfile = "sx" # input seismic image
  fpfile = "fpi" # fault probability by cnn
  setupForSubset("cnpc")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  fp = readImage(fpfile)
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")

def goCrf():
  gxfile = "gxl" # input seismic image
  fpfile = "fpx" # fault probability by cnn
  setupForSubset("crf")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  '''
  fp = readImage(fpfile)
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")
  '''

def goBgp():
  gxfile = "gx" # input seismic image
  fpfile = "fpi" # fault probability by cnn
  setupForSubset("bgp")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")

def goDy():
  gxfile = "gx" # input seismic image
  fpfile = "fpx" # fault probability by cnn
  setupForSubset("dy")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  #plot3(s1,s2,s3,fp,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")

def goZ3():
  gxfile = "gx" # input seismic image
  fpfile = "fp" # fault probability by cnn
  setupForSubset("Z3")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")

def goNz():
  gxfile = "gr" # input seismic image
  fpfile = "fpr" # fault probability by cnn
  setupForSubset("NZ")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")
def goD2():
  gxfile = "gx" # input seismic image
  fpfile = "fpx" # fault probability by cnn
  setupForSubset("D2")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")

def goXin():
  gxfile = "gr" # input seismic image
  fpfile = "fpr" # fault probability by cnn
  setupForSubset("xin")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")

def goInterpCnpc():
  setupForSubset("cnpc")
  fpfile = "fp" # fault probability by cnn
  fp = readImage(fpfile)
  s1,s2,s3 = getSamplings()
  n2,n3=s2.count,s3.count
  m1 = 3000
  c1 = Sampling(m1,0.5,0)
  fpi = zerofloat(m1,n2,n3)
  si = SincInterpolator()
  for i3 in range(n3):
    for i2 in range(n2):
      si.interpolate(s1,fp[i3][i2],c1,fpi[i3][i2])
  fpi = clip(0,1,fpi)
  fpi = sub(fpi,min(fpi))
  fpi = div(fpi,max(fpi))
  writeImage('fpi',fpi)

def goInterp():
  setupForSubset("bgp")
  fpfile = "fpr" # fault probability by cnn
  fp = readImage(fpfile)
  s1,s2,s3 = getSamplings()
  n2,n3=s2.count,s3.count
  m1 = 2501
  c1 = Sampling(m1,0.5,0)
  fpi = zerofloat(m1,n2,n3)
  si = SincInterpolator()
  for i3 in range(n3):
    for i2 in range(n2):
      si.interpolate(s1,fp[i3][i2],c1,fpi[i3][i2])
  fpi = clip(0,1,fpi)
  fpi = sub(fpi,min(fpi))
  fpi = div(fpi,max(fpi))
  writeImage('fpi',fpi)

def goInterpX():
  setupForSubset("dy")
  fpfile = "fpr2" # fault probability by cnn
  fp = readImage(fpfile)
  s1,s2,s3 = getSamplings()
  m1,m2,m3 = 1251,701,801
  c1 = Sampling(m1,0.5,0)
  c2 = Sampling(m2,0.5,0)
  c3 = Sampling(m3,0.5,0)
  fpi = zerofloat(m1,m2,m3)
  si = SincInterpolator()
  for i3 in range(m3):
    print i3
    x3 = c3.getValue(i3)
    for i2 in range(m2):
      x2 = c2.getValue(i2)
      for i1 in range(m1):
        x1 = c1.getValue(i1)
        fpi[i3][i2][i1]=si.interpolate(s1,s2,s3,fp,x1,x2,x3)
  fpi = clip(0,1,fpi)
  fpi = sub(fpi,min(fpi))
  fpi = div(fpi,max(fpi))
  writeImage('fpi',fpi)


def goClyde():
  gxfile = "gxl" # input seismic image
  fpfile = "fp" # fault probability by cnn
  setupForSubset("clyde")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  ref = RecursiveGaussianFilter(1)
  ref.apply000(gx,gx)
  writeImage("gxs",gx)
  fp = readImage(fpfile)
  ks = [99,38,29]
  vt=[-0.05,-0.03,0.0]
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=[45,40],vt=vt,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=[45,40],vt=vt,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")
def goPTTEP():
  gxfile = "sx" # input seismic image
  fpfile = "fp" # fault probability by cnn
  setupForSubset("pttep")
  global pngDir
  pngDir = getPngDir()
  s1,s2,s3 = getSamplings()
  gx = readImage(gxfile)
  fp = readImage(fpfile)
  #gx = gain(gx)
  plot3(s1,s2,s3,gx,cmin=-2,cmax=2,clab="Amplitude",png="gx")
  plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=0.8,cmap=jetRamp(1.0),
          clab="Fault probability",png="fp")


def gain(x):
  g = mul(x,x) 
  ref = RecursiveExponentialFilter(20.0)
  ref.apply1(g,g)
  div(x,sqrt(g),x)
  return x

def checkNaN(gx):
  n3 = len(gx)
  n2 = len(gx[0])
  n1 = len(gx[0][0])
  for i3 in range(n3):
    for i2 in range(n2):
      for i1 in range(n1):
        #if(gx[i3][i2][i1]!=gx[i3][i2][i1]):
        if(abs(gx[i3][i2][i1])>10000000000000):
          gx[i3][i2][i1] = 0
  return gx

def smooth(sig,u):
  v = copy(u)
  rgf = RecursiveGaussianFilterP(sig)
  rgf.apply0(u,v)
  return v

def smooth2(sig1,sig2,u):
  v = copy(u)
  rgf1 = RecursiveGaussianFilterP(sig1)
  rgf2 = RecursiveGaussianFilterP(sig2)
  rgf1.apply0X(u,v)
  rgf2.applyX0(v,v)
  return v

def normalize(e):
  emin = min(e)
  emax = max(e)
  return mul(sub(e,emin),1.0/(emax-emin))

def slice12(k3,f):
  n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
  s = zerofloat(n1,n2)
  SimpleFloat3(f).get12(n1,n2,0,0,k3,s)
  return s

def slice13(k2,f):
  n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
  s = zerofloat(n1,n3)
  SimpleFloat3(f).get13(n1,n3,0,k2,0,s)
  return s

def slice23(k1,f):
  n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
  s = zerofloat(n2,n3)
  SimpleFloat3(f).get23(n2,n3,k1,0,0,s)
  return s

#############################################################################
# graphics

def jetFill(alpha):
  return ColorMap.setAlpha(ColorMap.JET,alpha)
def jetFillExceptMin(alpha):
  a = fillfloat(alpha,256)
  a[0] = 0.0
  return ColorMap.setAlpha(ColorMap.JET,a)
def jetRamp(alpha):
  return ColorMap.setAlpha(ColorMap.JET,rampfloat(0.0,alpha/256,256))
def bwrFill(alpha):
  return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,alpha)
def bwrNotch(alpha):
  a = zerofloat(256)
  for i in range(len(a)):
    if i<128:
      a[i] = alpha*(128.0-i)/128.0
    else:
      a[i] = alpha*(i-127.0)/128.0
  return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,a)
def hueFill(alpha):
  return ColorMap.getHue(0.0,1.0,alpha)
def hueFillExceptMin(alpha):
  a = fillfloat(alpha,256)
  a[0] = 0.0
  return ColorMap.setAlpha(ColorMap.getHue(0.0,1.0),a)

def addColorBar(frame,clab=None,cint=None):
  cbar = ColorBar(clab)
  if cint:
    cbar.setInterval(cint)
  cbar.setFont(Font("Arial",Font.PLAIN,32)) # size by experimenting
  cbar.setWidthMinimum
  cbar.setBackground(Color.WHITE)
  frame.add(cbar,BorderLayout.EAST)
  return cbar

def convertDips(ft):
  return FaultScanner.convertDips(0.2,ft) # 5:1 vertical exaggeration

def plot2(s1,s2,x,u=None,g=None,x1=None,c=None,
        cmap=ColorMap.GRAY,clab="Amplitude",
        cmin=-2,cmax=2,title=None,png=None):
  sp = SimplePlot(SimplePlot.Origin.UPPER_LEFT)
  if title:
    sp.setTitle(title)
  n1,n2=s1.count,s2.count
  sp.addColorBar(clab)
  #sp.setSize(955,400)
  sp.setSize(755,500)
  sp.setHLabel("Inline (sample)")
  sp.setVLabel("Depth (sample)")
  sp.plotPanel.setColorBarWidthMinimum(60)
  sp.setVLimits(0,n1-1)
  sp.setHLimits(0,n2-1)
  sp.setFontSize(16)
  pv = sp.addPixels(s1,s2,x)
  pv.setColorModel(cmap)
  pv.setInterpolation(PixelsView.Interpolation.LINEAR)
  if cmin<cmax:
    pv.setClips(cmin,cmax)
  if u:
    cv = sp.addContours(s1,s2,u)
    cv.setContours(80)
    cv.setLineColor(Color.YELLOW)
  if g:
    pv = sp.addPixels(s1,s2,g)
    pv.setInterpolation(PixelsView.Interpolation.NEAREST)
    pv.setColorModel(ColorMap.getJet(0.8))
    pv.setClips(0.1,s1.count)
  if x1:
    x1k = zerofloat(n2)
    x2  = zerofloat(n2)
    x1s  = zerofloat(n1)
    for i1 in range(n1):
      x1s[i1] = i1
    cp = ColorMap(0,n1,ColorMap.JET)
    rgb = cp.getRgbFloats(x1s)
    ref = RecursiveExponentialFilter(1)
    for k in range(20,n1-20,15):
      for i2 in range(n2):
        x2[i2] = i2
        x1k[i2] = x1[i2][k]
      ref.apply(x1k,x1k)
      pv = sp.addPoints(x1k,x2)
      pv.setLineWidth(2.5)
      r,g,b=rgb[k*3],rgb[k*3+1],rgb[k*3+2]
      pv.setLineColor(Color(r,g,b))
  if pngDir and png:
    sp.paintToPng(700,3.333,pngDir+png+".png")

def plot3(s1,s2,s3,f,g=None,cmin=-2,cmax=2,zs=1.0,sc=1.4,
        ks=[175,330,377],ae=[45,35],vt=[-0.1,-0.06,0.0],
        cmap=None,clab=None,cint=None,surf=None,png=None):
  n3 = len(f)
  n2 = len(f[0])
  n1 = len(f[0][0])
  d1,d2,d3 = s1.delta,s2.delta,s3.delta
  f1,f2,f3 = s1.first,s2.first,s3.first
  l1,l2,l3 = s1.last,s2.last,s3.last
  sf = SimpleFrame(AxesOrientation.XRIGHT_YOUT_ZDOWN)
  cbar = None
  if g==None:
    ipg = sf.addImagePanels(s1,s2,s3,f)
    if cmap!=None:
      ipg.setColorModel(cmap)
    if cmin!=None and cmax!=None:
      ipg.setClips(cmin,cmax)
    else:
      ipg.setClips(-2.0,2.0)
    if clab:
      cbar = addColorBar(sf,clab,cint)
      ipg.addColorMapListener(cbar)
  else:
    ipg = ImagePanelGroup2(s1,s2,s3,f,g)
    ipg.setClips1(-2,2)
    if cmin!=None and cmax!=None:
      ipg.setClips2(cmin,cmax)
    if cmap==None:
      cmap = jetFill(0.8)
    ipg.setColorModel2(cmap)
    if clab:
      cbar = addColorBar(sf,clab,cint)
      ipg.addColorMap2Listener(cbar)
    sf.world.addChild(ipg)
  if cbar:
    cbar.setWidthMinimum(120)
  if surf:
    tg = TriangleGroup(True,surf)
    sf.world.addChild(tg)
  ipg.setSlices(ks[0],ks[1],ks[2])
  if cbar:
    sf.setSize(987,720)
  else:
    sf.setSize(850,720)
  vc = sf.getViewCanvas()
  vc.setBackground(Color.WHITE)
  radius = 0.5*sqrt(n1*n1+n2*n2+n3*n3)
  ov = sf.getOrbitView()
  zscale = zs*max(n2*d2,n3*d3)/(n1*d1)
  ov.setAxesScale(1.0,1.0,1.0)
  ov.setScale(sc)
  ov.setWorldSphere(BoundingSphere(BoundingBox(f3,f2,f1,l3,l2,l1)))
  ov.setTranslate(Vector3(vt[0],vt[1],vt[2]))
  ov.setAzimuthAndElevation(ae[0],ae[1])
  sf.setVisible(True)
  if png and pngDir:
    sf.paintToFile(pngDir+png+".png")
    if cbar:
      cbar.paintToPng(720,1,pngDir+png+"cbar.png")
#############################################################################
# Run the function main on the Swing thread
import sys
class _RunMain(Runnable):
  def __init__(self,main):
    self.main = main
  def run(self):
    self.main(sys.argv)
def run(main):
  SwingUtilities.invokeLater(_RunMain(main)) 
run(main)
