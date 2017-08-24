# color.py
# Ben Zager
# Functions for color feature extraction


import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt

from imageTools import *
from skimage import morphology


np.set_printoptions(precision=3,suppress=True)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 8

######################################################
################ RED PIXEL EXTRACTION ################
######################################################

# defines range of HSV space to extract red pixels
def redRange():
	#lr = np.array([0,50,50]) # original
	#ur = np.array([11,255,255])

	lr = np.array([2,60,75]) # lower H,S,V
	ur = np.array([12,250,250]) # upper H,S,V

	return lr,ur

# masks pixels of img outside redRange
def redMask(img):
	hsv = toHSV(img,form="cv2")
	lower_red,upper_red = redRange()
	
	return cv2.inRange(hsv,lower_red,upper_red)

# Counts number of pixels left after masking those outside redRange
# returns true if proportion of remaining pixels > thresh
# otherwise returns false 
# [Petricca et al.]
def redFilter(img,thresh):
	mask = redMask(img)
	ret,maskbin = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

	height,width = maskbin.shape
	percent = cv2.countNonZero(maskbin)/float(height*width)
	
	return percent > thresh

# gets list of blocks containing at least blockThr proportion of red pixels
# also gets corresponding indices from original image of top left pixel of blocks
def getRedBlocks(img,mask,d,blockThr,useMasked=False):
	redBlocks = []; redInds = []
	
	if useMasked:
		blocks,inds = getBlocks(getMasked(img,mask),d)
	else:
		blocks,inds = getBlocks(img,d)
	
	for i in range(len(blocks)):
		bmask = getBlockMask(mask,inds[i],d)
		prop = np.sum(bmask[:,:,0]) / d**2
		
		if prop > blockThr and bmask.shape[:2] == (d,d):
			redBlocks.append(blocks[i])
			redInds.append(inds[i])
	
	return redBlocks,redInds

# cleans mask by removing isolated masked or unmasked regions with < minPix pixels
# followed by morphological closing using disk structuring element of given radius
def cleanMask(mask,minPix,radius=4):
	mask = morphology.remove_small_objects(label(mask)[0],min_size=minPix)
	mask = morphology.remove_small_holes(label(mask)[0],min_size=minPix) 
	
	selem = morphology.disk(radius)
	mask = morphology.binary_closing(mask,selem=selem)

	return mask

# gets mask for non-red pixels in image
# uses color image format (n,m,3)
def getRed(img,minPix=64,clean=True):
	mask = redMask(img)
	if clean: mask = cleanMask(mask,minPix)
	
	return np.stack([mask,mask,mask],axis=-1).astype(bool)

# converts HSV format from float to byte, or vice versa
def convertHSVType(img):
	ker = np.ones([img.shape[0],img.shape[1],3])
	ker[:,:,0] = 255; ker[:,:,0] = 255; ker[:,:,0] = 255

	if np.amax(dataRange(img)) == 1:
		return np.multiply(img,ker)
	elif np.amax(dataRange(img)) == 255: 
		return np.divide(img,ker)

	return img


#############################################################
###################### COLOR STATISTICS #####################
#############################################################

# calculates the 1st 4 moments of a component of an hsv image
def colorStats(img,mask,param=0):
	hsv = toHSV(img)
	par = hsv[:,:,param]
	data = par[mask[:,:,0]]
	
	if data.size == 0:
		return ()
	
	stat = sp.stats.describe(data)
	mean,var,skew,kurt,minim,maxim = stat[2],stat[3],stat[4],stat[5],stat[1][0],stat[1][1]
	med = np.median(data)
	 
	return mean,var,skew,kurt #,minim,maxim
	
# gets colorStats for hue and saturation components of image
def hsStats(img,mask):
	
	hue = list(colorStats(img,mask,param=0))
	sat = list(colorStats(img,mask,param=1))
	
	return np.asarray(hue+sat)


# plots masked image, 2d HS histogram, hue histogram, saturation histogram
def plotImgHist2d(img,mask,nbins=(180,256),dim=2):
	
	h,xedges,yedges = hist2d(toHSV(img),mask,nbins=nbins)
	
	fig = plt.figure(figsize=plt.figaspect(0.5))
	ax = fig.add_subplot(1,2,1)
	ax.imshow(getMasked(img,mask)) # masked image
	ax.set_xticks([]); ax.set_yticks([])

	if dim == 2:
		histPlot2d(h,xedges,yedges,fig,title=list2str(hsStats(img,mask)))
	elif dim == 3:
		histPlot3d(h,xedges,yedges,fig,title=list2str(hsStats(img,mask)))
	
	return h

# plots bounding box on HS histogram showing bounds of redRange
def plotBounds(ax):
	lr,ur = redRange()
	minS = lr[1]/255
	maxS = ur[1]/255
	minH = lr[0]/179
	maxH = ur[0]/179

	wid = 1; col = "w"
	ax.axhline(minS,minH,maxH,linewidth=wid,color=col) # saturation
	ax.axhline(maxS,minH,maxH,linewidth=wid,color=col)
	ax.axvline(minH,minS,maxS,linewidth=wid,color=col) # hue
	ax.axvline(maxH,minS,maxS,linewidth=wid,color=col)

# plots a 2d histogram
def histPlot2d(h,xedges,yedges,fig,title="",showBounds=True):
	ax = fig.add_subplot(1,2,2)
	im = ax.imshow(h.T,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='lower',interpolation='nearest')
	#ax.set_xlim([0,1]); ax.set_ylim([0,1]);
	fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
	ax.set_xlabel("Hue"); ax.set_ylabel("Saturation")
	ax.set_title(title)
	if showBounds:
		plotBounds(ax)

# plots a 2d histogram in 3d
def histPlot3d(h,xedges,yedges,fig,title=""):
	ax = fig.add_subplot(1,1,1,projection='3d')
	#x = (xedges[:-1]+xedges[1:]) / 2
	#y = (yedges[:-1]+yedges[1:]) / 2

	maxH = int(h.shape[0]/5)

	x = np.linspace(0,1,h.shape[0])[:maxH]
	y = np.linspace(0,1,h.shape[1])
	X,Y = np.meshgrid(x,y)

	ax.plot_surface(X,Y,h[:maxH,:].T,cmap=plt.cm.plasma)
	ax.set_xlabel("Hue"); ax.set_ylabel("Saturation")
	ax.set_zlim([0,1.05*np.amax(h)])
	ax.set_title(title)


		