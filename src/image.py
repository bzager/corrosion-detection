# image.py 
# Ben Zager
# Image class for corrosion classifier
# 

import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2

from imageTools import *
from color import *
from texture import *


# loads an image or returns given image
def getImg(name,directory,maxsize,img):
	if img == None:
		return loadScaled(name,directory=directory,maxsize=maxsize)
	else:
		return img


class Image:

	def __init__(self,name,direc,group,maxsize=1000000,img=None):
		self.name = name # name of file
		self.group = group # pos or test
		self.rgb = getImg(name,direc,maxsize,img)
		self.hsv = toHSV(self.rgb,form="sk")
		self.gray = toGray(self.rgb)

		self.shape = self.gray.shape # (n,m)
		self.size = self.gray.size # n*m

		self.mask = None   # mask for non-red pixels in color format (n,m,3)
		self.gmask = None  # mask for non-red pixels in gray format (n,m)
		self.masked = None # masked rgb image
		self.redBlocks = [] # blocks containing a given proportion of red pixels
		self.redInds = [] # i,j indices of red blocks
		
		self.hsHist = None # hue,saturation histogram
		self.hsBins = []   # [xedges,yedges]
		self.hsComp = None # comparison with template histogram

		self.edges = None # edge image from canny edge detection
		self.edgeness = None # array of edge densities for each block
		self.edgeHist = None # histogram of edge densities
		self.edgeShape = []  # center and width for plotting edge density hist
		self.edgeComp = None # comparison of edge density hist with template

		self.edgeMap = None # image with blocks color coded by edge density

		#self.cstats = []
		#self.tstats = []
		#self.features = [] # feature vector
		
		self.label = 1 # label 1=rusted, 0=non-rusted
		self.actual = getActual(group,name)

		self.title = "" # descriptive label for plotting

	# reinitialize image
	def reInit(self,maxsize):
		self.rgb = scaleDown(self.full,maxsize)
		self.hsv = toHSV(self.rgb,form="cv2")
		self.gray = toGray(self.rgb)
		self.shape = self.gray.shape
		self.size = self.gray.size


###########################################################
########################## COLOR ##########################
###########################################################

	# initialize color mask, gray mask, and masked rgb image
	def setRed(self,minPix=64):
		self.mask = getRed(self.rgb,minPix=minPix)
		self.gmask = grayMask(self.mask)
		self.masked = getMasked(self.rgb,self.mask)

	# initialize red blocks
	def setRedBlocks(self,d,blockThr):
		self.redBlocks,self.redInds = getRedBlocks(self.rgb,self.mask,d,blockThr)

	# calculates 2d histogram in HS space
	def setHSHist(self,nbins=(45,64)):
		self.hsHist,xedges,yedges = hist2d(self.hsv,self.mask,nbins=nbins)
		self.hsBins.append(xedges); self.hsBins.append(yedges)

	# initialize color features
	def initColor(self,d=8,blockThr=0.9,minPix=64,nbins=(45,64)):
		self.setRed(minPix=minPix)
		self.setRedBlocks(d,blockThr)
		self.setHSHist(nbins=nbins)

	# compare HS hist to given hist
	def compareHSHist(self,hist):
		if self.hsHist != None and hist != None:
			self.hsComp = compareHist(self.hsHist,hist)

	"""
	# first 4 moments of hue and saturation
	# 8 features
	def setColorStats(self):
		h,s,v = splitComp(self.hsv)
		hue = h[self.gmask]; sat = s[self.gmask]
		self.cstats = getStats(hue)+getStats(sat)
	"""

###########################################################
######################### TEXTURE #########################
###########################################################
	
	# initialize edge features
	# locate edges, calculate edge densities, build edge hist
	def setEdges(self,nbins=32,sig=0.1):
		if self.label:
			self.edges = edges(self.gray,mask=self.gmask,sigma=sig)
			self.edgeness = edgenessTotal(self.edges,self.redInds,self.redBlocks[0].shape[0])
			self.edgeHist,cen,wid = hist(self.edgeness,nbins=nbins,range=(0,0.5))
			self.edgeShape.append(cen); self.edgeShape.append(wid)

	# compare edge hist to given hist
	def compareEdgeHist(self,hist):
		if self.edgeHist != None and hist != None:
			self.edgeComp = compareHist(self.edgeHist,hist)
		else:
			self.edgeComp = (np.nan,np.nan,np.nan)

	# label blocks by edge density
	def setEdgeMap(self):
		if self.label:
			d = self.redBlocks[0].shape[0]
			vals = self.edgeness
			self.edgeMap = drawBlocksColorCoded(self.rgb,self.redInds,d,vals)
		else:
			self.edgeMap = self.rgb

	# initialize texture features
	def initTexture(self,d=8,nbins=32,sig=0.1):
		self.setEdges(nbins=nbins,sig=sig)
		self.setEdgeMap()
		
		#if self.label: self.setTextureStats()
		#else: self.tstats = [np.nan for i in range(8)]
	"""
	#
	def setTextureStats(self):
		self.tstats = textureStats(self.rgb,self.gray,self.mask,self.gmask,self.redBlocks,self.redInds)
	"""

###########################################################
######################### DISPLAY #########################
###########################################################
	
	# returns random list of redBlocks
	def randBlocks(self,num):
		idx = np.random.randint(0,len(self.redBlocks),size=num)
		blocks = [self.redBlocks[i] for i in idx]
		inds = [self.redInds[i] for i in idx]
		return blocks,inds

	# title for displaying image w/ some basic stats
	def setTitle(self):
		numBlocks = "nb: "+str(len(self.redBlocks))
		label = " Label: "+str(self.label)
		status = " Actual: "+str(self.actual)
		hsComp = "\nHS=("+list2str(self.hsComp)+")"
		edgeComp = " Edge=("+list2str(self.edgeComp)+")"
		self.title = self.name+"\n"+numBlocks+label+status+hsComp+edgeComp

	# builds stacked image of using:
	# [rgb, masked, edges, edgeMap]
	def getStacked(self,method="square"):
		stacked = skimage.util.img_as_ubyte(np.stack([self.edges,self.edges,self.edges],axis=-1))
		imgs = [self.rgb,self.masked,stacked,self.edgeMap]

		return stack4(imgs,method=method)

	# display HS and edge histograms side by side
	def histDisplay(self):
		h2 = self.hsHist
		xedges = self.hsBins[0]; yedges = self.hsBins[1]
		
		fig,ax = plt.subplots(nrows=1,ncols=2)
		im = ax[0].imshow(h2.T,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='lower',interpolation='nearest')
		#ax[0].set_xlim([0,1]); ax[0].set_ylim([0,1]);
		ax[0].set_xlabel("Hue"); ax[0].set_ylabel("Saturation")
		ax[0].set_title(list2str(self.hsComp))

		if self.label:
			h = self.edgeHist
			cen = self.edgeShape[0]; wid = self.edgeShape[1]
			ax[1].bar(cen,h,wid)
			ax[1].set_title(list2str(self.edgeComp))
			ax[1].set_xlim([cen[0]-0.5*wid,cen[-1]+0.5*wid])


###########################################################
###################### CLASSIFICATION #####################
###########################################################
	"""
	#
	def setFeatures(self):
		self.features = np.asarray(self.cstats+self.tstats)
	"""

	# determines if image is correctly classified
	def isCorrect(self):
		if self.actual == None:
			return None
		return self.label == self.actual


