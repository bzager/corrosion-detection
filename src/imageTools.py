# imageTools.py
# Ben Zager
# image processing functions for corrosion detection
# mostly wrappers around scikit-image and opencv
# used by color.py,texture.py,image.py,classifier.py
# 


import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage
import cv2

from skimage import data,color

np.set_printoptions(precision=3,suppress=True)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 8


#############################################################
############################ I/O ############################
#############################################################

# loads rgb image from given directory
def load(name,directory=""):
	#return data.imread("../data/"+directory+"/"+name)
	return cv2.cvtColor(cv2.imread("../data/"+directory+"/"+name),cv2.COLOR_BGR2RGB)

# saves image to given directory as given name
def save(name,directory,img):
	path = "../data/"+directory+"/"+name
	skimage.io.imsave(path,img)
	
# loads image, scaling it to below max number of pixels
def loadScaled(name,directory="",maxsize=1000000):
	file = load(name,directory)
	return scaleDown(file,maxsize)

# scales an image down to below maxsize pixels
def scaleDown(img,maxsize):
	size = img[:,:,0].size
		
	if maxsize != None and size > maxsize:
		scale = np.sqrt(maxsize/float(size))
		rescaled = rescale(img,scale)
		return pad(rescaled)
	
	return pad(img)


# loads all images from a directory
def loadAll(directory="",maxsize=1000000):
	files = []
	
	for fname in os.listdir("../data/"+directory):
		if fname==".DS_Store":
			continue 
		#files.append(Image(fname,directory))     
		files.append(loadScaled(fname,directory,maxsize=maxsize))
	
	return files

#############################################################
################## UTILITIES/PREPROCESSING ##################
#############################################################

# returns tuple of image intensity limits
# should be either (0,1) or (0,255)
def dataRange(img):
	return skimage.util.dtype_limits(img)

# rescales image by a given size
def rescale(img,scale):
	scaled = skimage.transform.rescale(img,scale,mode='reflect',preserve_range=True)
	return scaled.astype(np.ubyte)

# pads an image with zeros s.t.
# each image dimension is evenly divisible by corresponding dimension
def pad(arr,req=(8,8)):
	cur = np.asarray(arr.shape[0:2])
	req = np.asarray(req)
	over = np.mod(cur,req)
	add = np.mod(req-over,req)
	width = ((0,add[0]),(0,add[1]),(0,0))
	return skimage.util.pad(arr,tuple(width),'edge')

# adjusts image intensities to enhance contrast
def equalize(img):
	eq = skimage.exposure.equalize_hist(img,nbins=256)
	#eq = skimage.exposure.adjust_sigmoid(img,cutoff=0.5,gain=5,inv=False)
	#eq = skimage.exposure.equalize_adapthist(img,kernel_size=None,clip_limit=0.01)
	return skimage.util.img_as_ubyte(eq)
	
# equalizes all images in list
def equalizeAll(imgs):
	eq = []
	for im in imgs:
		eq.append(equalize(im))

	return eq

# adds noise to image
# modes: gaussian,localvar,poisson,salt,pepper,s&p,speckle
def addNoise(img,mode="gaussian"):
	return toByte(skimage.util.random_noise(img,mode=mode))

# gets label from image
# assumes images are labeled with 0 or 1 as last character before file extension
# 0 -> non-corroded, 1 -> corroded
def getLabel(name):
	lab = name[-5]

	if lab == "1" or lab == "0":
		return int(lab)
	else:
		return None

# gets the label for a test image or training image
def getActual(group,name):
	if group == "pos":
		return 1
	elif group == "test":
		return getLabel(name)

########################################################
###################### CONVERSION ######################
########################################################

# convert rgb to hsv
# sk: float format -> H[0-1],S[0-1],V[0-1]
# cv2: integer format -> H[0-179],S[0-255],V[0-255]
def toHSV(rgb,form="sk"):
	if form == "sk":
		return skimage.util.img_as_float(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
	elif form == "cv2":
		return cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)

# convert hsv to rgb
def toRGB(hsv,form="sk"):
	if form == "sk":
		return color.hsv2rgb(hsv)
	elif form == "cv2":
		return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

# convert rgb to lab
def toLab(rgb):
	return color.rgb2lab(rgb)

# convert rgb to yiq
def toYIQ(rgb):
	return color.rgb2ycbcr(rgb)

# convert rgb to ycbcr
def toYCbCr(rgb):
	return color.rgb2yiq(rgb)

# convert rgb to grayscale
def toGray(rgb,form=int):
	if form == int:
		return skimage.util.img_as_ubyte(color.rgb2gray(rgb))
	elif form == float:
		return color.rgb2gray(rgb)

# convert rgb to any of above color spaces
def convert(rgb,space):
	if space == "hsv":
		return toHSV(rgb)
	elif space == "lab":
		return toLab(rgb)
	elif space == "gray":
		return toGray(rgb)
	elif space == "yiq":
		return toYIQ(rgb)
	elif space == "ycbcr":
		return toYCbCr(rgb)
	else:
		return rgb

# converts float image to byte image
# [0,1] -> [0,255]
def toByte(img):
	if skimage.util.dtype_limits(img) == (0,255):
		return img
	return skimage.util.img_as_ubyte(img)

# convert byte image to float image
# [0,255] -> [0,1]
def toFloat(img):
	return skimage.util.img_as_float(img)

# converts a (n_row,n_col,3) rgb mask to a (n_row,n_col) gray mask
def grayMask(mask):
	if mask.ndim == 2:
		return mask
	return mask[:,:,0]

# converts graymask to color mask
def colorMask(mask):
	if mask.ndim == 3:
		return mask
	return np.stack([mask,mask,mask],axis=-1).astype(bool)

# returns a mask of all true, masking none of an image
def emptyMask(img):
	return np.ones(img.shape,dtype=bool)

# converts a list of floats to a string
def list2str(lst):
	if lst == None or len(lst) == 0:
		return ""
	return '  '.join(str(np.around(f,3)) for f in lst)

# splits an image into its components
def splitComp(img):
	return img[:,:,0],img[:,:,1],img[:,:,2]

# combines 3 components to form color image
def combComp(a,b,c):
	return np.stack([a,b,c])

# get grayscale and each component of image in given space
def get4Comps(img,space):
	comps = []    
	comps.append(toGray(img))
	[a,b,c] = splitComp(convert(img,space))
	
	for comp in splitComp(convert(img,space)):
		comps.append(comp)
	
	return comps

########################################################
################### PLOTTING/DISPLAY ###################
########################################################

# gets shape of bar graph for a histogram, given the bins
def getShape(bins):
	center = (bins[:-1] + bins[1:]) / 2
	width = 1.0*(bins[1] - bins[0])
	return center,width

# gets mesh for 3d plot of grayscale image
def getMesh(img):
	ny,nx = img.shape
	X = np.linspace(0,1,nx)
	Y = np.linspace(0,1,ny)
	return np.meshgrid(X,Y)

# display 2 images side by side
def display(img1,img2,titles=[],colorbar=False):
	fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
	im1 = ax[0].imshow(img1)
	im2 = ax[1].imshow(img2)
	ax[0].set_xticks([]); ax[0].set_yticks([]); ax[1].set_xticks([]); ax[1].set_yticks([]);
	
	if len(titles) != 0:
		ax[0].set_title(titles[0])
		ax[1].set_title(titles[1])
	if colorbar:
		fig.colorbar(im1,ax=ax[0],fraction=0.046,pad=0.04)
		fig.colorbar(im2,ax=ax[1],fraction=0.046,pad=0.04)

# display a list of images
def displayAll(imgs,titles=[]):
	while len(titles) < len(imgs):
		titles.append("")

	for i in range(0,len(imgs),2):
		if i+1 < len(imgs):
			display(imgs[i],imgs[i+1],titles=titles[i:i+2])
		else:
			display(imgs[i],np.ones(imgs[i].shape),titles=[titles[i],""])    
	return

# display 4 image side by side
def displayFour(imgs,titles=["","","",""],colorbar=True):
	fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(10,4))
	
	for i in range(4):
		im = ax[i].imshow(imgs[i])
		ax[i].set_xticks([]); ax[i].set_yticks([])
		ax[i].set_title(titles[i])
		
		if colorbar: fig.colorbar(im,ax=ax[i],fraction=0.046,pad=0.04)
	return

# display grayscale and each component of given space
def displayComp(img,space):
	comps = get4Comps(img,space)

	fig,axes = plt.subplots(nrows=1,ncols=len(comps),figsize=(10,4))
	for ax,comp in zip(axes,comps):
		ax.imshow(toByte(comp),plt.cm.gray)
		ax.set_xticks([]); ax.set_yticks([])

# stacks 4 images into single image
# method = square -> 2x2 stack
# method = long   -> 1x4 stack
def stack4(imgs,method="square"):
    stack1 = np.concatenate((imgs[0],imgs[1]),axis=1)
    stack2 = np.concatenate((imgs[2],imgs[3]),axis=1)
    if method == "square":
    	return np.concatenate((stack1,stack2),axis=0)
    elif method == "long":
    	return np.concatenate((stack1,stack2),axis=1)


# Visualization of confusion matrix
# notmatthancock.github.io/2015/10/28/confusion-matrix.html
def showConfMatrix(C,class_labels=['Non-corroded','Corroded'],title=""):
	tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1]

	NP = fn+tp; NN = tn+fp; N = NP+NN

	fig = plt.figure(figsize=(4,4))
	ax  = fig.add_subplot(111)
	ax.imshow(C,interpolation='nearest',cmap=plt.cm.gray,aspect=0.4)
	plt.figtext(0.95,0.5,title)

	ax.set_xlim(-0.5,2.5); ax.set_ylim(2.5,-0.5)
	ax.plot([-0.5,2.5],[0.5,0.5],'-k',lw=2)
	ax.plot([-0.5,2.5],[1.5,1.5],'-k',lw=2)
	ax.plot([0.5,0.5],[-0.5,2.5],'-k',lw=2)
	ax.plot([1.5,1.5],[-0.5,2.5],'-k',lw=2)

	#ax.set_xlabel('Predicted',fontsize=12)
	ax.set_xticks([0,1,2]); ax.set_xticklabels(class_labels+[''],fontsize=8)
	ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
	# These coordinate might require some tinkering. Ditto for y, below.
	ax.xaxis.set_label_coords(0.34,1.06)

	# Set ylabels
	#ax.set_ylabel('Actual',fontsize=12,rotation=90)
	ax.set_yticklabels(class_labels+[''],rotation=90,fontsize=6); ax.set_yticks([0,1,2])
	ax.yaxis.set_label_coords(-0.09,0.65)

	bb = dict(fc='w',boxstyle='round,pad=0.1')

	# Fill in initial metrics: tp, tn, etc...
	ax.text(0,0,'TN: %d\n(#Neg: %d)'%(tn,NN),va='center',ha='center',bbox=bb)
	ax.text(0,1,'FN: %d'%fn,va='center',ha='center',bbox=bb)
	ax.text(1,0,'FP: %d'%fp,va='center',ha='center',bbox=bb)
	ax.text(1,1,'TP: %d\n(#Pos: %d)'%(tp,NP),va='center',ha='center',bbox=bb)

	# Fill in secondary metrics: accuracy, true pos rate, etc...
	ax.text(2,0,'FP Rate: %.2f'%(fp / (fp+tn+0.)),va='center',ha='center',bbox=bb)
	ax.text(2,1,'TP Rate: %.2f'%(tp / (tp+fn+0.)),va='center',ha='center',bbox=bb)
	ax.text(2,2,'Total: %.2f'%((tp+tn+0.)/N),va='center',ha='center',bbox=bb)
	ax.text(0,2,'NPV: %.2f'%(1-fn/(fn+tn+0.)),va='center',ha='center',bbox=bb)
	ax.text(1,2,'PPV: %.2f'%(tp/(tp+fp+0.)),va='center',ha='center',bbox=bb)


##########################################
################ BLOCKING ################
##########################################

# returns masked imaged from img and boolean mask
def getMasked(img,mask):    
	masked = np.copy(img)
	
	if img.ndim == 3 and mask.ndim == 2:
		mask = colorMask(mask)
	elif img.ndim == 2 and mask.ndim == 3:
		mask = grayMask(mask)
	masked[np.logical_not(mask)] = 0
	return masked

# returns block of img of size d at given ind
def findBlock(img,ind,d):
	i,j = ind
	if len(img.shape) == 2:
		return img[i:i+d,j:j+d]
	else:
		return img[i:i+d,j:j+d,:]

# find all blocks of img of size d at given inds
def findBlocks(img,inds,d):
	blocks = []
	for ind in inds:
		blocks.append(findBlock(img,ind,d))

	return blocks

# adds blocks of size d at given inds to existing mask
def maskBlocks(mask,inds,d):
	newmask = np.copy(mask)

	for ind in inds:
		i,j = ind
		if newmask.ndim == 2:
			newmask[i:i+d,j:j+d] = False
		else:
			newmask[i:i+d,j:j+d,:] = False
	return newmask

# gets all blocks
def getBlocks(img,d):
	blocks = []; ind = []
	for i in range(0,img.shape[0],d):
		for j in range(0,img.shape[1],d):
			block = findBlock(img,(i,j),d)
			blocks.append(block)
			ind.append((i,j))
	return blocks,ind

# gets blocks of size d at ind from a mask
def getBlockMask(mask,ind,d):
	i,j = ind
	return mask[i:i+d,j:j+d,:]

# get all block masks
def getBlockMasks(mask,inds,d):
	bmasks = []
	for ind in inds:
		bmasks.append(getBlockMask(mask,ind,d))
	return bmasks

# labels segmented regions of image
def label(mask):
	return skimage.measure.label(mask.astype(int),connectivity=1,return_num=True)

# draws a block at given (i,j) index
def drawBlock(img,ind,d):
	col = np.array([100,255,0])
	drawn = np.copy(img)
	x,y = ind
	rr,cc = skimage.draw.polygon(np.array([x,x,x+d,x+d]),np.array([y,y+d,y+d,y]))
	
	skimage.draw.set_color(drawn,(rr,cc),col,alpha=0.6)
	
	return drawn

# draws blocks at all given indices
def drawBlocks(img,inds,d):
	drawn = np.copy(img)
	for ind in inds:
		drawn = drawBlock(drawn,ind,d)
	
	return drawn

# draws block with relative shading corresponding to given value
def drawBlockColorCoded(img,ind,d,val,col):
	x,y = ind
	rr,cc = skimage.draw.polygon(np.array([x,x,x+d,x+d]),np.array([y,y+d,y+d,y]))
	skimage.draw.set_color(img,(rr,cc),col,alpha=val)
	return img

# draw blocks with color opacity corresponding to given value
def drawBlocksColorCoded(img,inds,d,vals):
	drawn = np.copy(img)
	col = np.array([128,255,0])
	for i in range(len(inds)):
		drawn = drawBlockColorCoded(drawn,inds[i],d,vals[i],col)
	
	return drawn

# get the neighbors of a pixel
def getNeighbors(img,i,j,size):
	return img[i-size:i+size+1,j-size:j+size+1]


############################################################
######################## STATISTICS ########################
############################################################

# calculates histogram for 1D data
# also returns data for plotting
def hist(data,nbins=256,range=None):
	his,bins = np.histogram(data,bins=nbins,density=True,range=range)
	center,width = getShape(bins)
	return his,center,width

# mean,var,skew,kurt,min,max
def getStats(data):
	if data.size == 0:
		return [np.nan for i in range(4)]
	stat = sp.stats.describe(data)
	return [stat[2],stat[3],stat[4],stat[5]]

# calculates 2d histogram of given components of image
def hist2d(img,mask,nbins=256,comps=(0,1)):
	comp1 = img[:,:,comps[0]]
	comp2 = img[:,:,comps[1]]
	x = comp1[grayMask(mask)]
	y = comp2[grayMask(mask)]
	
	return np.histogram2d(x,y,bins=nbins,range=[[0,1],[0,1]],normed=True)

# plots a given histogram
# center is array of bin centers
# width is bin width
def histPlot(hist,center,width,title=""):
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,2))
	ax.bar(center,hist,width=width)
	ax.set_title(title)
	ax.set_xlim([center[0]-0.5*width,center[-1]+0.5*width])
	#ax.set_ylim([0,0.1])

# calculates and plots histogram for given data
def fullHist(data,title="",nbins=256,range=None,plot=True):
	h,cen,wid = hist(data,nbins=nbins,range=range)
	if plot:
		histPlot(h,cen,wid,title=title)
	return h

# plots an image and histogram
def plotImgHist(img,hist,center,width,text=" "):
	fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
	ax[0].imshow(img); ax[0].set_xticks([]); ax[0].set_yticks([]);
	ax[1].bar(center,hist,width=width)
	ax[1].set_xlim([center[0]-0.5*width,center[-1]+0.5*width])
	#ax[1].set_ylim([0,0.1])
	ax[1].set_title(text)


# histogram intersection metric [Swain et al.]
def intersect(h1,h2):
	minima = np.minimum(h1,h2)
	return np.true_divide(np.sum(minima),np.sum(h2))

# histogram correlation (Pearson correlation)
# covariance(h1,h2) / (std(h1)*std(h2))
def correlate(h1,h2):
	num = np.sum(np.multiply(h1-np.mean(h1),h2-np.mean(h2)))
	
	a = np.sum(np.square(h1-np.mean(h1)))
	b = np.sum(np.square(h2-np.mean(h2)))
	denom = np.sqrt(np.multiply(a,b))

	return np.true_divide(num,denom)

# bhattacharyya distance between 2 histograms
def bhattacharyya(h1,h2):
	tot = np.sum(np.sqrt(np.multiply(h1,h2)))
	norm = 1 / np.sqrt(np.mean(h1)*np.mean(h2) * h1.size**2)

	return 1-np.sqrt(1-norm*tot)

# compares 2 histograms with all 3 methods
def compareHist(h1,h2):
	return [intersect(h1,h2),correlate(h1,h2),bhattacharyya(h1,h2)]

# returns 1 (corroded) or 0 (non-corroded) depending on histogram comparison
# methods = list of comparison methods to use (0,1,and/or 2)
# if 1 method is given, comparison must be greater than thresh
# if 2 methods are given
#	if condition = any, just 1 must be greater than thresh
# 	if condition = all, both must be greater than thresh
# if 3 methods are given, 2/3 must be greater than thresh
def compareThr(comp,thresh,methods,cond="any"):
	if len(methods) == 1: # must be > thresh
		return int(np.all(comp[methods] > thresh))
	
	elif len(methods) == 2: # at least 1 must be > thresh
		if cond == "any":
			return int(np.any(comp[methods] > thresh))
		elif cond == "all":
			return int(np.all(comp[methods] > thresh))
	
	elif len(methods) == 3: # at least 2 must be > thresh
		posvotes = np.sum(comp[methods] > thresh)
		negvotes = np.size(comp[methods]) - posvotes
		return int(posvotes >= negvotes)
	return 
			
# compares all hists to template h0
def compareAll(h0,hists):
	sims = []
	for h in hists:
		sims.append(compareHist(h0,h))
	return sims

# normalizes feature vector, s.t. all components are in range [0,1]
def normalize(feats):
	for i in range(feats.shape[1]):
		maximum = np.amax(np.absolute(feats[:,i]))
		if maximum > 1:
			feats[:,i] = feats[:,i] / maximum
			
	return feats


