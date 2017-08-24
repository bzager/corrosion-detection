# texture.py
# Ben Zager
# Functions for texture feature extraction
# 


import numpy as np
import scipy as sp
import skimage
import pywt

from imageTools import *

from skimage import filters,feature


#############################################################
########################## TEXTURE ##########################
#############################################################

# calculates edge magnitude with sobel gradient operator
def edgeMag(img,mask=None):
	mag = filters.sobel(img,mask=mask)
	return mag

# calculates edge direction with sobel gradient operator
def edgeDir(img,mask=None):
	hori = filters.scharr_h(img,mask=mask)
	vert = filters.scharr_v(img,mask=mask)
	return np.arctan2(hori,vert)

# locates edge pixels with canny edge detection
def canny(img,mask=None,sigma=0.1,low=None,high=None):
	return feature.canny(img,sigma=sigma,mask=mask,low_threshold=low,high_threshold=high)

# locates edges with canny edge detector or thresholding edge magnitude
def edges(img,mask=None,sigma=0.1,thresh=0.1):
	#return edgeMag(img,mask=mask) > thresh
	return canny(img,mask=mask,sigma=sigma)

# calculates edge density of block from edge image
def blockEdgeness(bl):
	return (np.sum(bl))/bl.size

# calculates edge density of all given blocks
def edgenessTotal(edge,redInds,d):
	edgeness = np.zeros(len(redInds))
	if len(redInds) == 0:
		return edgeness
	edgeBlocks = findBlocks(edge,redInds,d)

	for i in range(len(redInds)):
		edgeness[i] = blockEdgeness(edgeBlocks[i])
	return edgeness

# normalizes fft
def fftNorm(fft):
	fft = np.absolute(fft)
	return fft/np.sqrt(np.sum(np.square(fft[1:,1:])))

# calculates and normalizes fft of image
def fourier(img):
	return fftNorm(sp.fftpack.fft2(img))

# calculates log of normalized fft
def logFourier(im):
	return np.log(fourier(im))

# calculates fft of all components of given color space
# Possible color spaces: HSV,Lab,YCbCr(CbCr),YIQ
def fourierAll(img,space,log=False):
	a,b,c = splitComp(convert(img,space))
	gray = toGray(img)
	
	if log: fft = logFourier
	else: fft = fourier
	
	return [fft(gray),fft(a),fft(b),fft(c)]

# inverse fft
def ifourier(fft):
	return sp.fftpack.ifft2(fft)

# calculates energy of array as sum of squares of each element
def energy(arr):
	return np.sum(np.square(arr))

# calculates energy of all arrays in a list
def allEnergy(data):
	energies = []
	for dat in data:
		energies.append(energy(dat))

	return np.asarray(energies)

# calculates proportion of energy in each array from list
# energy of array divided by total energy of all arrays
def normEnergy(energies):
	return np.true_divide(energies,np.sum(energies))

# rearranges wavelet coefficients into correct subbands
def rearrange(wav):
	re = []
	for w in wav:
		if type(w) == tuple:
			for u in w:
				re.append(u)
		else:
			re.append(np.asarray(w))
	return re

# 2d wavelet transform
# Order 4 Debauchies wavelet
def wavelet(img):
	return pywt.wavedec2(img,'db1',level=3)

# converts wavelet coefficients to single array
def wav2arr(wav):
	arr,slices = pywt.coeffs_to_array(wav,padding=0)
	return arr

# calculates normalized wavelet energy
def waveletEnergy(img):
	wav = rearrange(wavelet(img))
	energies = allEnergy(wav)
	return normEnergy(energies)

# calculates normalized wavelet energy for all blocks in image
def waveletEnergyAll(redBlocks,space="gray",comp=0):
	energies = []
	for bl in redBlocks:
		if space == "gray": 
			im = toGray(bl)
		else: 
			im = splitComp(convert(bl,space))[comp]
		energies.append(waveletEnergy(im))

	return np.asarray(energies)

# gets stats for wavelet energies
def waveletStats(energies):
	stats = []
	for i in range(energies.shape[1]):
		stats += getStats(energies[:,i])
	
	return stats

# gets average and standard deviation for wavelet energy components 
# over all blocks
def waveletAnalysis(blocks,space,comps):
	avgs = []; stds = []
	for c in comps:
		en = waveletEnergyAll(blocks,space=space,comp=c)
		avgs.append(np.mean(en,axis=0))
		stds.append(np.std(en,axis=0))
	return avgs,stds

# coefficient of variation of gray level intensities
# std / mean
def cov(gray,gmask):
	return np.std(gray[gmask]) / np.mean(gray[gmask])

# cov of all blocks 
def covAll(gray,mask,redBlocks,redInds):
	cv = np.zeros(len(redBlocks))
	d = redBlocks[0].shape[0]

	for i in range(len(redBlocks)):
		bmask = getBlockMask(mask,redInds[i],d)
		cv[i] = cov(toGray(redBlocks[i]),grayMask(bmask))
	return cv

# 
def covLocal(img,size):
	result = np.zeros(img.shape)
	d = 2*size+1
	wind = skimage.util.view_as_windows(img,(d,d),step=1)
	newshape = (wind.shape[0],wind.shape[1],d**2)
	wind = np.reshape(wind,newshape)

	return np.std(wind,axis=2) / np.mean(wind,axis=2)

# builds gray level co-occurrence matrix
def getGLCM(gray,dist=1,ang=0):
	return feature.greycomatrix(gray,[dist],[ang],symmetric=True,normed=True)

# calcualtes GLCM energy of grayscale image
def glcmEnergy(gray,dist=5,ang=0):
	glcm = getGLCM(gray,dist,ang)
	energy = feature.greycoprops(glcm,prop="energy")
	return energy[0][0]

# calcualtes GLCM energy of list of grayscale images
def glcmEnergyAll(redBlocks):
	energies = np.zeros(len(redBlocks))

	for i in range(len(redBlocks)):
		energies[i] = glcmEnergy(toGray(redBlocks[i]))
	return energies

# Laws' texture energy filters
def getLaws():
	L = np.array([1,4,6,4,1])
	E = np.array([-1,-2,0,2,1])
	S = np.array([-1,0,2,0,-1])
	R = np.array([1,-4,6,-4,1])
	W = np.array([-1,2,0,-2,1])

	lawVecs = {"L":L,"E":E,"S":S,"R":R,"W":W}
	lawFilts = {}
	for name0,vec0 in lawVecs.items():
		for name1,vec1 in lawVecs.items():
			lawFilts[name0+name1] = np.outer(vec0,vec1)

	return lawFilts

# swaps 1st 2 characters in a string
def swapChar(string):
	return string[1]+string[0]

# 2d convolution of image with kernel
def convolve(img,kernel):
	return sp.ndimage.convolve(img,kernel,mode="reflect")

# smooths image with averaging kernel of given size
def smooth(img,size):
	kernel = np.ones([size,size])/size**2
	return convolve(img,kernel)

# combines redundant filters from Laws' filter bank
def combineLaws(results):
	filts = []
	names = []

	for name,filt in results.items():
		if swapChar(name) not in names:
			names.append(name)
			filts.append((filt+results[swapChar(name)])/2)
			
	return filts,names

# runs Laws' texture energy filtering
def filterLaws(img,size=15):
	filts = getLaws()
	results = {}

	img = smooth(img,size)

	for name,filt in filts.items():
		filt = convolve(img,filt)
		result = energyMap(filt,size)
		results[name] = result
	
	return results

# local averaging of filtered image for Laws' texture energy
def energyMap(filt,size=15):
	return convolve(np.absolute(filt),np.ones([size,size]))

# places all texture features into list
def textureStats(img,gray,mask,gmask,redBlocks,redInds):
	stats = []
	#cv = covAll(gray,mask,redBlocks,redInds)
	#glcmEn = glcmEnergyAll(redBlocks)
	#avgs,stds = waveletAnalysis(redBlocks,space="ycbcr",comps=[2])

	#stats += getStats(cv)
	#stats += getStats(glcmEn)
	#stats += avgs[0].tolist()

	#stats += 

	return stats


