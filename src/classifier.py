# classifier.py
# Ben Zager
# Classifier class for corrosion classifier
# 

import os

from imageTools import *
from color import *
from texture import *
from image import *

#from sklearn import svm,neighbors,decomposition,preprocessing

# load image from given directory
def loadImgs(directory,group,maxsize=1000000):
		files = []
		for fname in os.listdir("../data/"+directory):
			if fname==".DS_Store":
				continue 
			files.append(Image(fname,directory,group,maxsize=maxsize))
			print(fname)

		return files

# rescales all given images 
def rescaleAll(imgs,maxsize):
	results = []
	for im in imgs:
		if im.size > maxsize:
			scale = np.sqrt(maxsize/float(im.size))
			im.rgb = pad(rescale(im.rgb,scale))
			im.reInit()

	return imgs

# initializes images as Image objects
def initImgs(imgs,group,directory):
	new = []
	for im in imgs:
		new.append(Image(im.name,directory,group,img=im.rgb))
		
	return new

# initializes Image objects from list of raw images or load files
def getImgs(directory,group,maxsize,imgs):
	if imgs == None:
		return loadImgs(directory,group,maxsize=maxsize)
	else:
		return initImgs(imgs,group,directory)


class Classifier:

	def __init__(self,directories,maxsize=1000000,pos=None,test=None):

		self.directories = directories # {"pos":pos_dir,"test":test_dir}
		self.maxsize = maxsize
		self.pos = getImgs(directories["pos"],"pos",maxsize,pos)
		self.test = getImgs(directories["test"],"test",maxsize,test)

		self.posHSHist = None
		self.posEdgeHist = None

		self.confusion = None
		self.testLabels = np.ones(len(self.test))
		#self.trainLabels = np.concatenate((np.ones(len(self.pos)),np.zeros(len(self.neg))))
		
		#self.trainFeatures = None
		#self.testFeatures = None

	# 
	def resetLabels(self):
		self.testLabels = np.ones(len(self.test))
		for i in range(len(self.test)):
			self.test[i].label = 1

	# 
	def setLabel(self,i,label):
		self.testLabels[i] = label
		self.test[i].label = label

	# sets the blocks for each image using given size and blockThr
	def setBlocks(self,d,blockThr):
		for im in self.pos: im.setRedBlocks(d,blockThr)
		for im in self.test: im.setRedBlocks(d,blockThr)

	# initializes color features for all images
	def initColor(self,d=8,blockThr=0.9,minPix=64,nbins=(45,64)):
		for im in self.pos: im.initColor(d=d,blockThr=blockThr,minPix=minPix,nbins=nbins)
		for im in self.test: im.initColor(d=d,blockThr=blockThr,minPix=minPix,nbins=nbins)

	# initializes texture features for all images
	def initTexture(self,d=8,nbins=32,sig=0.1):
		for im in self.pos: im.initTexture(d=d,nbins=nbins,sig=sig)
		for im in self.test: im.initTexture(d=d,nbins=nbins,sig=sig)

	# sets HS histogram template from training images
	def setPosHSHist(self):
		hists = np.asarray([im.hsHist for im in self.pos])
		self.posHSHist = np.mean(hists,axis=0)

	# sets edge density template from training image
	def setPosEdgeHist(self):
		hists = np.asarray([im.edgeHist for im in self.pos])
		self.posEdgeHist = np.mean(hists,axis=0)

	# compares all test image HS histograms to template
	def compareHSHists(self):
		self.setPosHSHist()
		for im in self.pos: im.compareHSHist(self.posHSHist)
		for im in self.test: im.compareHSHist(self.posHSHist)

	# compares all test image edge density histograms to template
	def compareEdgeHists(self):
		self.setPosEdgeHist()
		for im in self.pos: im.compareEdgeHist(self.posEdgeHist)
		for im in self.test: im.compareEdgeHist(self.posEdgeHist)

	# classifies first round of images using color features
	# compares HS histogram comparison value to hsThr
	# (see compareThr in imageTools.py)
	def preclassify(self,hsThr=0.2,methods=[2],cond="any"):
		for i in range(len(self.test)):
			comp = np.asarray(self.test[i].hsComp)
			self.setLabel(i,compareThr(comp,hsThr,methods,cond=cond))
			
			if len(self.test[i].redBlocks) == 0:
				self.setLabel(i,0)
		self.setConfMatrix()

	# classifies second round of images using texture features
	def postclassify(self,edgeThr=0.2,methods=[2],cond="any"):
		for i in range(len(self.test)):
			if self.test[i].label == 1:
				comp = np.asarray(self.test[i].edgeComp)
				self.setLabel(i,compareThr(comp,edgeThr,methods,cond=cond))
		self.setConfMatrix()

	# sets HS histograms for all images, then sets templates
	def setHSHists(self,nbins=(45,64)):
		for im in self.pos: im.setHSHist(nbins=nbins)
		for im in self.test: im.setHSHist(nbins=nbins)
		self.setPosHSHist()
		self.setPosEdgeHist()

	# runs full classifier using given thresholds
	def run(self,d=8,blockThr=0.9,hsThr=0.2,edgeThr=0.2,minPix=64,sig=0.1,HSBins=(45,64),edgeBins=32,premethods=[2],postmethods=[2]):
		print("    Color...")
		self.initColor(d=d,blockThr=blockThr,minPix=minPix,nbins=HSBins)
		self.compareHSHists()
		self.preclassify(hsThr=hsThr,methods=premethods)
		print("    Texture...")
		self.initTexture(d=d,sig=sig,nbins=edgeBins)
		self.compareEdgeHists()
		self.postclassify(edgeThr=edgeThr,methods=postmethods)
		self.setTitles()


	# calculates confusion matrix 
	# t/f = true/false, p/n = positive/negative
	def setConfMatrix(self):
		tp = 0; tn = 0; fp = 0; fn = 0
		for im in self.test:
			if im.label == 1 and im.actual == 1:
				tp += 1
			elif im.label == 0 and im.actual == 0:
				tn += 1
			elif im.label == 1 and im.actual == 0:
				fp += 1
			elif im.label == 0 and im.actual == 1:
				fn += 1

		self.confusion = np.array([[tn,fp],[fn,tp]])

	# correct / total 
	def accuracy(self):
		correct = self.confusion[0,0] + self.confusion[1,1]
		return correct / np.sum(self.confusion)

	# TP / NP
	def sensitivity(self):
		TP = self.confusion[1,1]
		NP = TP + self.confusion[1,0]
		return TP / NP

	# FP / NN
	def FPR(self):
		FP = self.confusion[0,1]
		NN = FP + self.confusion[0,0]
		return FP / NN

	# (TNR) TN / NN
	def specificity(self):
		TN = self.confusion[0,0]
		NN = TN + self.confusion[0,1]
		return TN / NN

	"""
	def setFeatures(self):
		for i in range(len(self.pos)): self.pos[i].setFeatures()
		for i in range(len(self.test)): self.test[i].setFeatures()
	
	# 
	def prepFeatures(self):
		feats = []
		for i in range(len(self.pos)):
			if self.pos[i].label: feats.append(self.pos[i].features)
		for i in range(len(self.test)):
			if self.test[i].label: feats.append(self.test[i].features)
		
		features = normalize(np.asarray(feats))
		self.trainFeatures = features[:len(self.pos+self.neg),:]
		self.testFeatures = features[len(self.pos+self.neg):,:]

	#
	def train(self,C=1.0):
		self.svm.C = C
		self.svm.fit(self.trainFeatures,self.trainLabels)

	#
	def predict(self):
		for i in range(len(self.test)):
			if self.testLabels[i]:
				self.testLabels[i] = self.svm.predict(self.testFeatures[i])
	"""

######################################################
################## PLOTTING/DISPLAY ##################
######################################################
	
	# gets list of all blocks from every image in given group
	def getAllBlocks(self,group):
		blocks = []

		if group == "pos": imgs = self.pos
		else: imgs = self.test

		for im in imgs:
			blocks += im.redBlocks

		return blocks

	# gets list of falsely classified images
	def getFalse(self,group=""):
		false = []
		for im in self.test:
			if not im.isCorrect():
				if group == "pos" and im.actual == 0: # 
					continue
				elif group == "neg" and im.actual == 1:
					continue
				false.append(im)

		return false

	# gets positive test images 
	def getPosTest(self):
		pos = []
		for im in self.test:
			if im.actual == 1:
				pos.append(im)
		return pos

	# gets negative test images
	def getNegTest(self):
		neg = []
		for im in self.test:
			if im.actual == 0:
				neg.append(im)
		return neg


	#
	def setTitles(self):
		for i in range(len(self.pos)): self.pos[i].setTitle()
		for i in range(len(self.test)): self.test[i].setTitle()

	#
	def getGroup(self,group):
		if group == "test":
			return self.test
		elif group == "pos":
			return self.pos
		else:
			return self.test

	# display all images in given group
	def displayAll(self,group="test",count=10):
		imgroup = self.getGroup(group)
		imgs = [im.rgb for im in imgroup[:count]]
		titles = [im.title for im in imgroup[:count]]
		displayAll(imgs,titles=titles)

	# display all masked images in given group
	def displayMasked(self,group="test",count=10):
		imgroup = self.getGroup(group)
		masked = [im.masked for im in imgroup[:count]]
		titles = [im.title for im in imgroup[:count]]
		displayAll(masked,titles=titles)



