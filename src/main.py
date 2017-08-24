# main.py
# Ben Zager
# runs corrosion detection classifier

import sys
import warnings

from imageTools import *
from image import Image
from classifier import Classifier

warnings.simplefilter('ignore')


def getInput():
	if len(sys.argv) >= 2:	
		testDir = sys.argv[1]
	else:
		testDir = "test"

	if len(sys.argv) >= 3:
		hsThr = sys.argv[2]
		edgeThr = sys.arg[3]
	else:
		hsThr = 0.18 
		edgeThr = 0.2 

	return testDir,hsThr,edgeThr

# writes results to text file
# lists file names of images classified as corroded
def writeResults(clf):
	res = open("../results/results_"+testDir+".txt","w+")
	for im in clf.test:
		if im.label == 1:
			res.write(im.name+"\n")
	res.close()


if __name__=="__main__":
	trainDir = "train"
	testDir,hsThr,edgeThr = getInput()
	
	blockThr = 0.9
	sig = 0.1
	HSBins = (180,256)
	directories = {"pos":trainDir,"test":testDir}

	print("Initializing classifier...")
	clf = Classifier(directories)
	print("Running classifier...")
	clf.run(blockThr=blockThr,hsThr=hsThr,edgeThr=edgeThr,sig=sig,HSBins=HSBins)
	print("Done!")

	writeResults(clf)

	#showConfMatrix(clf.confusion) # shows confusion matrix for test runs
	#plt.show() 
