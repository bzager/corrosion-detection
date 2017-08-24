# old.py
# Ben Zager
# Old functions from imageTools.py, no longer being used


##################################################
############## Plotting/Histograms ###############
##################################################

# computes histogram for an image
# param = 0 for hue, param=1 for saturation
def histo(hsv,mask,param=0,nbins=256):
	par = hsv[:,:,param]
	newmask = mask[:,:,0]
	count = par[newmask]
	return np.histogram(count,bins=nbins,density=True)

# computes histograms for list of hsv images
# param = 0 for hue, param=1 for saturation
def hsvHistAll(hsv,masks,param=0,nbins=128):
	hists = []
	for mask in masks:
		hists.append(histo(hsv,mask,param,nbins))

	return hists

# 
def histPlotAll(img,masks):
    for mask in masks:
        histPlotOne(img,mask)
    
    return

# Plots HS histogram in 3D
def hsHistPlot3D(hsv,mask,nbins=256):
    hist,xedges,yedges = hist2d(hsv,mask,nbins=32)

    xcen,xwid = getShape(xedges)
    ycen,ywid = getShape(yedges)
    X,Y = np.meshgrid(xcen,ycen)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    ax.plot_surface(X,Y,hist,cmap=plt.cm.magma)
    ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,np.amax(hist)])

    return

##################################################
############## K-means segmentation ##############
##################################################

# segments img into nseg segments using k-means clustering
def segment(img,nseg,comp,sig):
	return skimage.segmentation.slic(img,n_segments=nseg,compactness=comp,sigma=sig,convert2lab=True,enforce_connectivity=False) 

# marks the boundaries of a segmented image
def mark(img,seg):
	return skimage.segmentation.mark_boundaries(img,seg)

# creates boolean masks for each segment
def getMasks(img,seg):
	masks = []
	for (i,segVal) in enumerate(np.unique(seg)):
		mask = np.zeros(img.shape,dtype=bool)
		mask[seg == segVal] = True
		masks.append(mask)
	return masks

# creates images for each segment
def maskImgs(img,masks):
	masked = []

	for mask in masks:
		masked.append(getMasked(img,mask))
	return masked

#
def plotMaxima(img):
	lab = toLab(img)
	hist,bins = histo(lab,emptyMask(img),param=0,nbins=128)
	maxima = sp.signal.argrelmax(hist,order=5,mode="wrap")[0]
	
	fig,ax = plt.subplots(1,figsize=(4,2))
	ax.plot(bins[0:-1],hist)
	ax.plot(bins[maxima],hist[maxima],"o")
	
	return

#
def selectK(img):
	lab = toLab(img)
	hist,bins = histo(lab,emptyMask(img),param=0,nbins=128)
	maxima = sp.signal.argrelmax(hist,order=5,mode="wrap")[0]
	
	return np.amin([4,len(maxima)])

# run color analysis of image
def kcolorAnalysis(img,hsv,comp=0.1,sig=1,plot=True):
	k = selectK(img)
	
	seg = segment(img,k,comp,sig) # segment image
	marked = mark(img,seg) # mark boundaries
	masks = getMasks(img,seg) # get bool masks for each segment
	masked = maskImgs(img,masks) # get masked images
	
	if plot:
		display(marked,seg) # display original image with segment boundaries marked
		histPlotAll(img,hsv,masks)
		
	return masks


##################################################
################ Masking/Blocking ################
##################################################

#
def cleanMask(img,mask,minPix):
	labels,num = label(mask)
	keep = []
	
	for i in range(num):
		newmask = labels == i
		count = np.count_nonzero(newmask)
		if count > minPix:
			keep.append(i)
		else:
			labels[newmask] = 0
			mask[newmask] = False

	return keep,labels,mask

def fullClean(img,mask,minPix):
	mask = cleanMask(img,mask,minPix)[2]
	mask = np.invert(cleanMask(img,np.invert(mask),minPix)[2])
	return mask
