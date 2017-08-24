Image classifier for corrosion detection, written in Python

Installation:

Requires Python 2 or 3 with the following packages:
numpy, scipy, matplotlib, scikit-image, opencv, pywavelets, jupyter

If Python is not already installed, it can be installed from the Python website. On Mac OSX, I recommend installing with Homebrew.  The above packages can be installed with pip.

Running:

All code is in the src directory. Images are in the data directory, with training images and test images placed in separate subdirectories.  These are called train and test by default.

To run, change to the src directory and enter:

python main.py (test image directory name) (hue-saturation threshold) (edge threshold)

where (test image directory name) is the name of the directory with the images to be classified.  The name should be test, unless other directories are added. If the thresholds are not given, the defaults are HS threshold = 0.18 and edge threshold = 0.2. 

The classification takes around 2-3 seconds per image for 1,000,000 pixel images.

The results are saved as a text file in the results directory.  
The file lists the names of all images classified as corroded.
