from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import urllib
import requests
import numpy as np
from astropy.io import fits

def standard(X):
	"""
	standard : This function makes data ragbe between 0 and 1.
	
	Arguments:
		X (numoy array) : input data.
	
	--------
	Returns:
		standard data.
	"""
	xmin = X.min()
	X = X-xmin
	xmax = X.max()
	X = X/xmax
	return X
    
def ch_mkdir(directory):
    """
    ch_mkdir : This function creates a directory if it does not exist.

    Arguments:
        directory (string): Path to the directory.

    --------
    Returns:
        null.		
    """
    if not os.path.exists(directory):
          os.makedirs(directory)    
          
def the_print(text,style='bold',tc='gray',bgc='red'):
    """
    prints table of formatted text format options
    """
    colors = ['black','red','green','yellow','blue','purple','skyblue','gray']
    if style == 'bold':
        style = 1
    elif style == 'underlined':
        style = 4
    else:
        style = 0
    fg = 30+colors.index(tc)
    bg = 40+colors.index(bgc)
    
    form = ';'.join([str(style), str(fg), str(bg)])
    print('\x1b[%sm %s \x1b[0m' % (form, text))

def report(count, blockSize, totalSize):
  	percent = int(count*blockSize*100/(totalSize))
  	sys.stdout.write("\r%d%%" % percent + ' complete')
  	sys.stdout.flush()

def connected(url='http://www.google.com/'):
    timeout=5
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("Internet connection problem.")
        return False

def download(getFile, saveFile=None):
    assert connected(),'Error! check your Internet connection.'
    if saveFile is None:
        saveFile = getFile.split('/')[-1]
    sys.stdout.write('\rFetching ' + saveFile + '...\n')
    try:
        urllib.urlretrieve(getFile, saveFile, reporthook=report)
    except:
        urllib.request.urlretrieve(getFile, saveFile, reporthook=report)
    sys.stdout.write("\rDownload complete, saved as %s" % (saveFile) + '\n\n')
    sys.stdout.flush()

def pop_percent(i,ntot):
    sys.stdout.write("\r{:3.1f}%".format(100.*(i+1)/ntot)+ ' complete')
    sys.stdout.flush()
