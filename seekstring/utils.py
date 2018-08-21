import os
import cv2
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
