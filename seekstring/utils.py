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

def canny(d_in,R,meth,edd):
    d = d_in
    if (R!=0):
        dt = np.fft.fft2(d)
        if meth=='g':
            for i in range(sz):
                for j in range(sz):
                    k2 = 1.*(i*i+j*j)/d.shape[0]
                    dt[i,j]=dt[i,j]*np.exp(-k2*R*R/2)

        if meth=='tp':
            for i in range(sz):
                for j in range(sz):
                    k = np.sqrt(0.001+i*i+j*j)/sz
                    dt[i,j]=dt[i,j]* 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3

        d = np.fft.ifft2(dt)
        d = abs(d)

    if edd=='lap':
        d = cv2.Laplacian(d,cv2.CV_64F)

    if edd=='sob':
        sobelx = cv2.Sobel(d,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(d,cv2.CV_64F,0,1,ksize=3)
        d =np.sqrt(sobelx**2+sobely**2)

    if edd=='sch':
        scharrx = cv2.Scharr(d,cv2.CV_64F,1,0)
        scharry = cv2.Scharr(d,cv2.CV_64F,0,1)
        d =np.sqrt(scharrx**2+scharry**2)
        
    return d
    
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
