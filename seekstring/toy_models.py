from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from skimage.draw import polygon
from scipy.ndimage.interpolation import rotate

def normal(x):
    xmax = x.max()
    if xmax!=0:
        x = x/xmax
    return x

class Simple_String(object):

    def __init__(self,nx=1000,ny=1000,num=10,l_min=None,l_max=None):
                
        self.nx = nx
        self.ny = ny
        self.num = num
        if l_min is None:
            self.l_min = min(nx,ny)//10
        else:
            self.l_min = l_min
        if l_max is None:
            self.l_max = min(nx,ny)//3
        else:
            self.l_max = l_max
    
    def string(self,ns=100):
    
        margx = 2*(self.nx//5)
        margy = 2*(self.ny//5)

        imx = self.nx+margx
        imy = self.ny+margy
        
        rec = np.zeros((imx,imy))
        for i in range(ns):

            sys.stdout.write("\r{:3.1f}%".format(100.*(i+1)/ns)+ ' complete')
            sys.stdout.flush()

            width = 2*np.random.randint(self.l_min,self.l_max)
            height = width/2
            rand = np.random.randint((imx-width)*(imy-height))
            
            r0 = rand%(imx-width)
            c0 = rand//(imx-width)
            angle = np.random.uniform(0,180)
        
            rr, cc = [r0, r0 + width//2, r0 + width//2., r0], [c0, c0, c0 + height, c0 + height]
            rr, cc = polygon(rr, cc)    
            rec[rr, cc] += 1
            
            rr, cc = [r0 + width//2, r0 + width, r0 + width, r0 + width//2], [c0, c0, c0 + height, c0 + height]
            rr, cc = polygon(rr, cc)    
            rec[rr, cc] -= 1
        
            rec = rotate(rec, angle, axes=(1, 0), reshape=0) 
        print('')       
        return normal(rec[margx//2:-margx//2,margy//2:-margy//2])
            
    







