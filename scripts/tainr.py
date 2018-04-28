import os
import numpy as np
import healpy as hp
import pylab as plt

import seekstring as ss

n_gaussian = 10

nside = 2048
lmax = 2500
fwhm = 0.0
fwhm = fwhm*np.pi/(180*60)
gmu = 0.5

def func(dt):
    return canny(dt,0,'none','sch')

def filt_all(maps,func):
    out1 = []
    for m in maps:
        out1.append(func(m))
        
#     return np.stack([maps,np.array(out1)],axis=3)
    return np.array(out1)

gaussians = ['../data/maps/gaussian/'+'map_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.fits' for i in range(n_gaussian)]
dp_gaussian = ss.Data_Provider(gaussians,dtype = np.float32)

strings = ['../data/maps/string/map1n_allz_rtaapixlw_2048_'+str(i+1)+'.fits' for i in range(3)]
dp_string = ss.Data_Provider(strings,dtype = np.float32,coef=0.5)

def dp_total(n):
    l = 200
    string = dp_string(n,l)
    gaussian = dp_gaussian(n,l)
    
    y = filt_all(string,func).reshape(n,l,l,1)
    x = (gaussian+string).reshape(n,l,l,1)
    
    return x,y

model_add = './models/model_'+str(nside)+'_'+str(fwhm)

conv = ss.ConvolutionalLayers(nx=200,ny=200,learning_rate = 0.001,n_channel=1,restore=os.path.exists(model_add),
                        model_add=model_add,arch_file_name='arch')

conv.train(data_provider=dp_total,training_epochs = 200,n_s = 10, dropout=0.7, time_limit=None, verbose=1)

