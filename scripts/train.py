import os
import argparse
import numpy as np
import healpy as hp
import pylab as plt

import seekstring as ss

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-r', action="store_true", default=False)
parser.add_argument('--nside', action="store", type=int, default=2048)
parser.add_argument('--lmax', action="store", type=int, default=3500)
parser.add_argument('--fwhm', action="store", type=float, default=1.0)
parser.add_argument('--nsim', action="store", type=int, default=10)
parser.add_argument('--gmu', action="store", type=float, default=0.5)
parser.add_argument('--noise', action="store", type=float, default=0.0)
parser.add_argument('--time_limit', action="store", type=int, default=60)

args = parser.parse_args()
replace = args.r
nside = args.nside
lmax = args.lmax
fwhm = args.fwhm
n_gaussian = args.nsim
gmu = args.gmu
noise = args.noise
time_limit = args.time_limit

if nside==2048:
	n_string=3
else:
	n_string=1

gaussians = ['../data/maps/gaussian/'+'map_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.fits' for i in range(n_gaussian)]

strings = ['../data/maps/string/map1n_allz_rtaapixlw_2048_'+str(i+1)+'.fits' for i in range(n_string)]

model_add = './models/model_'+str(nside)+'_'+str(fwhm)+'_'+str(gmu)

#nside = 2048
#lmax = 2500
#fwhm = 0.0
#fwhm = fwhm*np.pi/(180*60)
#gmu = 0.5

def func(dt):
    return ss.canny(dt,0,'none','sch')

def filt_all(maps,func):
    out1 = []
    for m in maps:
        out1.append(func(m))
#     return np.stack([maps,np.array(out1)],axis=3)
    return np.array(out1)

def dp_total(n):
    l = 200
    string = dp_string(n,l)
    gaussian = dp_gaussian(n,l)
    y = filt_all(string,func).reshape(n,l,l,1)
    x = (gaussian+string).reshape(n,l,l,1) 
    return x,y
    

dp_gaussian = ss.Data_Provider(gaussians,dtype = np.float32)
dp_string = ss.Data_Provider(strings,dtype = np.float32,coef=gmu)

conv = ss.ConvolutionalLayers(nx=200,ny=200,learning_rate = 0.001,n_channel=1,restore=os.path.exists(model_add),
                        model_add=model_add,arch_file_name='arch')

conv.train(data_provider=dp_total,training_epochs = 10000000,n_s = 100, dropout=0.7, time_limit=time_limit, verbose=1)

