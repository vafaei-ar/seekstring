#import matplotlib as mpl
#mpl.use('agg')
#import pylab as plt
import os
import sys
import glob
import numpy as np
import ccgpack as ccg
from src.utils import detection_test

print('Tests are initiating ...')
ccg.ch_mkdir('./data/tests')

crv = 7
filt = 'sch'
gmus = [1e-6,2e-6,3e-6,4e-6]

g_files = []
s_files = []
for i in range(n_gaussian): 
    for j in range(12):
        g_files.append('./data/maps/gaussian/map_p'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(12*i+j))

for i in range(n_string): 
    for j in range(12):
        if fwhm_arcmin==0.0:
            s_files.append('./data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j))
        else:
            s_files.append('./data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j)+'_'+str(fwhm_arcmin))

ng = len(g_files)
ns = len(s_files)

npatch = min(ng,ns)
nside = np.load(g_files[0]).shape[0]
print('Number of available patches (nside=512):',16*npatch)

nprt = npatch*16

for gmu in gmus:
    print('gmu: ',gmu)
    s2s = []
    prc = 0
    for ip in range(npatch):
        gaussian = np.load(g_files[ip])
        string = np.load(s_files[ip])        
        for i in range(4):
            for j in range(4):
                gg = 1e-6*gaussian[i*512:(i+1)*512,j*512:(j+1)*512]
                ss = string[i*512:(i+1)*512,j*512:(j+1)*512]
#                print gg.std(),gmu*ss.std()
                sample = gg+gmu*ss
                s2 = detection_test(sample,crv=crv,filt=filt)
                s2s.append(s2)   

                prc += 1
                sys.stdout.write("\r{:3.2f}%".format(100.*prc/nprt)+ ' complete')
                sys.stdout.flush()
    print('gmu: ',gmu,' is completed.')
    np.save('./data/tests/detection_'+str(gmu)+'_'+str(crv)+'_'+filt,np.array(s2s))
    
print('\nDetection analysis is completed.')
