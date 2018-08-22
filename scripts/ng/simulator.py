import matplotlib as mpl
mpl.use('agg')

import os
import sys
import argparse
import numpy as np
import pylab as plt
import gzip
import shutil
import healpy as hp
from healpy import cartview
from seekstring.utils import download,ch_mkdir
from ccgpack import sky2face

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-r', action="store_true", default=False)
parser.add_argument('--nside', action="store", type=int, default=2048)
parser.add_argument('--lmax', action="store", type=int, default=3500)
parser.add_argument('--fwhm', action="store", type=float, default=1.0)
args = parser.parse_args()
replace = args.r
lmax = args.lmax
fwhm = args.fwhm
fwhm_arcmin = args.fwhm
fwhm = fwhm*np.pi/(180*60)

cl = np.loadtxt('../../data/cl_planck_lensed')
ll = cl[:lmax,0]
cl = cl[:lmax,1]

ch_mkdir('./'+wset+'_set/gaussian/') 

nside = 2048
n_gaussian = 11
n_string=3

wsets = ['training']*(n_gaussian-1)+['test']

for i in range(n_gaussian):
    wset = wsets[i]

    if not os.path.exists('./'+wset+'_set/gaussian/'+'map_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.fits') or replace:
        print('Simulation gaussian map: '+str(i))

        m,alms = hp.sphtfunc.synfast(cl, nside=nside, lmax=lmax, mmax=None, alm=True, pol=False, pixwin=False, fwhm=fwhm, sigma=None, new=1, verbose=0)
        cl_map = hp.sphtfunc.alm2cl(alms)
        hp.write_map('./'+wset+'_set/gaussian/'+'map_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.fits', m, overwrite=1)
        m = hp.read_map('./'+wset+'_set/gaussian/'+'map_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.fits',verbose=0,nest=1)	
        hp.mollview(m, nest=1, cmap=cmap)	    
        plt.savefig('./'+wset+'_set/gaussian/'+'map_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.jpg')
        plt.close()

        patches = sky2face(m)
        for j in range(12):
            np.save('./'+wset+'_set/gaussian/map_p'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(12*i+j),patches[j])
            plt.imshow(patches[j], cmap=cmap)
            plt.savefig('./'+wset+'_set/gaussian/map_'+str(nside)+'_'+str(fwhm_arcmin)+'_p'+str(12*i+j)+'.jpg',bbox_inches='tight')
            plt.close()
        
        plt.figure(figsize=(10,6))

        dl1 = []
        dl2 = []
        for j in range(ll.shape[0]):
            dl1.append(ll[j]*(ll[j]+1)*cl[j]/(2*np.pi))
            dl2.append(ll[j]*(ll[j]+1)*cl_map[j]/(2*np.pi))

        plt.plot(ll,dl2,'r--',label='Simulation')
        plt.plot(ll,dl1,'b--',lw=2,label='Orginal')
        plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(labelsize=15)
        plt.xlabel(r'$\ell$',fontsize=25)
        plt.ylabel(r'$D_{\ell}$',fontsize=25)

        plt.legend(loc='best',fontsize=20)
        plt.savefig('./'+wset+'_set/gaussian/power_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.jpg')
        plt.close()

ch_mkdir('./'+wset+'_set/string/') 
	
wsets = ['training']*(n_string-1)+['test']

for i in range(n_string): 
    wset = wsets[i]

    if not os.path.exists('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz'):
        print('Downloading string: '+str(i))
        download('http://cp3.irmp.ucl.ac.be/~ringeval/upload/data/'+str(nside)+'/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz',
          './'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz')

    load_status = 0
    if not os.path.exists('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits'):
        print('Extracting string: '+str(i))
        with gzip.open('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz', 'rb') as f_in:
            with open('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        ss = hp.read_map('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits',verbose=0,nest=1)	
        load_status = 1	    
        hp.mollview(ss, nest=1, cmap=cmap)	
        plt.savefig('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.jpg')
        plt.close()
        patches = sky2face(ss)
        for j in range(12):
            np.save('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j),patches[j])
            plt.imshow(patches[j], cmap=cmap)
            plt.savefig('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j)+'_'+str(fwhm_arcmin)+'.jpg',bbox_inches='tight')
            plt.close()
		        
    if fwhm!=0.0:
        if load_status==0:
            ss = hp.read_map('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits',verbose=0,nest=1)
        print('Beaming string: '+str(i))
        ss = hp.sphtfunc.smoothing(ss,fwhm=fwhm)
        hp.write_map('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'_'+str(fwhm_arcmin)+'.fits', ss, overwrite=1)
        patches = sky2face(ss)
        for j in range(12):
            np.save('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j)+'_'+str(fwhm_arcmin),patches[j])
            plt.imshow(patches[j], cmap=cmap)
            plt.savefig('./'+wset+'_set/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j)+'_'+str(fwhm_arcmin)+'.jpg',bbox_inches='tight')
            plt.close()
	    
		    
		    

