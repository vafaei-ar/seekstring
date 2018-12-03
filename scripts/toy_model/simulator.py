import matplotlib as mpl
mpl.use('agg')

import os
import sys
import argparse
import numpy as np
import pylab as plt
import urllib
import gzip
import shutil
import healpy as hp
from ccgpack import sky2patch
from healpy import cartview
from seekstring.toy_models import Simple_String

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-r', action="store_true", default=False)
parser.add_argument('--nside', action="store", type=int, default=2048)
parser.add_argument('--lmax', action="store", type=int, default=0)
parser.add_argument('--nsim', action="store", type=int, default=5)
parser.add_argument('--nstring', action="store", type=int, default=100)

args = parser.parse_args()
replace = args.r
nside = args.nside
if args.lmax==0:
    lmax = 2*nside
n_gaussian = args.nsim
nstring = args.nstring

#n_gaussian = 10
#nside = 2048
#lmax = 3500
#fwhm = float(sys.argv[1])

cl = np.load('../../data/cl_planck_lensed.npy')
ll = cl[:lmax,0]
cl = cl[:lmax,1]
ss = Simple_String(nx=nside,ny=nside)

if not os.path.exists('./data/gaussian/'):
    os.makedirs('./data/gaussian/') 
if not os.path.exists('./data/string/'):
    os.makedirs('./data/string/') 
    
for i in range(n_gaussian):

    if not os.path.exists('./data/gaussian/'+'map_'+str(nside)+'_'+str(i)+'.fits') or replace:

        m,alms = hp.sphtfunc.synfast(cl, nside=nside, lmax=lmax, mmax=None, alm=True, pol=False, pixwin=False, fwhm=0, sigma=None, new=1, verbose=0)
        cl_map = hp.sphtfunc.alm2cl(alms)

        hp.mollview(m, nest=0, cmap=cmap)
        hp.write_map('./data/gaussian/'+'map_'+str(nside)+'_'+str(i)+'.fits', m, overwrite=1)
        plt.savefig('./data/gaussian/'+'map_'+str(nside)+'_'+str(i)+'.jpg')
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
        plt.savefig('./data/gaussian/power_'+str(nside)+'_'+str(i)+'.jpg')
        plt.close()

        m = hp.reorder(m,r2n=1)
        patches = sky2patch(m, 1)
        for j in range(12):
            print('Simulation : '+str(12*i+j)+'/'+str(12*n_gaussian))
            np.save('./data/gaussian/g_'+str(nside)+'_'+str(12*i+j),patches[j])
            plt.imshow(patches[j], cmap=cmap)
            plt.savefig('./data/gaussian/g_'+str(nside)+'_p'+str(12*i+j)+'.jpg',bbox_inches='tight')
            plt.close()
            
            x = ss.string(ns=nstring)
            x -= x.min()
            x /= x.max()
            np.save('./data/string/s_'+str(nside)+'_'+str(12*i+j),x)
            plt.imshow(x,cmap=cmap)
            plt.savefig('./data/string/s_'+str(nside)+'_'+str(12*i+j)+'.jpg',bbox_inches='tight')
            plt.close()

