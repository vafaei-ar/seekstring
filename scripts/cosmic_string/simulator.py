import matplotlib as mpl
mpl.use('agg')

import os
import sys
import argparse
import numpy as np
import pylab as plt
import urllib
import shutil
import healpy as hp
from ccgpack import sky2patch
from healpy import cartview

import urllib
import requests
def connected(url='http://www.google.com/'):
    timeout=5
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("Internet connection problem.")
        return False

#def internet_check():
#    try:
#        urllib.urlopen('http://216.58.192.142')
#        return True
#    except urllib2.URLError as err: 
#        return False

def report(count, blockSize, totalSize):
  	percent = int(count*blockSize*100/(totalSize))
  	sys.stdout.write("\r%d%%" % percent + ' complete')
  	sys.stdout.flush()

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

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-r', action="store_true", default=False)
parser.add_argument('--nside', action="store", type=int, default=2048)
parser.add_argument('--lmax', action="store", type=int, default=0)
parser.add_argument('--fwhm', action="store", type=float, default=0.0)
parser.add_argument('--nsim', action="store", type=int, default=10)
args = parser.parse_args()
replace = args.r
nside = args.nside
if args.lmax==0:
    lmax = 2*nside
fwhm = args.fwhm
fwhm_arcmin = args.fwhm
fwhm = fwhm*np.pi/(180*60)
n_gaussian = args.nsim

#n_gaussian = 10
#nside = 2048
#lmax = 3500
#fwhm = float(sys.argv[1])

cl = np.load('../../data/cl_planck_lensed.npy')
ll = cl[:lmax,0]
cl = cl[:lmax,1]

if not os.path.exists('./data/gaussian/'):
    os.makedirs('./data/gaussian/') 

for i in range(n_gaussian):

    if not os.path.exists('./data/gaussian/'+'map_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.fits') or replace:
        print('Simulation gaussian map: '+str(i))

        m,alms = hp.sphtfunc.synfast(cl, nside=nside, lmax=lmax, mmax=None, alm=True, pol=False, pixwin=False, fwhm=fwhm, sigma=None, new=1, verbose=0)
        cl_map = hp.sphtfunc.alm2cl(alms)

        hp.mollview(m, nest=0, cmap=cmap)
        hp.write_map('./data/gaussian/'+'map_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.fits', m, overwrite=1)
        plt.savefig('./data/gaussian/'+'map_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.jpg')
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
        plt.savefig('./data/gaussian/power_'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(i)+'.jpg')
        plt.close()

        m = hp.reorder(m,r2n=1)
        patches = sky2patch(m, 1)
        for j in range(12):
            np.save('./data/gaussian/map_p'+str(nside)+'_'+str(fwhm_arcmin)+'_'+str(12*i+j),patches[j])
            plt.imshow(patches[j], cmap=cmap)
            plt.savefig('./data/gaussian/map_'+str(nside)+'_'+str(fwhm_arcmin)+'_p'+str(12*i+j)+'.jpg',bbox_inches='tight')
            plt.close()
                
if not os.path.exists('./data/string/'):
    os.makedirs('./data/string/') 
    
if nside==2048:
    n_string=3
    ex = 'gz'
    import gzip
    
    def extract(in_file,out_file):
        with gzip.open(in_file, 'rb') as f_in:
            with open(out_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
elif nside==4096:
    n_string=1
    ex = 'xz'
        
    def extract(in_file,out_file):
        os.system('unxz '+in_file)   
#    unxz file.xz
else:
    assert 0,'Nside have to be either 2048 or 4096!'

for i in range(n_string): 
    strnum = str(i+1)
    if nside==4096:
        strnum = strnum+'b'
    if not os.path.exists('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex):
        print('Downloading string: '+str(i))
        download('http://cp3.irmp.ucl.ac.be/~ringeval/upload/data/'+str(nside)+'/map1n_allz_rtaapixlw_'+str(nside)+'_'+strnum+'.fits.'+ex,
          './data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex)

    load_status = 0
    if not os.path.exists('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits'):
        print('Extracting string: '+str(i))
#        with gzip.open('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex, 'rb') as f_in:
#            with open('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits', 'wb') as f_out:
#                shutil.copyfileobj(f_in, f_out)
        in_file = './data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.'+ex
        out_file = './data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits'
        extract(in_file,out_file)
        
        ss = hp.read_map('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits',verbose=0,nest=1)    
        load_status = 1        
        hp.mollview(ss, nest=1, cmap=cmap)    
        plt.savefig('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.jpg')
        plt.close()
        patches = sky2patch(ss, 1)
        for j in range(12):
            np.save('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j),patches[j])
            plt.imshow(patches[j], cmap=cmap)
            plt.savefig('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j)+'_'+str(fwhm_arcmin)+'.jpg',bbox_inches='tight')
            plt.close()
            
    if fwhm!=0.0:
        if load_status==0:
            ss = hp.read_map('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits',verbose=0,nest=1)
        print('Beaming string: '+str(i))
        ss = hp.sphtfunc.smoothing(ss,fwhm=fwhm)
        hp.write_map('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'_'+str(fwhm_arcmin)+'.fits', ss, overwrite=1)
        patches = sky2patch(ss, 1)
        for j in range(12):
            np.save('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j)+'_'+str(fwhm_arcmin),patches[j])
            plt.imshow(patches[j], cmap=cmap)
            plt.savefig('./data/string/map1n_allz_rtaapixlw_'+str(nside)+'_p'+str(12*i+j)+'_'+str(fwhm_arcmin)+'.jpg',bbox_inches='tight')
            plt.close()
        
            
        
