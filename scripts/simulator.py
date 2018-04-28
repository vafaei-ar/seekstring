import matplotlib as mpl
mpl.use('agg')

import os
import numpy as np
import pylab as plt
import urllib
import gzip
import shutil
import healpy as hp
from healpy import cartview

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)

n_gaussian = 10

nside = 2048
lmax = 2500
fwhm = 0.0
fwhm = fwhm*np.pi/(180*60)

cl = np.loadtxt('../data/cl_planck_lensed')
ll = cl[:lmax,0]
cl = cl[:lmax,1]

if not os.path.exists('../data/maps/gaussian/'):
	os.makedirs('../data/maps/gaussian/') 

for i in range(n_gaussian):

	if not os.path.exists('../data/maps/gaussian/'+'map_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.fits'):
		print('Simulation gaussian map: '+str(i))

		m,alms = hp.sphtfunc.synfast(cl, nside=nside, lmax=lmax, mmax=None, alm=True, pol=False, pixwin=False, fwhm=fwhm, sigma=None, new=1, verbose=0)
		cl_map = hp.sphtfunc.alm2cl(alms)

		hp.mollview(m, nest=0, cmap=cmap)
		hp.write_map('../data/maps/gaussian/'+'map_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.fits', m, overwrite=1)
		plt.savefig('../data/maps/gaussian/'+'map_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.jpg')
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
		plt.savefig('../data/maps/gaussian/power_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.jpg')
		plt.close()

if not os.path.exists('../data/maps/string/'):
	os.makedirs('../data/maps/string/') 
	

for i in range(3): 

	if not os.path.exists('../data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz'):
		print('Downloading string(s)...')
		urllib.urlretrieve('http://cp3.irmp.ucl.ac.be/~ringeval/upload/data/'+str(nside)+'/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz',
		  '../data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz')

	if not os.path.exists('../data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits'):
		with gzip.open('../data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits.gz', 'rb') as f_in:
			with open('../data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits', 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
				
	if fwhm!=0.0:
		ss = hp.read_map('../data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'.fits')
		ss = hp.sphtfunc.smoothing(ss,fwhm=fwhm)
		hp.write_map('../data/maps/string/map1n_allz_rtaapixlw_'+str(nside)+'_'+str(i+1)+'_'+str(fwhm)+'.fits', ss, overwrite=1)
		
			
			

