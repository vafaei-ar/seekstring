import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pylab as plt
import urllib.request
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

print('Simulation gaussian maps...')

for i in range(n_gaussian):
	m,alms = hp.sphtfunc.synfast(cl, nside=nside, lmax=lmax, mmax=None, alm=True, pol=False, pixwin=False, fwhm=fwhm, sigma=None, new=1, verbose=0)
	cl_map = hp.sphtfunc.alm2cl(alms)

	hp.mollview(m, nest=0, cmap=cmap)
	hp.write_map('../data/maps/gaussian/'+'map_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.fits', m, overwrite=1)
	plt.savefig('../data/maps/gaussian/'+'map_'+str(nside)+'_'+str(fwhm)+'_'+str(i)+'.jpg')
	plt.close()

	plt.figure(figsize=(10,6))

	dl1 = []
	dl2 = []
	for i in range(ll.shape[0]):
		  dl1.append(ll[i]*(ll[i]+1)*cl[i]/(2*np.pi))
		  dl2.append(ll[i]*(ll[i]+1)*cl_map[i]/(2*np.pi))

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

print('Beginning file download with urllib2...')

print('Downloading string(s)...')
for i in range(3): 
	urllib.request.urlretrieve('http://cp3.irmp.ucl.ac.be/~ringeval/upload/data/2048/map1n_allz_rtaapixlw_2048_'+str(i)+'.fits.gz',
    '../data/maps/string/'+str(i+1)+'.fits.gz')

    with gzip.open('../data/maps/string/'+str(i)+'.fits.gz', 'rb') as f_in:
        with open('../data/maps/string/'+str(i)+'.fits', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

