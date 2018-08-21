import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pylab as plt
import argparse
from matplotlib.colors import LogNorm
from seekstring.toy_models import Simple_String
from ccgpack import ch_mkdir,filters,StochasticFieldSimulator

parser = argparse.ArgumentParser()
parser.add_argument('--nside', action="store", type=int, required=1)
parser.add_argument('-i', action="store", type=int, required=1)

parser.add_argument('--test', action="store_true", required=False, default=False)

parser.add_argument('--size', action="store", type=int, default=50)
parser.add_argument('--nstring', action="store", type=int, default=200)

args = parser.parse_args()
nside = args.nside
i = args.i
if args.test:
    mode = 'test'
else:
    mode = 'training'
size = args.size
nstring = args.nstring

cmap = plt.cm.jet
cmap.set_under('w')
cmap.set_bad('gray',1.)

def Xtractor(m, crv, filt):
    mp = filters(m, edd_method=filt)
    return mp

ss = Simple_String(nx=nside,ny=nside)

cl = np.loadtxt('../../data/cl_planck_lensed')
sfs = StochasticFieldSimulator(cl)
ch_mkdir(mode+'_set')

print (mode+' set simulation...')

print('string ...')
x = ss.string(ns=nstring)
np.save(mode+'_set/s_'+str(i),x)
plt.imshow(x,cmap=cmap)
plt.savefig(mode+'_set/s_'+str(i)+'.jpg',bbox_inches='tight')
plt.close()

print('feature ...')
fx = Xtractor(x, 7, 'sch')
np.save(mode+'_set/f_'+str(i),fx)
plt.imshow(fx,cmap=cmap)
plt.savefig(mode+'_set/f_'+str(i)+'.jpg',bbox_inches='tight')
plt.close()

print('gaussian ...')
x = sfs.simulate(nside,size)
np.save(mode+'_set/g_'+str(i),x)
plt.imshow(x,cmap=cmap)
plt.savefig(mode+'_set/g_'+str(i)+'.jpg',bbox_inches='tight')
plt.close()

