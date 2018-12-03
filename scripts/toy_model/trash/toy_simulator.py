import matplotlib as mpl
mpl.use('agg')

import numpy as np
import pylab as plt
import argparse
from matplotlib.colors import LogNorm
from seekstring.toy_models import Simple_String
from ccgpack import ch_mkdir,filters,StochasticFieldSimulator

parser = argparse.ArgumentParser()
parser.add_argument('--nside', action="store", type=int, default=300)
parser.add_argument('-n', action="store", type=int, default=10)

parser.add_argument('--test', action="store_true", required=False, default=False)

parser.add_argument('--size', action="store", type=int, default=50)
parser.add_argument('--nstring', action="store", type=int, default=100)

args = parser.parse_args()
nside = args.nside
num = args.n
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

mode = 'training'
ch_mkdir(mode+'_set')
print (mode+' set simulation...')

for i in range(num):
    print(i)
    x = ss.string(ns=nstring)
    x -= x.min()
    x /= x.max()
    np.save(mode+'_set/s_'+str(i),x)
    plt.imshow(x,cmap=cmap)
    plt.savefig(mode+'_set/s_'+str(i)+'.jpg',bbox_inches='tight')
    plt.close()
    print('string is saved.')

    fx = Xtractor(x, 7, 'sch')
    fx -= fx.min()
    fx /= fx.max()
    np.save(mode+'_set/f_'+str(i),fx)
    plt.imshow(fx,cmap=cmap)
    plt.savefig(mode+'_set/f_'+str(i)+'.jpg',bbox_inches='tight')
    plt.close()
    print('feature is saved.')

    x = sfs.simulate(nside,size)
    x -= x.min()
    x /= x.max()
    np.save(mode+'_set/g_'+str(i),x)
    plt.imshow(x,cmap=cmap)
    plt.savefig(mode+'_set/g_'+str(i)+'.jpg',bbox_inches='tight')
    plt.close()
    print('gaussian is saved.')

mode = 'test'
ch_mkdir(mode+'_set')
print (mode+' set simulation...')

for i in range(num//3):
    print(i)
    x = ss.string(ns=nstring)
    x -= x.min()
    x /= x.max()
    np.save(mode+'_set/s_'+str(i),x)
    plt.imshow(x,cmap=cmap)
    plt.savefig(mode+'_set/s_'+str(i)+'.jpg',bbox_inches='tight')
    plt.close()
    print('string is saved.')

    fx = Xtractor(x, 7, 'sch')
    fx -= fx.min()
    fx /= fx.max()
    np.save(mode+'_set/f_'+str(i),fx)
    plt.imshow(fx,cmap=cmap)
    plt.savefig(mode+'_set/f_'+str(i)+'.jpg',bbox_inches='tight')
    plt.close()
    print('feature is saved.')

    x = sfs.simulate(nside,size)
    x -= x.min()
    x /= x.max()
    np.save(mode+'_set/g_'+str(i),x)
    plt.imshow(x,cmap=cmap)
    plt.savefig(mode+'_set/g_'+str(i)+'.jpg',bbox_inches='tight')
    plt.close()
    print('gaussian is saved.')
