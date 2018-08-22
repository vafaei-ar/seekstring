import os
import argparse
import numpy as np
import healpy as hp
import pylab as plt

import seekstring as ss

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--gmu', action="store", type=float, default=0.5)
parser.add_argument('--noise', action="store", type=float, default=0.0)
parser.add_argument('--time_limit', action="store", type=int, default=60)
parser.add_argument('--arch', action="store", type=int, default=1)
parser.add_argument('--learning_rate', action="store", type=float, default=0.0001)
parser.add_argument('--train', action="store_true", default=False)
parser.add_argument('--pp', action="store_true", default=False)

parser.add_argument('-r', action="store_true", default=False)
parser.add_argument('--nsim', action="store", type=int, default=10)
parser.add_argument('--fwhm', action="store", type=float, default=1.0)
parser.add_argument('--nside', action="store", type=int, default=2048)
parser.add_argument('--lmax', action="store", type=int, default=3500)

args = parser.parse_args()
replace = args.r
nside = args.nside
lmax = args.lmax
fwhm = args.fwhm
n_gaussian = args.nsim
gmu = args.gmu
noise = args.noise
time_limit = args.time_limit
learning_rate = args.learning_rate

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

def dp_total(n):
    l = 200
    string = dp_string(n,l)
    gaussian = dp_gaussian(n,l)
    y = filt_all(string,func).reshape(n,l,l,1)
    x = (10*gaussian+string).reshape(n,l,l,1) 
    return x,y
    

dp_gaussian = ss.Data_Provider(gaussians,dtype = np.float32)
dp_string = ss.Data_Provider(strings,dtype = np.float32)

conv = ss.ConvolutionalLayers(nx=200,ny=200,n_channel=1,restore=os.path.exists(model_add),model_add=model_add,arch_file_name='arch')

if args.train:
    for i in range(5):
        print('Training stage: '+str(i))
        conv.train(data_provider=dp_total,training_epochs = 10000000,n_s = 100,learning_rate = learning_rate, dropout=0.7, time_limit=time_limit, verbose=1)
        learning_rate = learning_rate/5.

else:
    pred_dir = 'predictions/'
    ssg.ch_mkdir(pred_dir)

#    import pickle
#    res_file = 'results/'
#    ssg.ch_mkdir(res_file)
#    weights = conv.get_filters()
#    with open(res_file+'_filters', 'w') as filehandler:
#        pickle.dump(weights, filehandler)

    files = ['test_set/s_'+str(i)+'.npy' for i in range(11)]
    fx_files = ['test_set/f_'+str(i)+'.npy' for i in range(11)]
    num = len(files)
    
    times = []

    for i in range(num):
        fil = files[i]
        xfil = fx_files[i]

        fname = fil.split('/')[-1]
        x = np.load(fil)
        x = np.expand_dims(x,axis=0)
        x = np.expand_dims(x,axis=-1)
        y = np.load(xfil)
        
        s = time()
        pred = conv.conv_large_image(x,pad=10,lx=w_size,ly=w_size)
        e = time()
        times.append(e-s)
        np.save(pred_dir+fname)

        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,8))
        ax1.imshow(x[0,:,:,0],aspect='auto')
        ax2.imshow(y,aspect='auto')
        ax3.imshow(pred,aspect='auto')

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'.jpg',dpi=30)
        plt.close()  
    print np.mean(times)     



