import matplotlib as mpl
mpl.use('agg')

import os
import argparse
import numpy as np
import pylab as plt
import seekstring as ssg
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', action="store", type=int, default=120)
parser.add_argument('--learning_rate', action="store", type=float, default=0.01)
parser.add_argument('--train', action="store_true", default=False)

args = parser.parse_args()
time_limit = args.time_limit
learning_rate = args.learning_rate

w_size = 128
model_add = 'model'

files = ['training_set/s_'+str(i)+'.npy' for i in range(51)]
fx_files = ['training_set/f_'+str(i)+'.npy' for i in range(51)]

dp = ssg.Data_Provider2(files=files,fx_files=fx_files,
                        wx=w_size,wy=w_size)

conv = ssg.ConvolutionalLayers(nx=w_size,ny=w_size,n_channel=1,
                               restore=os.path.exists(model_add),
                               model_add=model_add,
                               arch_file_name='arch_0')

if args.train:
    for i in range(time_limit/30):
        print('Training stage: '+str(i))
        conv.train(data_provider=dp,training_epochs = 10000000,
                   n_s = 100,learning_rate = learning_rate,
                   dropout=0.7, time_limit=time_limit, verbose=1)
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
        np.save(pred_dir+fname,pred)

        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,8))
        ax1.imshow(x[0,:,:,0],aspect='auto')
        ax2.imshow(y,aspect='auto')
        ax3.imshow(pred,aspect='auto')

        plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        plt.savefig(pred_dir+fname+'.jpg',dpi=30)
        plt.close()  
    print np.mean(times)     

#print('FX is initiating ...')
#ch_mkdir('./data/features')


#s_files = glob.glob('data/maps/string/*.npy')
#ns = len(s_files)
#assert ns!=0, 'Error! you may need to run simulator.py first.'

#nside = np.load(s_files[0]).shape[0]
#print('Number of available patches:',ns)

#nprt = 0
#for i in range(256,2048,256):
#    for j in range(256,2048,256):
#        nprt += 1
#nprt *= ns

#prc = 0
#for ip in range(ns):
#    string = np.load(s_files[ip]) 
#    
##   If you want to test extractor un-comment one of the bellow tests:
#    # TEST one:
##    string = np.zeros(string.shape) 
##    string[:, ::80] = 1 
##    string[:, 1::80] = 1 
##    string[::80, :] = 1 
##    string[1::80, :] = 1 
#    
#    # TEST two:
##    from skimage.draw import circle_perimeter
##    for radius in [100,300,500,700,1000]:
##        rr, cc = circle_perimeter(1024, 1024, radius, shape=None)  
##        string[rr, cc] = 1
#    fxmap = np.zeros(string.shape)  
#    
#    # loops are sliding over image to avoid boundary effects.
#    for i in range(256,2048,256):
#        for j in range(256,2048,256):
#            ss = string[i-256:i+256,j-256:j+256]
##            print i-256,i+256,'----',j-256,j+256
##            print i-128,i+128,'----',j-128,j+128
##            print '-----------------------------'
#            
#            fx = Xtractor(ss, crv, filt)
#            fxmap[i-128:i+128,j-128:j+128] = fx[256-128:256+128,256-128:256+128]
#            
#            if i==256:
#                fxmap[i-256:i-128,j-128:j+128] = fx[:128,256-128:256+128]
#            elif i==2048-256:
#                fxmap[i+128:i+256,j-128:j+128] = fx[256+128:,256-128:256+128]
#            if j==256:
#                fxmap[i-128:i+128,j-256:j-128] = fx[256-128:256+128,:128]
#            elif j==2048-256:
#                fxmap[i-128:i+128,j+128:j+256] = fx[256-128:256+128,256+128:]

#            if i==256 and j==256:  
#                fxmap[:128,:128] = fx[:128,:128]
#            elif i==2048-256 and j==256:  
#                fxmap[2048-128:,:128] = fx[512-128:,:128]   
#            elif i==256 and j==2048-256:  
#                fxmap[:128,2048-128:] = fx[:128,512-128:] 
#            elif i==2048-256 and j==2048-256:  
#                fxmap[2048-128:,2048-128:] = fx[512-128:,512-128:] 
#                    
#            prc += 1
#            sys.stdout.write("\r{:3.2f}%".format(100.*prc/nprt)+ ' complete')
#            sys.stdout.flush()

#    fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(10,4))
#    ax1.imshow(string, cmap=cmap)
#    ax2.imshow(fxmap, cmap=cmap)
#    fname = s_files[ip].split('/')[-1][:-4]
#    plt.savefig('./data/features/fx_'+fname+'.jpg',bbox_inches='tight',dpi=200)
#    plt.close()
#    
#    fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(10,4))
#    ax1.imshow(string, cmap=cmap, norm=LogNorm())
#    ax2.imshow(fxmap, cmap=cmap, norm=LogNorm())
#    fname = s_files[ip].split('/')[-1][:-4]
#    plt.savefig('./data/features/fx_log_'+fname+'.jpg',bbox_inches='tight',dpi=200)
#    plt.close()

#    np.save('./data/features/fx_'+fname,fxmap)
#            

#print('\nFX making is completed.')


