import os
import numpy as np
import healpy as hp
from sky2face import sky_to_patch

class Data_Provider(object):
    
    """
    CLASS GeneralDataProvider: This class will provide data to feed network.
    --------
    METHODS:
    __init__:
    | arguments:
    |        files_list: list of paths to the maps. 
    |        dtype (default=np.float16): 
    |        nest (default=1): 
    |        lp (default=4096):  
    
    __call__:
    This method provides 
    | Arguments:
    |        num: number of returned patches.
    |        l: number of returned patches.
    | Returns:
    |        Image, Demand map, coordinates (if coord is true)
    """

    def __init__(self,files,
                 n_cycle = None,
                 dtype = np.float16):
                 
        npatch = 1
        numpa = 12
        if n_cycle is None:
            n_cycle = 100
        self.n_cycle = n_cycle
        self.dtype = dtype
        self.n_call = 0
        
        if type(files) is not list:
            files = [files]
        self.files = files
        n_files = len(files)
        
        fits_hdu = hp.fitsfunc._get_hdu(files[0], hdu=1, memmap=False)
        self.nside = fits_hdu.header.get('NSIDE')
        self.n_patch = 12
#            nest_ordering = fits_hdu.header.get('ORDERING')

#        print("Data Loaded:\n\tpatch number=%d\n\tsize in byte=%d" % (self.n_patch, self.patchs.nbytes))
#        print("\tmin value=%f\n\tmax value=%f\n\tmean value=%f\n\tSTD value=%f" % (self.min, self.max, self.mean, self.std))

    def read_file(self,file_name):
        m = hp.read_map(file_name,dtype=self.dtype,verbose=0,nest=1)
        self.mean = np.mean(m)
        self.std = np.std(m)
        return sky_to_patch(m,1,12,self.nside)

    def cycle(self):
        self.file_name = np.random.choice(self.files)
        print(self.file_name+' is under process ...')
        self.patchs = self.read_file(self.file_name)    

    def __call__(self,num,w_size):
        nside = self.nside
        self.n_call += 1
        if self.n_call%self.n_cycle==1:
            self.cycle()
        
        assert w_size<nside,'ERROR!'
        x = np.zeros((num,w_size,w_size))

        for i in range(num):
            face = np.random.randint(self.n_patch)
            i0 = np.random.randint(nside-w_size)
            j0 = np.random.randint(nside-w_size)
            xx = self.patchs[face,i0:i0+w_size,j0:j0+w_size]

            xx = np.rot90(xx,np.random.randint(4))
            if 0 == np.random.randint(2):
                xx = np.flip(xx,0)

            x[i,:,:] = xx

        return x
