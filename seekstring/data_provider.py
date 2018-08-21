import os
import numpy as np
import healpy as hp

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
        assert n_cycle!=0, 'Please! n_cycle have to be greater than 1!'
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
        if self.n_call%self.n_cycle==0:
            self.cycle()
        self.n_call += 1
        
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

class Data_Provider2(object):
    
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

    def __init__(self,files,wx,wy,fx_files=None,n_chunk=10,
                 n_cycle = None, use_preprocess=True):
                 
        if n_cycle is None:
            n_cycle = 100000000
        self.n_cycle = n_cycle
        assert n_cycle!=0, 'Please! n_cycle have to be greater than 1!'
#        self.dtype = dtype
        self.n_call = 0
        
        if type(files) is not list:
            files = [files]
        self.files = files
        self.wx,self.wy = wx,wy
        self.n_files = len(files)
        
        self.nside = np.load(files[0]).shape[1]
        self.n_chunk = min(n_chunk,self.n_files)
        
        self.fx = fx_files is not None
        self.fx_files = fx_files
        
        mmx = []
        mmn = []
        for i in self.files:
            d = np.load(i)
            mmx.append(d.max())
            mmn.append(d.min())

        self.min = np.min(mmn)
        self.max = np.max(mmx)
        self.use_preprocess = use_preprocess
        
        if self.fx:
            assert self.n_files==len(fx_files),'Incompatibility in number of FX maps.'
            
#            nest_ordering = fits_hdu.header.get('ORDERING')

#        print("Data Loaded:\n\tpatch number=%d\n\tsize in byte=%d" % (self.n_patch, self.patchs.nbytes))
#        print("\tmin value=%f\n\tmax value=%f\n\tmean value=%f\n\tSTD value=%f" % (self.min, self.max, self.mean, self.std))

#    def read_file(self,file_name):
#        m = hp.read_map(file_name,dtype=self.dtype,verbose=0,nest=1)
#        self.mean = np.mean(m)
#        self.std = np.std(m)
#        return sky_to_patch(m,1,12,self.nside)

        print("\tmin value=%f\n\tmax value=%f\n\t" % (self.min, self.max))

    def preprocess(self, inp):
        scl = float(self.max - self.min)
        return ((inp - self.min) / scl)

    def cycle(self):
        indx = np.arange(self.n_files)
        np.random.shuffle(indx)
        self.patchs = []
        if self.fx:
            self.fx_patchs = []
        for j in range(self.n_chunk):
            self.patchs.append(np.load(self.files[indx[j]]))
            if self.fx:
                self.fx_patchs.append(np.load(self.fx_files[indx[j]]))
        
#        self.patchs = self.read_file(self.file_name)   

        self.patchs = np.array(self.patchs)
        if self.fx:
            self.fx_patchs = np.array(self.fx_patchs)
            
        if self.use_preprocess:
            self.patchs = self.preprocess(self.patchs)
            
    def __call__(self,num=1):
        nside = self.nside
        wx,wy = self.wx,self.wy
        if self.n_call%self.n_cycle==0:
            self.cycle()
        self.n_call += 1
        
        assert max(wx,wy)<nside,'ERROR!'
        x = np.zeros((num,wx,wy))
        if self.fx:
            y = np.zeros((num,wx,wy))

        for i in range(num):
            face = np.random.randint(self.n_chunk)
            i0 = np.random.randint(nside-wx)
            j0 = np.random.randint(nside-wy)
            rr = np.random.randint(4)
            rf = np.random.randint(2)
            
            xx = self.patchs[face,i0:i0+wx,j0:j0+wy]
            xx = np.rot90(xx,rr)
            if self.fx:
                yy = self.fx_patchs[face,i0:i0+wx,j0:j0+wy]
                yy = np.rot90(yy,rr)
            if 0 == rf:
                xx = np.flip(xx,0)
                if self.fx:
                    yy = np.flip(yy,0)

            x[i,:,:] = xx
            if self.fx:
                y[i,:,:] = yy

        if self.fx:
            return np.expand_dims(x,axis=-1),np.expand_dims(y,axis=-1)
        else:
            return np.expand_dims(x,axis=-1)

