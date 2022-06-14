import os
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from flowdec.nb import utils as nbutils 
from flowdec import psf as fd_psf
from flowdec import data as fd_data
from scipy import ndimage
import dask
import dask.array as da
import tensorflow as tf
from flowdec.restoration import RichardsonLucyDeconvolver
from skimage import io
from pathlib import Path
import operator

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def ISS_flowdec(main_stacked_folder, output_main, PSF_metadata):
    
    """
    
    main_stacked_folder: 
    
    output_main: 
    
    PSF_metadata: 
    
    
    """
    stacked_folders = os.listdir(main_stacked_folder)
    os.makedirs(output_main + '/preprocessing/mipped/')
    for cyc, folder in enumerate(stacked_folders):
        os.makedirs(output_main + '/preprocessing/mipped/Base_'+ str(cyc+1))
        outfoldercyc=output_main + '/preprocessing/mipped/Base_'+ str(cyc+1)+'/'
        files = os.listdir(main_stacked_folder+'/'+folder)
        acq = io.imread(main_stacked_folder+'/'+folder+'/'+files[0])
        shape_im = acq.shape
        zsize=shape_im[0]
        
        
        psf_dictionary = {}
        print ('Generating PSFs')
        for i in PSF_metadata['channels']:
            print ('Generating PSF for channel:' + str(i))
            psf_dictionary[i] = fd_psf.GibsonLanni(na = float(PSF_metadata['na']),
                                                           m = float(PSF_metadata['m']),
                                                           ni0 = float(PSF_metadata['ni0']),
                                                           res_lateral = float(PSF_metadata['res_lateral']),
                                                          res_axial =  float(PSF_metadata['res_axial']),
                                                          wavelength = float(PSF_metadata['channels'][i]['wavelength']),
                                                          size_x=shape_im[2], 
                                                            size_y=shape_im[1], 
                                                            size_z=shape_im[0]).generate()
        for ch, i in enumerate(PSF_metadata['channels']):
            print ('Deconvolving channel:' +str(ch)+':'+str(i))
            psf = psf_dictionary[i]
            #print (psf)
            # iterate over files in
            # that directory
            files = Path(main_stacked_folder+'/'+folder).glob('*'+str(ch)+'.tif')

            for file in files:
                filestring=str(file)
                filename=filestring.split('stacked_')
                #print (filename[2])
                acq = io.imread(filestring)
                print('Reading: '+filename[1])
                stage=filename[1].split('--')[1].split('Stage')[1]
                newname=('Base_'+str(cyc+1)+'_s'+stage+'_C0'+str(ch)+'.tif')


                #define chunk size for small GPUs or CPU. Needs to be done for each channel as it chunks the PSF as well 
                chunk_size = (zsize,512,512)

                # chunked dask array
                arr = da.from_array(acq.data, chunks=chunk_size)

                # kernel cropped to chunk size
                # Marco: this is probably not optimal as kernel doesn't need to be chunked at every iteration per image
                # but rather at every channel change. Think about it
                cropped_kernel = cropND(psf, chunk_size)
                algo = RichardsonLucyDeconvolver(acq.data.ndim, pad_mode="2357", pad_min=(6,6,6))

                #the number at the end of the tmp line corresponds to the number of iterations
                def deconv(chunk):
                    # note that algo and cropped_kernel are from global scope ... ugly
                    print("chunk shape", chunk.shape)
                    tmp = algo.initialize().run(fd_data.Acquisition(data=chunk, kernel=cropped_kernel), 50)
                    return tmp.data 
                result_overlap = arr.map_overlap(deconv,depth=(6,6,6), boundary='reflect', dtype='uint16').compute(num_workers=1)
                print ('Deconvolution done, now doing max-projection...')
                result_maxproj= np.max(result_overlap, axis=0)
                print ('Saving projected image from: '+filename[1]+ " to "+newname )       
                tifffile.imsave(outfoldercyc+newname, result_maxproj.astype('uint16'))