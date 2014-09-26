import numpy as np
import time
import matplotlib.pyplot as plt
import os, pdb, sys
import scipy.optimize
import scipy.interpolate
import glob
import fitsio
import atpy

def extract_spatscan_specra( images, ap_radius=60 ):

    # TODO - wrap this up as spectroscopy object somehow
    cross_axis = 0
    disp_axis = 1
    frame_axis = 2
    z = np.shape( images ) # maybe build this into object
    ncross = z[cross_axis]
    ndisp = z[disp_axis]
    nframes = z[frame_axis]
    spectra = np.zeros( [ nframes, ndisp ] )
    cross_centers = np.zeros( nframes )
    x = np.arange( ncross )
    
    print '\nExtracting spectra from 2D images:'
    for i in range( nframes ):
        print '... image {0} of {1}'.format( i+1, nframes )
        image = images[:,:,i]
        # Extract the cross-dispersion profile, i.e. along
        # the axis of the spatial scan:
        cross_profile = np.sum( image, axis=disp_axis )
        # Determine the center of the scan by taking the
        # point midway between the edges:
        threshold = 0.2*np.median( cross_profile )
        ixs = ( cross_profile>threshold )
        cross_centers[i] = np.mean( x[ixs] )
        # Determine the cross-dispersion coordinates between
        # which the integration will be performed:
        xmin = cross_centers[i] - ap_radius
        xmax = cross_centers[i] + ap_radius
        # Determine the rows that are fully contained
        # within the aperture and integrate along the
        # cross-dispersion axis:
        xmin_full = int( np.ceil( xmin ) )
        xmax_full = int( np.floor( xmax ) )
        ixs_full = ( x>=xmin_full )*( x<=xmax_full )
        spectra[i,:] = np.sum( image[ixs_full,:], axis=cross_axis )
        # Determine any rows that are partially contained
        # within the aperture at either end of the scan and
        # add their weighted contributions to the spectrum:
        xlow_partial = xmin_full - xmin
        spectra[i,:] += image[xlow_partial,:]
        xupp_partial = xmax - xmax_full
        spectra[i,:] += image[xupp_partial,:]
    pdb.set_trace()

    return spectra
