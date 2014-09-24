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
    ncross, ndisp, nimages = np.shape( images ) # maybe build this into object
    spectra = np.zeros( [ nimages, ndisp ] )
    cross_centers = np.zeros( nimages )
    x = np.arange( ncross )
    
    for i in range( nimages ):
        # Extract the cross-dispersion profile, i.e. along
        # the axis of the spatial scan:
        cross_profile = np.sum( images[i], axis=1 )
        # Determine the center of the scan by taking the
        # point midway between the edges:
        threshold = 0.2*np.median( spatial_profile )
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
        spectra[i,:] = np.sum( images[ixs_full,:,i] )
        # Determine any rows that are partially contained
        # within the aperture at either end of the scan and
        # add their weighted contributions to the spectrum:
        xlow_partial = xmin_full - xmin
        spectra[:,i] += images[xlow_partial,:,i]
        xupp_partial = xmax - xmax_full
        spectra[:,i] += images[xupp_partial,:,i]

    data = { 'mjd':mjd, 'spectra':spectra, 'centers':centers }

    return data
