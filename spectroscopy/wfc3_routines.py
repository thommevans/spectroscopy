import numpy as np
import time
import matplotlib.pyplot as plt
import os, pdb, sys
import scipy.optimize
import scipy.interpolate
import glob
import fitsio
import atpy

# TODO - wrap these routines up as attributes of spectroscopy
# opjects somehow.

def calc_spectra_variations( spectra, ref_spectrum, max_wavshift=5, dwav=0.01 ):
    """
    GIVEN SPECTRA EXTRACTED FROM EACH FRAME AND A REFERENCE
    SPECTRUM, WILL HORIZONTALLY SHIFT AND VERTICALLY STRETCH
    EACH OF THE FORMER TO GIVE THE CLOSEST MATCH TO THE 
    REFERENCE SPECTRUM, AND RETURNS THE SHIFTS AND STRETCHES.
    NOTE THAT DWAV IS IN UNITS OF PIXELS, AND THE RETURNED 
    WAVSHIFTS VARIABLE IS ALSO IN UNITS OF PIXELS.
    """
    
    frame_axis = 0 # TODO = this should call the object property
    disp_axis = 1 # TODO = this should call the object property
    nframes, ndisp = np.shape( spectra ) # TODO = this should call the object property
    dashifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )

    # Interpolate the reference spectrum on to a grid with
    # increments equal to the dwav shift increment:
    ninterp = int( np.round( ndisp/float( dwav ) ) )
    x = np.arange( ndisp )
    xf = np.r_[ x.min():x.max():1j*ninterp ]
    reff = np.interp( xf, x, ref_spectrum )

    # Determine the shift increments along the wavelength
    # axis that will be trialled:
    ix0 = -int( np.round( max_wavshift/dwav ) )
    nshifts = 2*int( np.round( max_wavshift/float( dwav ) ) ) + 1
    
    # Convert the wavelenegth shifts to pixel increments:
    shifts0 = ( ix0 + np.arange( nshifts ) )*dwav

    # Determine the shifts for each frame:
    wavshifts = np.zeros( nframes )
    for j in range( nframes ):

        print 'Frame {0} of {1}'.format( j+1, nframes )

        # Interpolate the current science spectrum on to 
        # the fine-scale grid of shift increments:
        specf = np.interp( xf, x, spectra[j,:] )
        y = np.reshape( specf, [ ninterp, 1 ] )

        # Loop over each trial shift:
        rms = np.zeros( nshifts )
        vstretches_j = np.zeros( nshifts )
        for i in range( nshifts ):
            new_spec = np.zeros( ninterp )

            # Determine where the lower edge of the 
            # shifted spectrum will be located on the 
            # fine-scale grid:
            ix = ix0+i
            if ix<0:
                # Case 1 = The lower edge of the spectrum
                # falls off the lower edge of the scale:
                new_spec[:ix] = reff[np.abs(ix):]
                new_spec[ix:] = reff[-1]
            elif ix==0:
                # Case 2 = The lower edge of the spectrum
                # exactly coincides with the lower edge
                # of the scale:
                new_spec = reff
            elif ( ix>0 )*( ix<ninterp ):
                # Case 3 = The lower edge of the spectrum
                # falls within the scale:
                new_spec[ix:] = reff[:-np.abs(ix)]
                new_spec[:ix] = reff[0]
            else:
                # Case 4 = The spectrum falls outside the
                # scale, which is nonsensical:
                pdb.set_trace()
            # Use linear least squares to determine the
            # optimal vertical stretch for the shifted
            # spectrum to match the reference spectrum:
            A = np.reshape( new_spec, [ ninterp, 1 ] )
            z = np.linalg.lstsq(A,y)
            vstretches_j[i] = z[0]
            # RMS between shifted+stretched spectrum and 
            # the reference specrum:
            rms[i] = float( np.sqrt( z[1]/float( ninterp ) ) )
        # Identify the shift that minimised the rms:
        ix = np.argmin( rms )
        wavshifts[j] = shifts0[ix]
        vstretches[j] = vstretches_j[ix]

    return wavshifts, vstretches

def extract_spatscan_specra( images, ap_radius=60, ninterp=10000 ):
    """
    GIVEN IMAGES WILL CALCULATE THE CENTER OF THE SPATIAL
    SCAN AND INTEGRATE WITHIN SPECIFIED APERTURE ABOUT 
    THIS CENTER ALONG THE CROSS-DISPERSION AXIS.
    """
    
    cross_axis = 0
    disp_axis = 1
    frame_axis = 2
    z = np.shape( images ) # maybe build this into object
    ncross = z[cross_axis]
    ndisp = z[disp_axis]
    nframes = z[frame_axis]
    spectra = np.zeros( [ nframes, ndisp ] )
    cdcs = np.zeros( nframes )
    x = np.arange( ncross )
    nf = int( ninterp*len( x ) )
    xf = np.r_[ x.min():x.max():1j*nf ]

    print '\nExtracting spectra from 2D images:'
    for i in range( nframes ):
        print '... image {0} of {1}'.format( i+1, nframes )
        image = images[:,:,i]
        # Extract the cross-dispersion profile, i.e. along
        # the axis of the spatial scan:
        cdp = np.sum( image, axis=disp_axis )
        # Interpolate cross-dispersion profile to finer grid
        # in order to track sub-pixel shifts:
        cdpf = np.interp( xf, x, cdp )
        # Determine the center of the scan by taking the
        # point midway between the edges:
        thresh = 0.2*np.median( cdp )
        ixs = ( cdpf>thresh )
        cdcs[i] = np.mean( xf[ixs] )
        #print 'cross_center ', i, cdcs[i], xf[ixs].min(), xf[ixs].max()
        # Determine the cross-dispersion coordinates between
        # which the integration will be performed:
        xmin = cdcs[i] - ap_radius
        xmax = cdcs[i] + ap_radius
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

    return cdcs, spectra
