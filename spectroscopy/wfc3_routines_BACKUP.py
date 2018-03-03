import numpy as np
import time
import matplotlib.pyplot as plt
import os, pdb, sys
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import glob
import atpy

# TODO - wrap these routines up as attributes of spectroscopy
# opjects somehow.


def calc_spectra_variations( spectra, ref_spectrum, max_wavshift=5, dwav=0.01, smoothing_fwhm=None, disp_bound_ixs=[] ):
    """
    GIVEN SPECTRA EXTRACTED FROM EACH FRAME AND A REFERENCE
    SPECTRUM, WILL HORIZONTALLY SHIFT AND VERTICALLY STRETCH
    EACH OF THE FORMER TO GIVE THE CLOSEST MATCH TO THE 
    REFERENCE SPECTRUM, AND RETURNS THE SHIFTS AND STRETCHES.
    NOTE THAT DWAV IS IN UNITS OF PIXELS, AND THE RETURNED 
    WAVSHIFTS VARIABLE IS ALSO IN UNITS OF PIXELS.

    Returns:
      dspec - nframes x ndisp array containing the rescaled shifted spectrum minus the 
          reference spectrum.
      wavshifts - The amounts by which the reference spectrum had to be shifted along the shift
          in pixels along the dispersion axis in pixels to match the individual spectra.
      vstretches - The amounts by which the reference spectrum have to be vertically stretched
          to give the best match to the individual spectra.
    """

    frame_axis = 0 # TODO = this should call the object property
    disp_axis = 1 # TODO = this should call the object property
    nframes, ndisp = np.shape( spectra ) # TODO = this should call the object property

    # Convert smoothing fwhm to the standard deviation of the
    # Gaussian kernel, and smooth the reference spectrum:
    if smoothing_fwhm!=None:
        smoothing_sig = smoothing_fwhm/2./np.sqrt( 2.*np.log( 2. ) )
        ref_spectrum = scipy.ndimage.filters.gaussian_filter1d( ref_spectrum, smoothing_sig )
    else:
        smoothing_sig = None

    # Interpolate the reference spectrum on to a grid of
    # increments equal to the dwav shift increment:
    dwavs = np.r_[-max_wavshift:max_wavshift+dwav:dwav]
    nshifts = len( dwavs )
    pad = max_wavshift+1
    x = np.arange( ndisp )
    xi = np.arange( -pad, ndisp+pad )
    z = np.zeros( pad )
    ref_spectrumi = np.concatenate( [ z, ref_spectrum, z ] )
    #print 'aaaaa'
    interpf = scipy.interpolate.interp1d( xi, ref_spectrumi, kind='cubic' )
    #print 'bbbbb'
    shifted = np.zeros( [ nshifts, ndisp ] )
    for i in range( nshifts ):
        shifted[i,:] = interpf( x+dwavs[i] )

    # Now loop over the individual spectra and determine which
    # of the shifted reference spectra gives the best match:
    print '\nDetermining shifts and stretches:'
    wavshifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )
    dspec = np.zeros( [ nframes, ndisp ] )
    enoise = np.zeros( [ nframes, ndisp ] )
    ix0 = disp_bound_ixs[0]
    ix1 = disp_bound_ixs[1]
    A = np.ones([ndisp,2])
    coeffs = []
    for i in range( nframes ):
        print i+1, nframes
        rms_i = np.zeros( nshifts )
        diffs = np.zeros( [ nshifts, ndisp ] )
        vstretches_i = np.zeros( nshifts )
        for j in range( nshifts ):
            A[:,1] = shifted[j,:]
            b = np.reshape( spectra[i,:], [ ndisp, 1 ] )
            res = np.linalg.lstsq( A, b )
            c = res[0].flatten()
            fit = np.dot( A, c )
            vstretches_i[j] = c[1]
            diffs[j,:] = spectra[i,:] - fit
            rms_i[j] = np.sqrt( np.mean( diffs[j,:][ix0:ix1+1]**2. ) )
        ix = np.argmin( rms_i )
        dspec[i,:] = diffs[ix,:]#/ref_spectrum
        enoise[i,:] = np.sqrt( spectra[i,:] )#/ref_spectrum
        wavshifts[i] = dwavs[ix]
        vstretches[i] = vstretches_i[ix]
        print '--> wavshift={0:.3f}, vstretch={1:.3f}'.format( dwavs[ix], vstretches_i[ix] )
        if 0:
            plt.ion()
            plt.figure()
            plt.plot( dwavs, rms_i, '-ok' )
            plt.axvline( dwavs[ix], c='r' )
            plt.title( wavshifts[i] )
            pdb.set_trace()
            plt.close('all')
    return dspec, wavshifts, vstretches, enoise


def extract_spatscan_spectra( image_cube, ap_radius=60, ninterp=10000, cross_axis=0, disp_axis=1, frame_axis=2 ):
    """
    GIVEN IMAGES WILL CALCULATE THE CENTER OF THE SPATIAL
    SCAN AND INTEGRATE WITHIN SPECIFIED APERTURE ABOUT 
    THIS CENTER ALONG THE CROSS-DISPERSION AXIS.
    """
    
    z = np.shape( image_cube ) # maybe build this into object
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
        image = image_cube[:,:,i]
        # Extract the cross-dispersion profile, i.e. along
        # the axis of the spatial scan:
        cdp = np.sum( image, axis=disp_axis )
        # Interpolate cross-dispersion profile to finer grid
        # in order to track sub-pixel shifts:
        cdpf = np.interp( xf, x, cdp )
        # Only consider points above the background level, 
        # otherwise blank sky will bias the result:
        thresh = cdp.min() + 0.05*( cdp.max()-cdp.min() )
        ixs = ( cdpf>thresh )
        # Determine the center of the scan by taking the
        # point midway between the edges:
        cdcs[i] = np.mean( xf[ixs] )
        # Determine the cross-dispersion coordinates between
        # which the integration will be performed:
        xmin = max( [ 0, cdcs[i] - ap_radius ] )
        xmax = min( [ cdcs[i] + ap_radius, ncross-1 ] )
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
        if xmin_full>0:
            xlow_partial = xmin_full - xmin
            spectra[i,:] += xlow_partial*image[xmin_full,:]
        if xmax_full+1<ncross:
            xupp_partial = xmax - xmax_full
            spectra[i,:] += xupp_partial*image[xmax_full+1,:]

        # DELETE BELOW
        #plt.figure()
        #plt.subplot( 121 )
        #plt.imshow( image, interpolation='nearest', aspect='auto', vmin=0 )
        #plt.axhline( cdcs[i], c='m' )
        #plt.subplot( 122 )
        #plt.plot( xf, cdpf, '-c' )
        #plt.plot( xf[ixs], cdpf[ixs], '-r', lw=2 )
        #plt.axvline( cdcs[i], c='k', ls='-' )
        #pdb.set_trace()
        # DELETE ABOVE

    return cdcs, spectra
