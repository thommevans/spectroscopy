import numpy as np
import pdb, os, sys
import scipy.ndimage
import scipy.interpolate
import matplotlib.pyplot as plt



def shiftstretch( spectra, ref_spectrum, max_wavshift=5, dwav=0.01, disp_bound_ixs=[] ):
    """
    Horizontally shift and vertically stretch spectra to give best-fit to a reference spectrum.

    Inputs:
    ** spectra - 2D array containing the time series spectra in electrons.
    ** ref_spectrum - 1D array containing the reference spectrum.
    ** max_wavshift - The maximum number of pixels to horizontally shift each spectra
      by when comparing to the reference spectrum.
    ** dwav - Horizontal shift increments, in units of pixels.
    ** ixs - 2-element array, giving the start and end indices to use when calculating 
      the root-mean-square of the residuals for each fit to the reference spectrum. This
      makes it possible to exclude the edges, where the fit may be particularly poor 
      due to the need to extend the reference spectrum by linear extrapolation.
    """
    nframes, ndisp = np.shape( spectra )
    disp_shifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )
    nrange = int( np.round( max_wavshift/float( dwav ) ) )
    ntrials = 2*nrange + 1
    shifted = np.zeros( [ ndisp, ntrials ] )
    x = np.arange( -2, ndisp+2 )
    ref_spectrum_ext = np.concatenate( [ np.zeros( 2 ), ref_spectrum, np.zeros( 2 ) ] )
    interpf = scipy.interpolate.interp1d( x, ref_spectrum_ext, kind='cubic' )

    pixshifts = np.zeros( ntrials )
    for j in range( -nrange, nrange+1 ):
        dw = j*dwav
        shifted[:,j+nrange] = interpf( np.arange( ndisp )+dw )
        pixshifts[j+nrange] = dw
    dspec = np.zeros( [ nframes, ndisp ] )
    voff = np.zeros( nframes )
    vstretch = np.zeros( nframes )
    dispshift = np.zeros( nframes )
    for i in range( nframes ):
        A = np.ones( [ ndisp, 2 ] )
        print i+1, nframes
        diff = np.zeros( [ ntrials, ndisp ] )
        rms = np.zeros( ntrials )
        voff_i = np.zeros( ntrials )
        vstretch_i = np.zeros( ntrials )
        dispshift_i = np.zeros( ntrials )
        spec_i = spectra[i,:]
        for j in range( -nrange, nrange+1 ):
            A[:,1] = shifted[:,j+nrange]
            b = np.reshape( spec_i, [ ndisp, 1 ] )
            res = np.linalg.lstsq( A, b )
            c = res[0].flatten()
            fit = np.dot( A, c )
            voff_i[j] = c[0]
            vstretch_i[j] = c[1]
            diff[j+nrange,:] = spec_i - fit
            if disp_bound_ixs==[]:
                rms[j+nrange] = np.sqrt(np.mean(diff[j+nrange,:]**2))
            else:
                ix1 = disp_bound_ixs[0]
                ix2 = disp_bound_ixs[1]
                rms[j+nrange] = np.sqrt(np.mean(diff[j+nrange,ix1:ix2+1]**2))
        ix = np.argmin( rms )
        dspec[i,:] = diff[ix,:]/ref_spectrum
        voff[i] = voff_i[ix]
        vstretch[i] = vstretch_i[ix]
        dispshift[i] = pixshifts[ix]

    return dspec, voff, vstretch, dispshift
