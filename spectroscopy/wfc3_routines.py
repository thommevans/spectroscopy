import numpy as np
import time
import matplotlib.pyplot as plt
import os, pdb, sys
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import glob
import fitsio
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
    interpf = scipy.interpolate.interp1d( xi, ref_spectrumi, kind='cubic' )
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
        dspec[i,:] = diffs[ix,:]/ref_spectrum
        enoise[i,:] = np.sqrt( spectra[i,:] )/ref_spectrum
        wavshifts[i] = dwavs[ix]
        vstretches = vstretches_i[ix]
        
    return dspec, wavshifts, vstretches, enoise

def calc_spectra_variations_WORKING( spectra, ref_spectrum, max_wavshift=5, dwav=0.01, smoothing_fwhm=None, disp_bound_ixs=[] ):
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

    # Interpolate the reference spectrum on to a grid cor
    # increments equal to the dwav shift increment:
    dwavs = np.r_[-max_wavshift:max_wavshift+dwav:dwav]
    nshifts = len( dwavs )
    pad = max_wavshift+1
    x = np.arange( ndisp )
    xi = np.arange( -pad, ndisp+pad )
    z = np.zeros( pad )
    ref_spectrumi = np.concatenate( [ z, ref_spectrum, z ] )
    interpf = scipy.interpolate.interp1d( xi, ref_spectrumi, kind='cubic' )
    shifted = np.zeros( [ nshifts, ndisp ] )
    for i in range( nshifts ):
        shifted[i,:] = interpf( x+dwavs[i] )

    #######delete
    spec_deming = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_spec.txt')
    avspec_deming = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_avspec.txt')
    #######delete
    spectra = spec_deming.T
    ref_spectrum = avspec_deming
    nframes, ndisp = np.shape( spectra ) # TODO = this should call the object property
    xx = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_dwavs.txt').T
    shifted = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_shifted.txt').T
    nshifts, ndisp = np.shape( shifted )

    # Now loop over the individual spectra and determine which
    # of the shifted reference spectra gives the best match:
    wavshifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )
    dspec = np.zeros( [ nframes, ndisp ] )
    ix0 = disp_bound_ixs[0]
    ix1 = disp_bound_ixs[1]
    ####delete
    A=np.ones([ndisp,2])
    print '\nDetermining shifts and stretches:'
    hh=np.zeros(nframes)#delete
    zzrms=np.zeros([nframes,nshifts])
    ii1=-1000#delete
    ii2=1000#delete
    nrange0=1000#delete
    shifted=shifted.T#delete
    spec=spectra.T#delete
    avspec=ref_spectrum#delete
    dspec=dspec.T#delete
    for i in range( nframes ):
        print i+1, nframes
        rms_i = np.zeros( nshifts )
        diffs = np.zeros( [ nshifts, ndisp ] )
        for j in range( nshifts ):
            A[:,1] = shifted[:,j]
            b = np.reshape( spec[:,i], [ ndisp, 1 ] )
            res = np.linalg.lstsq( A, b )
            c = res[0].flatten()
            fit = np.dot( A, c )
            diffs[j,:] = spec[:,i] - fit
            rms_i[j] = np.sqrt(np.mean(diffs[j,:][ix0:ix1+1]**2))
        ix = np.argmin( rms_i )
        dspec[:,i] = diffs[ix,:]/avspec

    if 1:#delete this block
        zz = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_dspec.txt').T
        dspec=dspec.T
        print 'this should be the same --> ', np.shape(dspec), np.shape(zz)
        for ii in range(10):
            plt.close('all')
            plt.figure()
            plt.plot(dspec[ii+10,:],'.k')
            plt.plot(zz[ii+10,:],'.r')
            pdb.set_trace()

    return dspec, wavshifts, vstretches


def calc_spectra_variations_WORKINGDRAKECOPY( spectra, ref_spectrum, max_wavshift=5, dwav=0.01, smoothing_fwhm=None, disp_bound_ixs=[] ):
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

    #######delete
    spec_deming = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_spec.txt')
    avspec_deming = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_avspec.txt')
    if 0:
        
        n1=52+10 # +10 to remove first orbit, which was already done for shift_drake.py
        n2=100+10
        avspec_exeter= (np.sum(spectra[n1-10:n1-1,:],axis=0)+np.sum(spectra[n2+1:n2+10,:],axis=0))/20.0      
        plt.figure()
        plt.plot(avspec_deming/avspec_deming.max(),'-g')
        plt.plot(avspec_exeter/avspec_exeter.max(),'-r')
        plt.plot(ref_spectrum/ref_spectrum.max(),'-c')
        print spectra.max()
        print avspec_exeter.max()
        pdb.set_trace()
    #######delete

    # Interpolate the reference spectrum on to a grid cor
    # increments equal to the dwav shift increment:
    dwavs = np.r_[-max_wavshift:max_wavshift+dwav:dwav]
    nshifts = len( dwavs )
    pad = max_wavshift+1
    x = np.arange( ndisp )
    xi = np.arange( -pad, ndisp+pad )
    z = np.zeros( pad )
    ref_spectrumi = np.concatenate( [ z, ref_spectrum, z ] )
    interpf = scipy.interpolate.interp1d( xi, ref_spectrumi, kind='cubic' )
    shifted = np.zeros( [ nshifts, ndisp ] )
    for i in range( nshifts ):
        shifted[i,:] = interpf( x+dwavs[i] )

    #delete this block
    #print 'A', np.shape(spectra)
    #print 'B', np.shape(spec_deming)
    #pdb.set_trace()
    spectra = spec_deming.T
    ref_spectrum = avspec_deming
    nframes, ndisp = np.shape( spectra ) # TODO = this should call the object property
    xx = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_dwavs.txt').T
    shifted = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_shifted.txt').T
    nshifts, ndisp = np.shape( shifted )
    if 0:#delete this block
        print 'ddddd', np.max(np.abs(aa-dwavs))
        print np.shape(zz)
        print np.shape(shifted)
        plt.close('all')
        plt.plot(zz[0,:]/zz[0,:].max(),'-k')
        plt.plot(zz[-1,:]/zz[-1,:].max(),'-k')
        plt.plot(ref_spectrum/ref_spectrum.max(),'-g')
        plt.plot(avspec_deming/avspec_deming.max(),'-m')
        plt.plot(zz[0,:]/zz[0,:].max(),'.k')
        plt.plot(zz[-1,:]/zz[-1,:].max(),'.k')
        plt.plot(shifted[0,:]/shifted[0,:].max(),'-r')
        plt.plot(shifted[-1,:]/shifted[-1,:].max(),'-r')
        pdb.set_trace()
        for i in range( 10 ):
            plt.close('all')
            plt.figure()
            plt.plot(zz[10+i,:]/zz[10+i,:].max(),'-c')
            plt.plot(shifted[10+i,:]/shifted[10+i,:].max(),'-m')
            pdb.set_trace()
    

    # Now loop over the individual spectra and determine which
    # of the shifted reference spectra gives the best match:
    wavshifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )
    dspec = np.zeros( [ nframes, ndisp ] )
    ####delete
    #xx=np.loadtxt('/home/tevans/Desktop/delete.txt')#delete
    #print np.shape(spectra)
    #print np.shape(xx)
    #pdb.set_trace()
    lam1=71
    lam2=185
    ####delete
    A=np.ones([ndisp,2])
    print '\nDetermining shifts and stretches:'
    hh=np.zeros(nframes)#delete
    zzrms=np.zeros([nframes,nshifts])
    ii1=-1000#delete
    ii2=1000#delete
    nrange0=1000#delete
    shifted=shifted.T#delete
    spec=spectra.T#delete
    ndim=ndisp#delete
    avspec=ref_spectrum#delete
    dspec=dspec.T#delete
    for i in range( nframes ):
        print i+1, nframes
        rms_best=1e15  # initialize the best rms difference to a big number
        if i>=1:
            # we don't have to search the full range of shifts, since
            # it won't move that rapidly from one spectrum to the next
            nrange=300
        # make some limits so we don't overflow the dimensions
        #  search around the previous center by +/- nrange
        #  remember nrange0 is the orginal full range
        counter=0#delete
        for ii in range(ii1,ii2+1):
            #  a linear fit
            A[:,1] = shifted[:,ii+nrange0]
            #A = np.reshape( spec[:,i], [ ndim, 1 ] )
            b = np.reshape( spec[:,i], [ndim,1] )
            res = np.linalg.lstsq( A, b )
            c = res[0].flatten()
            #res=linfit(ytry,spec(*,i))
            #fit=res(0)+res(1)*ytry
            fit = np.dot( A, c )
            #  the residuals of the fit
            diff=(spec[:,i]-fit)/avspec.max()
            #  how much scatter they have
            #rms=np.sqrt(np.sum(diff[lam1:lam2+1]**2)/float((lam2-lam1+1)))
            rms=np.sqrt(np.mean(diff[lam1:lam2+1]**2))
            #zrms[i,ii+nrange0]=rms
            #zdiff[i,ii+nrange0]=rms
            if 0*(ii+nrange0==30):
                np.savetxt('A_delete.txt',A)
                np.savetxt('b_delete.txt',b)
                np.savetxt('c_delete.txt',res[0])
                np.savetxt('diff_delete.txt',diff)
                np.savetxt('shift_drake_spec.txt',spec)
                pdb.set_trace()
            if rms<rms_best:
                dspec[:,i]=avspec.max()*diff/avspec
                rms_best=rms
            counter+=1#delete



    if 1:#delete this block
        zz = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_dspec.txt').T
        dspec=dspec.T
        print 'this should be the same --> ', np.shape(dspec), np.shape(zz)
        for ii in range(10):
            plt.close('all')
            plt.figure()
            plt.plot(dspec[ii+10,:],'.k')
            plt.plot(zz[ii+10,:],'.r')
            pdb.set_trace()

    return dspec, wavshifts, vstretches


def calc_spectra_variations_WANT2WORK( spectra, ref_spectrum, max_wavshift=5, dwav=0.01, smoothing_fwhm=None, disp_bound_ixs=[] ):
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

    #######delete
    spec_deming = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_spec.txt')
    avspec_deming = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_avspec.txt')
    if 0:
        
        n1=52+10 # +10 to remove first orbit, which was already done for shift_drake.py
        n2=100+10
        avspec_exeter= (np.sum(spectra[n1-10:n1-1,:],axis=0)+np.sum(spectra[n2+1:n2+10,:],axis=0))/20.0      
        plt.figure()
        plt.plot(avspec_deming/avspec_deming.max(),'-g')
        plt.plot(avspec_exeter/avspec_exeter.max(),'-r')
        plt.plot(ref_spectrum/ref_spectrum.max(),'-c')
        print spectra.max()
        print avspec_exeter.max()
        pdb.set_trace()
    #######delete

    # Interpolate the reference spectrum on to a grid cor
    # increments equal to the dwav shift increment:
    dwavs = np.r_[-max_wavshift:max_wavshift+dwav:dwav]
    nshifts = len( dwavs )
    pad = max_wavshift+1
    x = np.arange( ndisp )
    xi = np.arange( -pad, ndisp+pad )
    z = np.zeros( pad )
    ref_spectrumi = np.concatenate( [ z, ref_spectrum, z ] )
    interpf = scipy.interpolate.interp1d( xi, ref_spectrumi, kind='cubic' )
    shifted = np.zeros( [ nshifts, ndisp ] )
    for i in range( nshifts ):
        shifted[i,:] = interpf( x+dwavs[i] )

    #delete this block
    #print 'A', np.shape(spectra)
    #print 'B', np.shape(spec_deming)
    #pdb.set_trace()
    spectra = spec_deming.T
    ref_spectrum = avspec_deming
    nframes, ndisp = np.shape( spectra ) # TODO = this should call the object property
    xx = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_dwavs.txt').T
    shifted = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_shifted.txt').T
    nshifts, ndisp = np.shape( shifted )
    if 0:#delete this block
        print 'ddddd', np.max(np.abs(aa-dwavs))
        print np.shape(zz)
        print np.shape(shifted)
        plt.close('all')
        plt.plot(zz[0,:]/zz[0,:].max(),'-k')
        plt.plot(zz[-1,:]/zz[-1,:].max(),'-k')
        plt.plot(ref_spectrum/ref_spectrum.max(),'-g')
        plt.plot(avspec_deming/avspec_deming.max(),'-m')
        plt.plot(zz[0,:]/zz[0,:].max(),'.k')
        plt.plot(zz[-1,:]/zz[-1,:].max(),'.k')
        plt.plot(shifted[0,:]/shifted[0,:].max(),'-r')
        plt.plot(shifted[-1,:]/shifted[-1,:].max(),'-r')
        pdb.set_trace()
        for i in range( 10 ):
            plt.close('all')
            plt.figure()
            plt.plot(zz[10+i,:]/zz[10+i,:].max(),'-c')
            plt.plot(shifted[10+i,:]/shifted[10+i,:].max(),'-m')
            pdb.set_trace()
    

    # Now loop over the individual spectra and determine which
    # of the shifted reference spectra gives the best match:
    wavshifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )
    dspec = np.zeros( [ nframes, ndisp ] )
    ####delete
    #xx=np.loadtxt('/home/tevans/Desktop/delete.txt')#delete
    #print np.shape(spectra)
    #print np.shape(xx)
    #pdb.set_trace()
    lam1=71
    lam2=185
    ####delete
    A=np.ones([ndisp,2])
    print '\nDetermining shifts and stretches:'
    hh=np.zeros(nframes)#delete
    zzrms=np.zeros([nframes,nshifts])
    for i in range( nframes ):
        print '... {0} of {1}'.format( i+1, nframes )
        # The current science spectrum:
        if smoothing_fwhm!=None:
            spec_i = scipy.ndimage.filters.gaussian_filter1d( spectra[i,:], smoothing_sig )
        else:
            spec_i = spectra[i,:]
        #A = np.reshape( spec_i, [ ndisp, 1 ] )
        #A[:,1]=spec_i
        b = np.reshape( spec_i, [ ndisp, 1 ] )
        rms_i = np.zeros( nshifts )
        voffs_i = np.zeros( nshifts )
        vstretches_i = np.zeros( nshifts )
        #if smoothing_sig!=None:
        #    # Smooth with a 1D Gaussian filter:
        #    spec_i = scipy.ndimage.filters.gaussian_filter1d( spec_i, smoothing_sig )
        diffs = []
        for j in range( nshifts ):
            #b = np.reshape( shifted[j,:], [ ndisp, 1 ] )
            A[:,1]=shifted[j,:]
            res = np.linalg.lstsq( A, b )
            c = res[0].flatten()
            fit = np.dot( A, c )
            diff = ( spectra[i,:]-fit )/ref_spectrum.max()
            rms_i[j]=np.sqrt(np.mean(diff[lam1:lam2+1]**2.))
            diffs += [ diff ]
            zzrms[i,j]=rms_i[j]
            if 0*( j==30 ):
                AA=np.loadtxt('A_delete.txt')
                bb=np.loadtxt('b_delete.txt')
                cc=np.loadtxt('c_delete.txt')
                diffdiff=np.loadtxt('diff_delete.txt')
                print c
                print cc
                print np.max(np.abs(diff-diffdiff))
                pdb.set_trace()
            #old stuff:
            if 0:
                voffs_i[j] = c[0][0]
                vstretches_i[j] = c[0][1]
                if disp_bound_ixs==[]:
                    rms_i[j] = float( np.sqrt( c[1]/float( ndisp ) ) )
                else:
                    fit = np.dot( A, c[0].flatten() )
                    resids = b.flatten()-fit
                    ix1 = disp_bound_ixs[0]
                    ix2 = disp_bound_ixs[1]
                    rms_i[j] = np.sqrt( np.mean( resids[ix1:ix2+1]**2. ) )
        ix = np.argmin( rms_i )
        print 'aaaaa', ix
        hh[i]=ix
        dspec[i,:] = diffs[ix]/ref_spectrum
        #old stuff:
        if 0:
            wavshifts[i] = dwavs[ix]
            vstretches[i] = vstretches_i[ix]
            #dspec[i,:] = ( spec_i*vstretches_i[ix] - shifted[ix,:] )/ref_spectrum
            bestcoeffs = np.reshape( np.array( [ voffs_i[ix], vstretches_i[ix] ] ), [ 2, 1 ] )
            A[:,1] = shifted[ix,:]
            bestfit = np.dot( A, bestcoeffs ).flatten()
            dspec[i,:] = ( bestfit-spec_i )/ref_spectrum
    hhh=np.loadtxt('hhh_delete.txt')
    zrms=np.loadtxt('zrms_delete.txt')
    print np.shape(hh)
    print np.shape(hhh)
    print np.shape(zrms)
    print np.shape(zzrms)
    pdb.set_trace()
    #plt.figure()
    #for i in range(10):
    #    plt.plot(dspec[i,:],'-')
    #    #plt.plot(dspec[i,:],'.k')
    #pdb.set_trace()
    if 1:#delete this block
        zz = np.loadtxt('/home/tevans/analysis/hst/HD209458/scripts/shift_drake_dspec.txt').T
        print 'this should be the same --> ', np.shape(dspec), np.shape(zz)
        for ii in range(10):
            plt.close('all')
            plt.figure()
            plt.plot(dspec[ii+10,:],'.k')
            plt.plot(zz[ii+10,:],'.r')
            pdb.set_trace()

    return dspec, wavshifts, vstretches


def calc_spectra_variations_BACKUP_SEP2015( spectra, ref_spectrum, max_wavshift=5, dwav=0.01, smoothing_fwhm=None ):
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

    # Interpolate the reference spectrum on to a grid cor
    # increments equal to the dwav shift increment:
    dwavs = np.r_[-max_wavshift:max_wavshift+dwav:dwav]
    nshifts = len( dwavs )
    pad = max_wavshift+1
    x = np.arange( ndisp )
    xi = np.arange( -pad, ndisp+pad )
    z = np.zeros( pad )
    ref_spectrumi = np.concatenate( [ z, ref_spectrum, z ] )
    interpf = scipy.interpolate.interp1d( xi, ref_spectrumi, kind='cubic' )
    shifted = np.zeros( [ nshifts, ndisp ] )
    for i in range( nshifts ):
        shifted[i,:] = interpf( x+dwavs[i] )

    if 0:
        zz = np.loadtxt('/home/tevans/Desktop/shift_drake_spectra.txt')
        print np.shape(zz)
        print np.shape(spectra)
        for i in range( 10 ):
            plt.close('all')
            plt.figure()
            plt.plot(zz[10+i,:],'-c')
            plt.plot(spectra[10+i,:],'--r')
            pdb.set_trace()
    

    # Now loop over the individual spectra and determine which
    # of the shifted reference spectra gives the best match:
    wavshifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )
    dspec = np.zeros( [ nframes, ndisp ] )
    xx=np.loadtxt('/home/tevans/Desktop/delete.txt')
    print np.shape(spectra)
    print np.shape(xx)
    pdb.set_trace()
    print '\nDetermining shifts and stretches:'
    for i in range( nframes ):
        print '... {0} of {1}'.format( i+1, nframes )
        # The current science spectrum:
        if smoothing_fwhm!=None:
            spec_i = scipy.ndimage.filters.gaussian_filter1d( spectra[i,:], smoothing_sig )
        else:
            spec_i = spectra[i,:]
        A = np.reshape( spec_i, [ ndisp, 1 ] )
        rms_i = np.zeros( nshifts )
        vstretches_i = np.zeros( nshifts )
        #if smoothing_sig!=None:
        #    # Smooth with a 1D Gaussian filter:
        #    spec_i = scipy.ndimage.filters.gaussian_filter1d( spec_i, smoothing_sig )
        for j in range( nshifts ):
            b = np.reshape( shifted[j,:], [ ndisp, 1 ] )
            c = np.linalg.lstsq( A, b )
            vstretches_i[j] = c[0]
            rms_i[j] = float( np.sqrt( c[1]/float( ndisp ) ) )
        ix = np.argmin( rms_i )
        wavshifts[i] = dwavs[ix]
        vstretches[i] = vstretches_i[ix]
        dspec[i,:] = ( spec_i*vstretches_i[ix] - shifted[ix,:] )/ref_spectrum
    #plt.figure()
    #for i in range(10):
    #    plt.plot(dspec[i,:],'-')
    #    #plt.plot(dspec[i,:],'.k')
    #pdb.set_trace()

    return dspec, wavshifts, vstretches


def calc_spectra_variations_ORIGINAL( spectra, ref_spectrum, max_wavshift=5, dwav=0.01, smoothing_fwhm=None ):
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

    # Convert smoothing fwhm to the standard deviation of the
    # Gaussian kernel, and smooth the reference spectrum:
    if smoothing_fwhm!=None:
        smoothing_sig = smoothing_fwhm/2./np.sqrt( 2.*np.log( 2. ) )
        ref_spectrum = scipy.ndimage.filters.gaussian_filter1d( ref_spectrum, smoothing_sig )
    else:
        smoothing_sig = None

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
    
    # Convert the pixel shifts to equivalent wavelength increments:
    shifts0 = ( ix0 + np.arange( nshifts ) )*dwav

    # Determine the shifts for each frame:
    wavshifts = np.zeros( nframes )
    for j in range( nframes ):

        print 'Frame {0} of {1}'.format( j+1, nframes )

        # The current science spectrum:
        spec_j = spectra[j,:]
        if smoothing_sig!=None:
            # Smooth with a 1D Gaussian filter:
            spec_j = scipy.ndimage.filters.gaussian_filter1d( spec_j, smoothing_sig )

        # Interpolate on to fine grid of shift increments:
        specf = np.interp( xf, x, spec_j )
        y = np.reshape( specf, [ ninterp, 1 ] )

        # Loop over each trial shift of the reference spectrum:
        rms = np.zeros( nshifts )
        vstretches_j = np.zeros( nshifts )
        for i in range( nshifts ):
            new_spec = np.zeros( ninterp )

            # Determine where the lower edge of the shifted
            # reference spectrum will be located on the 
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
        xmax = min( [ cdcs[i] + ap_radius, ncross ] )
        # Determine the rows that are fully contained
        # within the aperture and integrate along the
        # cross-dispersion axis:
        xmin_full = int( np.ceil( xmin ) )
        xmax_full = int( np.floor( xmax ) )
        ixs_full = ( x>=xmin_full )*( x<=xmax_full )
        spectra[i,:] = np.sum( image[ixs_full,:], axis=cross_axis )
        #plt.figure()
        #plt.imshow(image,interpolation='nearest',aspect='auto')
        #plt.figure()
        #z=np.zeros(np.shape(image))
        #z[ixs_full,:] = 1
        #plt.imshow(z,interpolation='nearest',aspect='auto')
        #pdb.set_trace()
        
        # Determine any rows that are partially contained
        # within the aperture at either end of the scan and
        # add their weighted contributions to the spectrum:
        xlow_partial = xmin_full - xmin
        spectra[i,:] += image[xlow_partial,:]
        xupp_partial = xmax - xmax_full
        spectra[i,:] += image[xupp_partial,:]

    return cdcs, spectra
