import numpy as np
import time
import matplotlib.pyplot as plt
import os, pdb, sys
import scipy.optimize
import scipy.interpolate
import glob
import fitsio
import atpy

"""

See the to-do list in the fit_traces() routine itself. Also:

- I'll use the previous point to extract an
  arbitrary number of spectral bins, by simply
  summing within the spectral aperture and
  then fitting a line between the two regions
  of sky either side (for each spectral bin)
  to interpolate the sky to the spectral
  aperture and then subtract that.

- For the time being, perhaps extract the spectra
  into numpy-friendly txt files, with the first
  column being an integer corresponding to the
  index of whatever bin it is, the second column
  being the pixel coordinate of the spectral bin,
  the third column being time, and the fourth
  column being the spectral bin flux.

- NOTE: The first column of bin indices makes
  sense because it's easy to separate the bins
  later; and putting the pixel coordinates of
  the bins in the second column rather than the
  wavelength means that you can use the files
  later on if you want to redo the wavelength
  calibration; basically, the wavelength
  calibration will come after this step and
  will simply map each pixel in the dispersion
  direction to a wavelength, so the second
  columns in these initial txt files can easily
  be converted to wavelengths in a later step.

"""


def identify_badpixels( stellar ):

    if stellar.star_names==None:
        stellar.star_names = []
        for k in range( stellar.nstars ):
            stellar.star_names += [ 'star{0}'.format( k ) ]
    
    # Read in the science images:
    science_images = np.loadtxt( stellar.science_images_list, dtype=str )
    nimages = len( science_images )
    sciences = []
    badpix_statics = []

    if ( stellar.badpix_static!=None )*( stellar.badpix_static!='' ):
        badpix_path = os.path.join( stellar.adir, stellar.badpix_static )
        badpix_static_hdu = fitsio.FITS( badpix_path )
        badpix_static = badpix_static_hdu[1].read_image()
        badpix_static_hdu.close()
    else:
        badpix_static = None

    # Do niterations passes of bad pixel flagging:
    nslide = 15
    nsigma_thresh = 10
    niterations = 2
    print '\nBad pixel flagging:'
    untainted_frames = np.ones( nimages )
    for i in range( niterations ):
        print ' ... iteration {0} of {1}'.format( i+1, niterations )

        for j in range( nimages ):

            if j%20==0:
                print '... up to image {0} of {1} (iteration {2} of {3})'\
                      .format( j+1, nimages, i+1, niterations )

            # Load current image and measure dimensions:
            image_filename = science_images[j]
            image_root = image_filename[:image_filename.rfind('.')]
            image_filepath = os.path.join( stellar.ddir, image_filename )
            image_hdu_current = fitsio.FITS( image_filepath, 'rw' )
            header0 = image_hdu_current[0].read_header()
            header1 = image_hdu_current[1].read_header()
            current_data = image_hdu_current[1].read_image()
            image_hdu_current.close()

            ixs_before = j - np.arange( nslide ) - 1
            ixs_after = j + np.arange( nslide ) + 1
            ixs_before = ixs_before[ixs_before>=0]
            ixs_after = ixs_after[ixs_after<nimages]
            ixs_before = ixs_before[np.argsort( ixs_before )]
            ixs_after = ixs_after[np.argsort( ixs_after )]
            ixs_slide = np.concatenate( [ ixs_before, [j], ixs_after ] )
            ncontrol = len( ixs_before ) + len( ixs_after )

            # If this is the first frame we need to construct
            # the slider frames to compare against:
            if j==0:
                dims = np.shape( current_data )
                disp_pixrange = np.arange( dims[stellar.disp_axis] )
                crossdisp_pixrange = np.arange( dims[stellar.crossdisp_axis] )
                slider_data = []
                for jj in ixs_slide:
                    image_filename_jj = science_images[jj]
                    image_root_jj = image_filename_jj[:image_filename_jj.rfind('.')]
                    image_filepath_jj = os.path.join( stellar.ddir, image_filename_jj )
                    image_hdu_jj = fitsio.FITS( image_filepath_jj )
                    slider_data += [ image_hdu_jj[1].read_image() ]
                    image_hdu_jj.close()
                slider_data = np.dstack( slider_data )
            else:
                # Drop the first frame of the slider:
                if j>nslide:
                    slider_data = slider_data[:,:,1:]
                # Add a new slide to the leading edge,
                # unless we're close to the end:
                if j<nimages-nslide:
                    image_filename_lead = science_images[ixs_slide[-1]]
                    image_root_lead = image_filename_lead[:image_filename_lead.rfind('.')]
                    image_filepath_lead = os.path.join( stellar.ddir, image_filename_lead )
                    image_hdu_lead = fitsio.FITS( image_filepath_lead )
                    lead_data = image_hdu_lead[1].read_image()
                    image_hdu_lead.close()
                    slider_data = np.dstack( [ slider_data, lead_data ] )
            # Determine which slider frames are untainted:
            untainted = ( untainted_frames[ixs_slide]==1 )*( ixs_slide!=j )
            ixs_use = np.arange( ncontrol+1 )[untainted]

            # Loop over each star on the image:
            badpix_j = np.zeros( dims )
            for k in range( stellar.nstars ):
                dl = stellar.disp_bounds[k][0]
                du = stellar.disp_bounds[k][1]
                cl = stellar.crossdisp_bounds[k][0]
                cu = stellar.crossdisp_bounds[k][1]
                crossdisp_ixs = ( crossdisp_pixrange>=cl )*( crossdisp_pixrange<=cu )
                disp_ixs = ( disp_pixrange>=dl )*( disp_pixrange<=du )
                if stellar.disp_axis==0:
                    subdarray = current_data[dl:du+1,cl:cu+1]
                    subslider = slider_data[dl:du+1,cl:cu+1,:]
                else:
                    subdarray = current_data[cl:cu+1,dl:du+1]
                    subslider = slider_data[cl:cu+1,dl:du+1,:]
                med_sub = np.median( subslider[:,:,ixs_use], axis=2 )
                sig_sub = np.std( subslider[:,:,ixs_use], axis=2 )
                delsigmas_sub = abs( ( subdarray - med_sub )/sig_sub )
                ixs_bad = ( delsigmas_sub>nsigma_thresh )
                frac_bad = ixs_bad.sum()/float( ixs_bad.size )
                if ( frac_bad>1e-3 )*( i<niterations-1 ):
                        untainted_frames[j] = 0
                        print 'Flagging frame {0} as containing >1e-3 bad pixel fraction for {1}'\
                              .format( image_filename, stellar.star_names[k] )
                ## ###
                ## plt.ion()
                ## plt.figure()
                ## plt.title('bad pixels')
                ## plt.imshow(ixs_bad,interpolation='nearest',origin='lower',aspect='auto')
                ## plt.colorbar()
                ## plt.figure()
                ## plt.title('delsigmas')
                ## plt.imshow(delsigmas_sub,interpolation='nearest',origin='lower',aspect='auto')
                ## plt.colorbar()
                ## plt.figure()
                ## plt.title('current')
                ## plt.imshow(subdarray,interpolation='nearest',origin='lower',aspect='auto')
                ## plt.colorbar()
                ## plt.figure()
                ## plt.title('med')
                ## plt.imshow(med_sub,interpolation='nearest',origin='lower',aspect='auto')
                ## plt.colorbar()
                ## plt.figure()
                ## plt.title('sigmas')
                ## plt.imshow(sig_sub,interpolation='nearest',origin='lower',aspect='auto')
                ## plt.colorbar()
                ## pdb.set_trace()
                ## ###

                if stellar.disp_axis==0:
                    badpix_j[dl:du+1,cl:cu+1][ixs_bad] = 1
                else:
                    badpix_j[cl:cu+1,dl:du+1][ixs_bad] = 1
                    
            if i==niterations-1:
                nbad_transient = badpix_j.sum()
                if badpix_static!=None:
                    badpix_j += badpix_static

                # If bad pixels have been flagged more than once,
                # set their value to 1 in the bad pixel map:
                ixs = ( badpix_j!=0 )
                badpix_j[ixs] = 1
                if os.path.isfile( image_filepath ):
                    os.remove( image_filepath )
                image_hdu = fitsio.FITS( image_filepath, 'rw' )
                image_hdu.write( None, header=header0 )
                image_hdu.write( current_data, header=header1 )
                image_hdu.write( badpix_j )
                image_hdu.close()
                if i==niterations-1:
                    nbad_total = badpix_j.sum()
                    if nbad_transient>0:
                        print 'Flagged {0} transient bad pixels in image {1}'\
                              .format( nbad_transient, image_filename )

    return None



def fit_traces( stellar, make_plots=False ):
    """
    Creates files containing the fitted (linear) traces to each of the
    science images. These files contain three columns:
    (1) an integer index specifying the star
    (2) a column containing the dispersion axis pixel coordinates
    (3) a column containing the corresponding cross-dispersion
        pixel coordinates.
    All of these trace files are saved in a specific location (see code
    below) and a file containing a list of all these trace files is
    also created.
    
    
    To-do:
    - perhaps return the fit values for the trace widths at
      each interpolant; but maybe it's better to keep the output
      simple and only print this information to screen, because
      you should already have a good idea of the width of the
      spatial profile from the inspect_images routines with the
      fully integrated spatial profile.
    """

    plt.ioff()
    #plt.ion()
    
    # Read in the science images:
    science_images = np.loadtxt( stellar.science_images_list, dtype=str )
    nimages = len( science_images )

    if stellar.goodbad==None:
        stellar.goodbad = np.ones( nimages )

    # Open files that will store a list of
    # the trace file names:
    science_traces_ofiles = []
    stellar.science_traces_list = []
    for k in range( stellar.nstars ):
        ext = 'science_traces_{0}.lst'.format( stellar.star_names[k] )
        science_trace_ofilepath = os.path.join( stellar.adir, ext )
        science_traces_ofiles += [ open( science_trace_ofilepath, 'w' ) ]
        stellar.science_traces_list += [ ext ]

    # Keep trace of the spectral trace fwhms across all images:
    fwhms = np.zeros( [ nimages, stellar.nstars ] )

    # Loop over the science images, fitting
    # for the spectral traces:
    binw_disp = stellar.tracefit_kwargs['binw_disp']
    for j in range( nimages ):
        t1=time.time()
        # Load current image and measure dimensions:
        if stellar.goodbad[j]==1:
            print '\nFitting traces in image {0} of {1}'.format( j+1, nimages )
        else:
            print '\nImage {0} of {1} flagged as bad - skipping'.format( j+1, nimages )
            continue
        image_filename = science_images[j]
        image_root = image_filename[:image_filename.rfind('.')]
        image_filepath = os.path.join( stellar.ddir, image_filename )
        image_hdu = fitsio.FITS( image_filepath )
        darray = image_hdu[1].read_image()
        badpix = image_hdu[2].read_image()
        darray = np.ma.masked_array( darray, mask=badpix )
        image_hdu.close()

        arr_dims = np.shape( darray )
        eps = 1e-8
        disp_pixrange = np.arange( arr_dims[stellar.disp_axis] + eps )
        crossdisp_pixrange = np.arange( arr_dims[stellar.crossdisp_axis] + eps )

        # These lists will be used to store arrays for
        # plotting the cross-dispersion profiles:
        spectra = [] # raw spectra for each star
        xc = [] # cross-dispersion pixels
        xd = [] # dispersion pixels        
        y = [] # cross-dispersion profiles
        g = [] # gaussian profile fits
        ti = [] # trace knot interpolants along dispersion axis
        ci = [] # cross-dispersion profile center interpolants
        t = [] # interpolated trace fit along dispersion axis
        s = [] # cross-dispersion profile standard deviations

        for k in range( stellar.nstars ):

            # Get the upper and lower edges of the window
            # in the cross-dispersion range that contains
            # the trace:
            cl = stellar.crossdisp_bounds[k][0]
            cu = stellar.crossdisp_bounds[k][1]
            dl = stellar.disp_bounds[k][0]
            du = stellar.disp_bounds[k][1]
            
            # Identify the rectangle containing the
            # current spectrum:
            crossdisp_ixs = ( crossdisp_pixrange>=cl )*( crossdisp_pixrange<=cu )
            crossdisp_pixs = crossdisp_pixrange[crossdisp_ixs]
            disp_ixs = ( disp_pixrange>=dl )*( disp_pixrange<=du )
            disp_pixs = disp_pixrange[disp_ixs]
            npix_disp = len( disp_pixs )
            xc += [ crossdisp_pixs ]
            xd += [ disp_pixs ]

            # Generate bins along the dispersion axis:
            ledges = np.arange( disp_pixs[0], disp_pixs[-1], binw_disp )
            bincents_disp = ledges + 0.5*binw_disp
            ixs = ( bincents_disp>disp_pixs.min() )*( bincents_disp<disp_pixs.max() )
            bincents_disp = bincents_disp[ixs]
            nbins_disp = len( bincents_disp )

            # Extract the raw stellar spectrum:
            if stellar.disp_axis==0:
                spectrum = np.mean( darray[disp_ixs,:][:,crossdisp_ixs], axis=1 )
            else:
                spectrum = np.mean( darray[:,disp_ixs][crossdisp_ixs,:], axis=0 )
            spectrum_amp = spectrum.max() - spectrum.min()
            spectra += [ spectrum ]

            # Fit Gaussians to each cross-dispersion
            # profile for the current star:
            y_k = []
            g_k = []
            s_k = []
            bincents_disp_good = []
            trace_interps = [] #np.zeros( nbins_disp )
            for i in range( nbins_disp ):

                # Extract the cross-dispersion profile form 
                # the current bin along the dispersion axis:
                dl_i = ledges[i]
                du_i = dl_i + binw_disp
                if stellar.disp_axis==0:
                    binwindow = darray[dl_i:du_i,cl:cu+1]
                else:
                    binwindow = darray[cl:cu+1,dl_i:du_i]

                nbad = binwindow.mask.sum()
                ntot = binwindow.mask.size
                frac_bad = float( nbad )/ntot
                if frac_bad>0.2:
                    print 'Too many bad pixels for bin - skipping'
                    continue
                else:

                    if stellar.disp_axis==0:
                        sky_rough_l = np.median( darray[dl_i:du_i,cl:cl+2] )
                        sky_rough_u = np.median( darray[dl_i:du_i,cu-2:cu] )
                        sky_slope = ( sky_rough_u - sky_rough_l )/( cu-cl+1 )
                        sky_rough = sky_slope*np.arange( cu-cl+1 ) + sky_rough_l
                        raw_flux = np.median( darray[dl_i:du_i,cl:cu+1], axis=0 )
                        crossdisp_prof = raw_flux - sky_rough
                    else:
                        sky_rough_l = np.median( darray[cl:cl+2,dl_i:du_i].flatten() )
                        sky_rough_u = np.median( darray[cu-2:cu,dl_i:du_i].flatten() )
                        sky_slope = ( sky_rough_u - sky_rough_l )/( cu-cl+1 )
                        sky_rough = sky_slope*np.arange( cu-cl+1 ) + sky_rough_l
                        raw_flux = np.median( darray[cl:cu+1,dl_i:du_i], axis=1 )
                        crossdisp_prof = raw_flux - sky_rough
                # Fit the cross-dispersion profile with a
                # Gaussian using least squares:
                A0 = crossdisp_prof.min()
                B0 = 0.
                C0 = np.max( crossdisp_prof ) - A0
                crossdisp_prof_downshift = crossdisp_prof - A0 - 0.8*C0
                ixs = ( crossdisp_prof_downshift.data>0 )
                if ixs.sum()>2:
                    pixs = crossdisp_pixs[ixs]
                    sig0 = 0.3*( pixs.max() - pixs.min() )
                else:
                    sig0 = 0.1*len( crossdisp_pixs )
                ix = np.argmax( crossdisp_prof )
                crossdisp_coord0 = crossdisp_pixs[ix]
                pars0 = np.array( [ A0, B0, C0, sig0, crossdisp_coord0 ] )
                pars_optimised = scipy.optimize.leastsq( gauss_resids, \
                                                         pars0, \
                                                         args=( crossdisp_pixs, \
                                                                crossdisp_prof ) )[0]
                A, B, C, sig, trace_interps_i = pars_optimised
                bincents_disp_good += [ bincents_disp[i] ]
                trace_interps += [ trace_interps_i ]

                y_k += [ crossdisp_prof ]
                g_k += [ gauss_profile( crossdisp_pixs, pars_optimised ) ]
                s_k += [ abs( sig ) ]

            ninterp = len( trace_interps )
            if ninterp<2:
                print 'Could not extract trace - skipping frame'
                continue
                
            fwhms[j,k] = 2.355*np.median( s_k )
            
            # Now that we've fit for the centres of each bin along the
            # dispersion axis, we can interpolate these to a spectral
            # trace evaluated at each pixel along the dispersion axis:
            bincents_disp_good = np.array( bincents_disp_good )
            trace_interps = np.array( trace_interps )
            trace = np.zeros( npix_disp )

            if stellar.tracefit_kwargs['method']=='linear_interpolation':

                # Create the interpolating function:
                interpf = scipy.interpolate.interp1d( bincents_disp_good, \
                                                      trace_interps, \
                                                      kind='linear' )

                # Interpolate between the bin centers:
                ixs = ( ( disp_pixs>=bincents_disp_good.min() )\
                        *( disp_pixs<=bincents_disp_good.max() ) )
                trace[ixs] = interpf( disp_pixs[ixs] )

                # Linearly extrapolate at the edges:
                # NOTE: I loop over each element in the series here to avoid a
                # bizarre bug with numpy. In theory, it should be possible to
                # do this bit simply using: 
                #ixsl = disp_pixs<bincents_disp_good.min()
                #ixsu = disp_pixs>bincents_disp_good.max()
                # I have absolutely no idea why numpy seems to be making such a
                # trivial error, but hopefully it gets fixed in the future. It's
                # a pretty big concern, actually, because it could introduce subtle
                # problems to other parts of code. Anyway, the problem seems to
                # only occur for the macports installed version:
                #       py27-numpy @1.8.0_2+atlas+gcc45
                # whereas I don't get the same odd behaviour for v1.7 for instance.
                # So in the future it should be possible to revert the following
                # lines to the simpler version above, presumably.
                ixsl = []
                for w in range( len( disp_pixs ) ):
                    if disp_pixs[w]<bincents_disp_good.min():
                        ixsl += [ w ]
                ixsl = np.array( ixsl )
                ixsu = []
                for w in range( len( disp_pixs ) ):
                    if disp_pixs[w]>bincents_disp_good.max():
                        ixsu += [ w ]
                ixsu = np.array( ixsu )
                
                trace[ixsl] = linear_extrapolation( disp_pixs[ixsl], \
                                                    bincents_disp_good[0:2], \
                                                    trace_interps[0:2] )
                trace[ixsu] = linear_extrapolation( disp_pixs[ixsu], \
                                                    bincents_disp_good[-2:], \
                                                    trace_interps[-2:] )
            else:
                pdb.set_trace() # haven't implemented any other methods yet
            y += [ y_k ]
            g += [ g_k ]
            s += [ s_k ]
            ci += [ bincents_disp_good ]
            ti += [ trace_interps ]
            t += [ trace ]

            if 0*(j>50):
                plt.figure()
                plt.subplot(211)
                plt.plot(crossdisp_pixs,crossdisp_prof)
                plt.axvline(crossdisp_coord0,ls='-')
                plt.axvline(crossdisp_coord0-sig0,ls='--')
                plt.axvline(crossdisp_coord0+sig0,ls='--')
                plt.subplot(212)                    
                plt.plot(crossdisp_pixs,crossdisp_prof)
                plt.axvline(trace_interps[i],ls='-')
                plt.axvline(trace_interps[i]-sig,ls='--')
                plt.axvline(trace_interps[i]+sig,ls='--')
                plt.suptitle( 'Image {0}, star {1}, bin {2}'.format( j, k, i ))
                pdb.set_trace()
                plt.close('all')



            # Save the trace centers for the current image to an output file:
            ofolder_ext = 'trace_files/{0}'.format( stellar.star_names[k] )
            otracename = '{0}_nbins{1}.npy'.format( image_root, nbins_disp )
            ofile_ext = os.path.join( ofolder_ext, otracename )
            otracepath = os.path.join( stellar.adir, ofile_ext )
            if os.path.isdir( os.path.dirname( otracepath ) )!=True:
                os.makedirs( os.path.dirname( otracepath ) )
            otrace = open( otracepath, 'w' )
            otrace.write( '# Trace fitting method = {0:s}\n'\
                          .format( stellar.tracefit_kwargs['method'] ) )
            otrace.write( '# Dispersion, Cross-dispersion' )
            for i in range( npix_disp ):
                ostr = ''
                ostr += '\n{0:.0f} {1:.2f}' .format( disp_pixs[i], \
                                                         trace[i] )
                otrace.write( ostr )
            otrace.close()
            print ' ... saved trace fit {0}/{1:s}'.format( stellar.star_names[k], otracename )
            science_traces_ofiles[k].write( '{0}\n'.format( ofile_ext ) )

        if make_plots==True:

            tracedir = os.path.join( stellar.adir, 'trace_pngs' )
            if os.path.isdir( tracedir )!=True:
                os.makedirs( tracedir )

            fig = plt.figure( figsize = [ 15, 11 ] )
            fig.suptitle( image_filename, fontsize=16 )
            
            buff = 0.05
            nrows = stellar.nstars
            axh = ( 1. - nrows*buff )/float( nrows )
            axw = ( 1. - 4*buff )/3.
            xlow1 = 1.5*buff
            xlow2 = xlow1 + axw + buff
            xlow3 = xlow2 + axw + buff
            for k in range( stellar.nstars ):
                row_number = k%nrows + 1
                ylow = 1. - buff - row_number*( axh + 0.5*buff )
                if k==0:
                    axprof = fig.add_axes( [ xlow1, ylow, axw, axh ] )
                    axtr = fig.add_axes( [ xlow2, ylow, axw, axh ] )
                    axspec = fig.add_axes( [ xlow3, ylow, axw, axh ] )
                    axspec0 = axspec
                    axprof0 = axprof
                    axtr0 = axtr
                    axprof.set_title( 'cross-disp profile' )
                    axtr.set_title( 'trace fit' )
                    axspec.set_title( 'raw spectrum' )
                    ymax = np.concatenate( y[k] ).max()
                    specmax = spectra[k].max()
                else:
                    axprof = fig.add_axes( [ xlow1, ylow, axw, axh ], sharey=axprof0 )
                    axtr = fig.add_axes( [ xlow2, ylow, axw, axh ], sharey=axtr0 )
                    axspec = fig.add_axes( [ xlow3, ylow, axw, axh ], sharey=axspec0 )
                axprof.set_ylabel( '{0}'.format( stellar.star_names[k] ) )
                if k==stellar.nstars-1:
                    axprof.set_xlabel( 'cross-disp pixel coord' )
                    axtr.set_xlabel( 'disp pixel coord' )
                    axspec.set_xlabel( 'disp pixel coord' )
                # Plot the raw stellar spectrum:
                axspec.plot( xd[k], spectra[k], '-k', lw=2 )
                # Plot the cross-dispersion profile:
                for i in range( len( y[k] ) ):
                    axprof.plot( xc[k], y[k][i], '-k', lw=2 )
                    axprof.plot( xc[k], g[k][i], '-g', lw=1 )
                xlow = max( [ np.median( np.array( ti[k] ) - 15*np.array( s[k] ) ), \
                              stellar.crossdisp_bounds[k][0] ] )
                xupp = min( [ np.median( np.array( ti[k] ) + 15*np.array( s[k] ) ), \
                              stellar.crossdisp_bounds[k][1] ] )
                axprof.set_xlim( [ xlow, xupp ] )
                axprof.text( 0.05, 0.85, 'profile stdv = {0:.2f} pix'.format( np.median( s[k] ) ), \
                             fontsize=8, horizontalalignment='left', transform=axprof.transAxes )
                # Plot the trace fit:
                axtr.plot( xd[k], t[k] - np.median( t[k] ), '-r', lw=2 )
                axtr.plot( ci[k], ti[k] - np.median( t[k] ), 'o', mec='k', mfc='k', ms=7 )
                axtr.fill_between( xd[k], \
                                   t[k] - np.median( t[k] ) - np.median( s[k] ), \
                                   t[k] - np.median( t[k] ) + np.median( s[k] ), \
                                   color=[0.8,0.8,0.8] )
            axprof0.set_ylim( [ 0, 1.1*ymax ] )
            axtr0.set_ylim( [ -10, +10 ] )
            axspec0.set_ylim( [ 0, 1.1*specmax ] )
            ofigname = 'traces_{0}.png'.format( image_root )
            ofigpath = os.path.join( tracedir, ofigname )
            plt.savefig( ofigpath )
            plt.close()
            print ' ... saved figure {0}'.format( ofigname )
        fwhms_str = ''
        for k in range( stellar.nstars ):
            fwhms_str += ' {0:.2f},'.format( fwhms[j,k] )
        print 'PSF FWHMs (pixels) -', fwhms_str[:-1]
        t2=time.time()
        #print t2-t1

    # Summarise the PSF FWHM info:
    med = np.median( fwhms, axis=0 ) # median PSF FWHM for each star
    std = np.std( fwhms, axis=0 ) # PSF FWHM scatter for each star
    print '\nPSF FWHMs across all images in units of pixels'
    print '# median, scatter'
    plt.figure()
    for k in range( stellar.nstars ):
        print '{0} = {1:.3f}, {2:.3f}'.format( stellar.star_names[k], med[k], std[k] )
        plt.plot( fwhms[:,k], '-', label='{0}'.format( stellar.star_names[k] ) )
    plt.ylabel( 'PSF FWHM (pixels)' )
    plt.xlabel( 'Image number' )
    plt.legend()
    ofigname = os.path.join( stellar.adir, 'psf_fwhms.png' )
    plt.savefig( ofigname )
    plt.close()
    ofilename = os.path.join( stellar.adir, 'psf_fwhms.npy' )
    np.savetxt( ofilename, fwhms )
    m = np.median( fwhms, axis=1 ) # median PSF FWHM for each image
    ix = np.argmax( m ) # image number with widest PSF
    print '\nThe frame with the largest median PSF is:'
    print science_images[ix]
    print 'with PSF FWHM of {0:.3f} pixels'.format( m[ix] )
    z = np.std( fwhms, axis=1 ) # spread amongst stars per image
    
    print '\nSaved list of traces for each star in:'
    for k in range( stellar.nstars ):
        science_traces_ofiles[k].close()
        ofilename = os.path.basename( stellar.science_traces_list[k] )
        print '{0} --> {1}'.format( stellar.star_names[k], ofilename )
    print '\nSaved PSF FWHMs for each star in each frame in:\n{0}'\
          .format( ofilename )
    print 'Saved corresponding figure:\n{0}'.format( ofigname )
    plt.ion()

    return None



def extract_spectra( stellar ):

    # Read in the bad pixel map if one has been provided:
    if np.any( stellar.badpix_static!=np.array( [ None, '' ] ) )!=True:
        badpix_static_hdu = fitsio.FITS( stellar.badpix_static )
        badpix_static = badpix_static_hdu[1].read_image()
    else:
        badpix_static = None

    # Read in the trace filenames for each star on each
    # image and open files to which lists of the extracted
    # spectra will be saved to:
    trace_files = []
    science_spectra_ofiles = []
    stellar.science_spectra_lists = []
    for k in range( stellar.nstars ):
        trace_list = os.path.join( stellar.adir, stellar.science_traces_list[k] )
        trace_files += [ np.loadtxt( trace_list, dtype=str ) ]
        ext = 'science_spectra_{0}.lst'.format( stellar.star_names[k] )
        science_spectrum_ofilepath = os.path.join( stellar.adir, ext )
        science_spectra_ofiles += [ open( science_spectrum_ofilepath, 'w' ) ]
        stellar.science_spectra_lists += [ ext ]

    # Read in the list of science images:
    science_images_list = os.path.join( stellar.adir, stellar.science_images_list )
    science_images = np.loadtxt( science_images_list, dtype=str )
    nimages = len( science_images )
    if stellar.goodbad==None:
        stellar.goodbad = np.ones( nimages )

    if stellar.gains==None:
        stellar.gains = np.ones( nimages )
    if stellar.jds==None:
        stellar.jds = np.arange( nimages )
    if stellar.exptime_secs==None:
        stellar.exptime_secs = np.ones( nimages )
    
    # Loop over each image, and extract the spectrum
    # for each star on each image:
    eps = 1e-10
    for j in range( nimages ):

        # Load in the image and header:
        if stellar.goodbad[j]==1:
            print 'Extracting spectra from image {0} of {1}'.format( j+1, nimages )
        else:
            print '\nImage {0} of {1} flagged as bad - skipping\n'.format( j+1, nimages )
            stellar.goodbad[j] = 0
            continue
        image_filename = science_images[j]
        image_root = image_filename[:image_filename.rfind('.')]
        image_filepath = os.path.join( stellar.ddir, image_filename )
        image_hdu = fitsio.FITS( image_filepath )
        header = image_hdu[0].read_header()
        darray = image_hdu[1].read_image()
        arr_dims = np.shape( darray )
        try:
            badpix_transient = image_hdu[2].read_image()
            if badpix_static!=None:
                badpix = badpix_transient + badpix_static
            else:
                badpix = badpix_transient
        except:
            if badpix_static!=None:
                badpix = badpix_static
        if badpix!=None:
            darray = np.ma.masked_array( darray, mask=badpix )            

        disp_pixrange = np.arange( arr_dims[stellar.disp_axis] )
        crossdisp_pixrange = np.arange( arr_dims[stellar.crossdisp_axis] )        
        image_hdu.close()
        jdobs = stellar.jds[j] + 0.5*stellar.exptime_secs[j]/60./60./24.

        subarrays = []
        crossdisp_ixs = []
        disp_ixs = []
        apflux = []
        nappixs = []
        skyppix = []
        for k in range( stellar.nstars ):
            # Identify the dispersion and cross-dispersion
            # pixels of the spectrum data:
            cl = stellar.crossdisp_bounds[k][0]
            cu = stellar.crossdisp_bounds[k][1]
            crossdisp_ixs_k = ( crossdisp_pixrange>=cl )*( crossdisp_pixrange<=cu )
            dl = stellar.disp_bounds[k][0]
            du = stellar.disp_bounds[k][1]
            disp_ixs_k = ( disp_pixrange>=dl )*( disp_pixrange<=du )
            # Cut out a subarray containing the spectrum data:
            if stellar.disp_axis==0:
                subarray_k = darray[disp_ixs_k,:][:,crossdisp_ixs_k]
            else:
                subarray_k = darray[:,disp_ixs_k][crossdisp_ixs_k,:]
            nbad = subarray_k.mask.sum()
            ntot = subarray_k.mask.size
            frac_bad = float( nbad )/ntot
            if frac_bad>0.2:
                print '\nToo many bad pixels in:\n{0}\n(skipping)\n'\
                      .format( image_filepath )
                stellar.goodbad[j] = 0
                break
            else:
                subarrays += [ subarray_k ]
                crossdisp_ixs += [ crossdisp_ixs_k ]
                disp_ixs += [ disp_ixs_k ]            

            crossdisp_pixs = crossdisp_pixrange[crossdisp_ixs[k]]
            disp_pixs = disp_pixrange[disp_ixs[k]]
            npix_disp = len( disp_pixs )
            
            # Read in the array containing the trace fit:
            trace_file_path_kj = os.path.join( stellar.adir, trace_files[k][j] )
            trarray = np.loadtxt( trace_file_path_kj )

            apflux_k = np.zeros( npix_disp )
            nappixs_k = np.zeros( npix_disp )
            skyppix_k = np.zeros( npix_disp )

            # Loop over each pixel column along the
            # dispersion axis:
            for i in range( npix_disp ):
                if stellar.disp_axis==0:
                    crossdisp_central_pix = trarray[i,1]
                    crossdisp_row = subarrays[k][i,:]
                else:
                    crossdisp_central_pix = trarray[i,1]
                    crossdisp_row = subarrays[k][:,i]

                # Before proceeding, make sure the fitted trace
                # center is actually located within the defined
                # cross-dispersion limits; if it's not, it suggests
                # that something went wrong and this is probably
                # an unusable frame, so skip it:
                if ( crossdisp_central_pix<crossdisp_pixs.min() )+\
                   ( crossdisp_central_pix>crossdisp_pixs.max() ):
                    print '\nProblem with trace fitting in {0}'\
                          .format( trace_file_path_kj )
                    print '(may be worth inspecting the corresponding image)\n'
                    stellar.goodbad[j] = 0
                    break

                # Determine the pixels that are fully
                # contained within the spectral aperture:
                l = max( [ crossdisp_central_pix - stellar.spectral_ap_radius, crossdisp_pixs.min() ] )
                u = min( [ crossdisp_central_pix + stellar.spectral_ap_radius, crossdisp_pixs.max() ] )
                l_full = np.ceil( l )
                u_full = np.floor( u ) - 1
                ixs_full = ( ( crossdisp_pixs>=l_full )\
                             *( crossdisp_pixs<=u_full ) )
                npix_full = len( crossdisp_pixs[ixs_full] )

                # Deal with any pixels that are only partially
                # contained within the spectral aperture:
                u_frac = int( np.floor( u ) )
                l_frac = int( np.floor( l ) )
                ix_frac_u = ( crossdisp_pixs==u_frac )
                ix_frac_l = ( crossdisp_pixs==l_frac )
                nfracpix_u = ( crossdisp_central_pix + stellar.spectral_ap_radius ) \
                             - ( u_full + 1 )
                nfracpix_l = l_full - \
                             ( crossdisp_central_pix - stellar.spectral_ap_radius )

                # Simple sanity check:
                # NOTE: The eps=1e-10 offsets are to account for the basic numpy bug
                # that seems to have crept in with the macports v1.8 installation...
                if ( npix_full + nfracpix_l + nfracpix_u )>2*stellar.spectral_ap_radius+eps:
                    pdb.set_trace() # this shouldn't happen
                elif ( npix_full + nfracpix_l + nfracpix_u )<2*stellar.spectral_ap_radius-eps:
                    print ' - aperture overflowing cross-dispersion edge for {0} in {1}'\
                          .format( stellar.star_names[k], image_filename )

                # Define variables containing the contributions of the
                # full and partial pixels within the spectral aperture:
                darray_full = crossdisp_row[ixs_full]
                if crossdisp_row.mask[ix_frac_l]==False:
                    darray_fraclower = float( crossdisp_row[ix_frac_l] )*nfracpix_l
                else:
                    darray_fraclower = 0
                if crossdisp_row.mask[ix_frac_u]==False:
                    darray_fracupper = float( crossdisp_row[ix_frac_u] )*nfracpix_u
                else:
                    darray_fracupper = 0

                # Now identify pixels above trace for sky:
                usky_l = int( np.ceil( crossdisp_central_pix ) + stellar.sky_inner_radius )
                usky_u = int( usky_l + stellar.sky_band_width )
                uixs = ( ( crossdisp_pixs>=usky_l )*( crossdisp_pixs<=usky_u ) )
                upix = np.mean( crossdisp_pixs[uixs] )

                # Repeat for sky below trace:
                lsky_u = int( np.floor( crossdisp_central_pix ) - stellar.sky_inner_radius )
                lsky_l = int( lsky_u - stellar.sky_band_width )
                lixs = ( ( crossdisp_pixs>=lsky_l )*( crossdisp_pixs<=lsky_u ) )
                lpix = np.mean( crossdisp_pixs[lixs] )

                # Account for the possibility that the sky
                # is outside the cross-dispersion range:
                lsky_l = max( [ lsky_l, crossdisp_pixs.min() ] )
                lsky_u = max( [ lsky_u, crossdisp_pixs.min() ] )
                usky_l = min( [ usky_l, crossdisp_pixs.max() ] )
                usky_u = min( [ usky_u, crossdisp_pixs.max() ] )
                if lsky_u>l:
                    lsky_l = None
                    lsky_u = None
                if usky_l<u:
                    lsky_l = None
                    lsky_u = None
                if lsky_l==lsky_u:
                    lsky_l = None
                    lsky_u = None
                if usky_l==usky_u:
                    usky_l = None
                    usky_u = None

                # Extract estimates for the sky values:
                uskypatch = crossdisp_row[uixs]
                lskypatch = crossdisp_row[lixs]

                ixsu = ( uskypatch > 0 )
                usky_med = np.median( uskypatch[ixsu] )
                ixsl = ( lskypatch > 0 )
                lsky_med = np.median( lskypatch[ixsl] )

                # Interpolate sky through the spectral aperture:
                if ( lsky_l==None )*( lsky_u==None )*( usky_l==None )*( usky_u==None ):
                    pdb.set_trace() # no valid sky regions; this shouldn't happen
                elif ( lsky_l==None )*( lsky_u==None ):
                    sky = usky_med*( nfracpix_l + nfracpix_u + npix_full )
                elif ( usky_l==None )*( usky_u==None ):
                    sky = lsky_med*( nfracpix_l + nfracpix_u + npix_full )
                else:
                    skyfunc = scipy.interpolate.interp1d( [ lpix, upix ], [ lsky_med, usky_med ], kind='linear' )
                    fullap_pixs = crossdisp_pixs[ ixs_full ]
                    sky_full = np.sum( skyfunc( fullap_pixs ) )
                    sky_lpartial = skyfunc( fullap_pixs[0]-1 )*nfracpix_l
                    sky_upartial = skyfunc( fullap_pixs[-1]+1 )*nfracpix_u
                    sky = sky_full + sky_lpartial + sky_upartial

                # Sum of the pixels that are fully and partially 
                # contained within the spectral aperture, and
                # subtract the sky contribution:
                apflux_k[i] = np.sum( darray_full ) \
                              + darray_fraclower \
                              + darray_fracupper \
                              - sky
                nappixs_k[i] = npix_full + nfracpix_u + nfracpix_l
                skyppix_k[i] = sky/float( nappixs_k[i] )
            apflux += [ apflux_k ]
            nappixs += [ nappixs_k ]
            skyppix += [ skyppix_k ]
            
        # If there was a problem with one of the trace fits
        # for one or more of the stars, skip the current frame:
        if stellar.goodbad[j]==0:
            continue
        else:

            # Save the spectra for the current star on 
            # the current image in a fits table:
            data = np.zeros( npix_disp, dtype=[ ( 'disp_pixs', np.int64 ), \
                                                ( 'apflux', np.float64 ), \
                                                ( 'nappixs', np.float64 ), \
                                                ( 'skyppix', np.float64 ) ] )
            for k in range( stellar.nstars ):

                # Define filename for the output spectrum of
                # the current star for the current image:
                ospec_root = 'spec1d_{0}'.format( image_root )
                ospec_name = '{0}.fits'.format( ospec_root )
                ospec_ext = 'spectra/{0}'.format( stellar.star_names[k] )
                ospec_ext = os.path.join( ospec_ext, ospec_name )
                ospec_filepath = os.path.join( stellar.adir, ospec_ext )
                ofolder = os.path.dirname( ospec_filepath )
                if os.path.isdir( ofolder )!=True:
                    os.makedirs( ofolder )
                
                data['disp_pixs'] = disp_pixs[k]
                data['apflux'] = apflux[k]*stellar.gains[j]
                data['nappixs'] = nappixs[k]
                data['skyppix'] = skyppix[k]*stellar.gains[j]
                if os.path.isfile( ospec_filepath ):
                    os.remove( ospec_filepath )
                fits = fitsio.FITS( ospec_filepath, 'rw' )
                trace_filename = os.path.basename( trace_files[k][j] )
                header = { 'IMAGE':image_filename, 'TRACE':trace_filename, 'JD-OBS':jdobs }
                fits.write( data, header=header )
                fits.close()
                print ' ... saved {0}'.format( ospec_filepath )
                science_spectra_ofiles[k].write( '{0}\n'.format( ospec_ext ) )
                
    # Save all the lists of spectra files:
    for k in range( stellar.nstars ):
        science_spectra_ofiles[k].close()
        print '\nSaved list of spectra in:\n{0}'.format( stellar.science_spectra_lists[k] )

    return None

    
def calibrate_wavelength_scale( stellar, poly_order=1, make_plots=False ):
    """
    Use the pre-determined dispersion pixel coordinates of the fiducial
    lines from the master arc frame to fit a polynomial mapping from the
    dispersion axis to the wavelength scale for each star.
    """

    plt.ioff()
    
    marc_hdu = fitsio.FITS( stellar.wcal_kws['marc_file'], 'r' )
    marc = marc_hdu[1].read_image()
    marc_hdu.close()
    arr_dims = np.shape( marc )
    if stellar.disp_axis==0:
        npix_disp, npix_crossdisp = arr_dims
    else:
        npix_crossdisp, npix_disp = arr_dims
    disp_pixrange = np.arange( 0, npix_disp, 1 )
    fiducial_lines = stellar.wcal_kws['fiducial_lines']
    nlines = len( fiducial_lines )
    stellar.wavsol = {}
    stellar.wavsol['poly_order'] = poly_order
    stellar.wavsol['disp_pixs_input'] = [] #pixel_coords = []
    stellar.wavsol['wav_fits_output'] = []
    stellar.wavsol['wavsol_coeffs'] = []
    print '\n\nFitting cross-dispersion line profiles for each star'
    print 'using a polynomial of order {0}:'.format( poly_order )
    for k in range( stellar.nstars ):
        crossdisp_pixbounds = stellar.wcal_kws['crossdisp_pixbounds'][k]
        disp_pixs_input_k = []
        print '\n{0}...'.format( stellar.star_names[k] )
        for j in range( nlines ):
            disp_pixbounds = stellar.wcal_kws['disp_pixbounds'][k][j]
            dl = disp_pixbounds[0]
            du = disp_pixbounds[1]
            disp_pixs_kj = np.arange( dl, du+1, 1 )
            cl = crossdisp_pixbounds[0]
            cu = crossdisp_pixbounds[1]
            if stellar.disp_axis==0:
                disp_prof = np.sum( marc[:,cl:cu+1][dl:du+1,:], axis=1 )
            else:
                disp_prof = np.sum( marc[cl:cu+1,:][:,dl:du+1], axis=0 )            
            # Ensure the dispersion profile is in reasonable units:
            disp_prof -= np.median( disp_prof )
            disp_prof /= disp_prof.max()
            
            A0 = disp_prof.min()
            B0 = 0.
            C0 = disp_prof.max() - disp_prof.min()
            ix = np.argmax( disp_prof )
            mu0 = disp_pixs_kj[ix]
            disp_prof_shifted = disp_prof - A0 - 0.7*C0
            ixs = ( disp_prof_shifted>0 )
            sig0 = disp_pixs_kj[ixs][-1] - disp_pixs_kj[ixs][0]
            pars0 = np.array( [ A0, B0, C0, sig0, mu0 ] )
            pars_optimised = scipy.optimize.leastsq( gauss_resids, pars0, \
                                                     args=( disp_pixs_kj, \
                                                            disp_prof ) )[0]
            A, B, C, sig, mu = pars_optimised
            disp_pixs_input_k += [ mu ]

        # Compute the linear mapping from pixel
        # coordinates to wavelength scale:
        wav_fiducial = np.array( fiducial_lines )
        pix_measured = np.array( disp_pixs_input_k )
        offset = np.ones( len( pix_measured ) )
        pix_poly_terms = []
        for i in range( poly_order ):
            pix_poly_terms += [ pix_measured**( i+1 ) ]
        pix_poly_terms = np.column_stack( pix_poly_terms )
        basis = np.column_stack( [ offset, pix_poly_terms ] )
        wavsol_coeffs_k = np.linalg.lstsq( basis, wav_fiducial )[0]
        wav_fits_output_k = np.dot( basis, wavsol_coeffs_k )
        residuals = wav_fiducial-wav_fits_output_k
        print '  WAV --> ', wav_fiducial
        print '  PIX --> ', pix_measured
        print '  Max wavelength discrepancy = {0}'.format( np.abs( residuals ).max() )
        stellar.wavsol['disp_pixs_input'] += [ np.array( disp_pixs_input_k ) ]
        stellar.wavsol['wav_fits_output'] += [ np.array( wav_fits_output_k ) ]
        stellar.wavsol['wavsol_coeffs'] += [ np.array( wavsol_coeffs_k ) ]
        
    # Now go through all the spectra that have already been generated
    # and add a column to the fits table containing the wavelengths:
    print '\nSuccessfully calibrated the wavelength scale.'
    if stellar.science_spectra_lists==None:
        print '\nNo science_spectra_lists attribute provided -'
        print 'spectra not updated'
    else:
        print 'Adding wavelength solutions to the existing spectrum files...'
        science_images_list = os.path.join( stellar.adir, stellar.science_images_list )
        science_image_files = np.loadtxt( science_images_list, dtype=str )

        spectra_files = []
        nspectra = []
        for k in range( stellar.nstars ):
            print ' ... {0}'.format( stellar.star_names[k] )
            science_spectra_list_filepath_k = os.path.join( stellar.adir, \
                                                            stellar.science_spectra_lists[k] )
            science_spectra_list_k = np.loadtxt( science_spectra_list_filepath_k, dtype=str )
            spectra_files += [ science_spectra_list_k ]
            nspectra_k = len( science_spectra_list_k )
            nspectra += [ nspectra_k ]
            for j in range( nspectra_k ):
                spectra_file_kj = os.path.join( stellar.adir, spectra_files[k][j] )
                spectrum_hdu = fitsio.FITS( spectra_file_kj, 'rw' )
                disp_pixs = spectrum_hdu[1]['disp_pixs'].read()
                offset = np.ones( len( disp_pixs ) )
                pix_poly_terms = []
                for i in range( poly_order ):
                    pix_poly_terms += [ disp_pixs**( i+1 ) ]
                pix_poly_terms = np.column_stack( pix_poly_terms )
                basis = np.column_stack( [ offset, pix_poly_terms ] )
                wav = np.dot( basis, stellar.wavsol['wavsol_coeffs'][k] )
                try:
                    spectrum_hdu[1].insert_column( 'wav', wav )
                except:
                    spectrum_hdu[1].write_column( 'wav', wav )
                spectrum_hdu.close()
        if make_plots==True:
            ofolder_ext = 'spectra_pngs'
            ofolder = os.path.join( stellar.adir, ofolder_ext )
            if os.path.isdir( ofolder )!=True:
                os.makedirs( ofolder )
            print 'Making plots and saving as png files...'
            nspectra0 = nspectra[0]
            for k in range( 1, stellar.nstars ):
                if nspectra[k]!=nspectra0:
                    print '\nWARNING: Number of spectra not the same for each star!'
                    print '(proceeding with plotting anyway)\n'
            for j in range( nspectra0 ):
                print ' ... spectra {0} of {1}'.format( j+1, nspectra[0] )
                plt.figure()
                for k in range( stellar.nstars ):
                    spectrum_file_kj = os.path.join( stellar.adir, spectra_files[k][j] )
                    spectrum_hdu = fitsio.FITS( spectrum_file_kj, 'r' )
                    wav = spectrum_hdu[1].read_column( 'wav' )
                    spectrum = spectrum_hdu[1].read_column( 'apflux' )
                    spectrum_hdu.close()
                    plt.plot( wav, spectrum, '-', lw=2, zorder=k, label='{0}'\
                              .format( stellar.star_names[k] ) )
                plt.xlabel( 'Wavelength' )
                plt.ylabel( 'Mean pixel counts (ADU)' )
                plt.legend()
                image_filename = science_image_files[j]
                image_root = image_filename[:image_filename.rfind('.')]
                ofigname = '{0}.png'.format( image_root )
                ofigpath = os.path.join( ofolder, ofigname )
                plt.savefig( ofigpath )
                plt.close()
        print 'Done.'
    plt.ion()
    
    return None

    
def gauss_resids( pars, x, y ):
    """
    Gaussian plus linear trend for the
    scipy.optimize.lstsq routine.
    """
    
    A = pars[0]
    B = pars[1]
    C = pars[2]
    sig = pars[3]
    mu = pars[4]
    m = gauss_profile( x, pars )
    r = y - m

    return r

    
def gauss_profile( x, pars ):
    """
    Gaussian plus linear trend.
    """
    A = pars[0]
    B = pars[1]
    C = pars[2]
    sig = pars[3]
    mu = pars[4]
    m = A + B*x + C*np.exp( -0.5*( ( ( x-mu )/sig )**2. ) )

    return m


def linear_extrapolation( xextrap, xs, ys):
    x1 = xs[0]
    x2 = xs[1]
    y1 = ys[0]
    y2 = ys[1]
    m = ( y1 - y2 )/( x1 - x2 )
    c = y2 - m*x2
    yextrap = m*xextrap + c
    return yextrap


