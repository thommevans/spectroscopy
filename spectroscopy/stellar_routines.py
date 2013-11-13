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
                if ixs_bad.max()==True:
                    if i<niterations-1:
                        untainted_frames[j] = 0
                        print 'Flagging frame {0} as containing bad pixels'.format( image_filename )
                    else:
                        if stellar.disp_axis==0:
                            badpix_j[dl:du+1,cl:cu+1][ixs_bad] = 1
                        else:
                            badpix_j[cl:cu+1,dl:du+1][ixs_bad] = 1

            if i==niterations-1:
                if badpix_static!=None:
                    badpix_j *= badpix_static
                if os.path.isfile( image_filepath ):
                    os.remove( image_filepath )
                image_hdu = fitsio.FITS( image_filepath, 'rw' )
                image_hdu.write( None, header=header0 )
                image_hdu.write( current_data, header=header1 )
                image_hdu.write( badpix_j )
                image_hdu.close()
                if i==niterations-1:
                    nbad = badpix_j.sum()
                    if nbad>0:
                        print 'Flagged {0} bad pixels in image {1}'\
                              .format( nbad, image_filename )

    return None

def identify_badpixels_WORKING( stellar ):
    
    # Read in the science images:
    science_images = np.loadtxt( stellar.science_images_list, dtype=str )
    #science_images = science_images[533:]
    
    nimages = len( science_images )
    sciences = []
    badpix_statics = []

    if ( stellar.badpix_static!=None )*( stellar.badpix_static!='' ):
        badpix_path = os.path.join( stellar.adir, stellar.badpix_static )
        badpix_static_hdu = fitsio.FITS( badpix_path )
        badpix_static = badpix_static_hdu[2].read_image()
        badpix_static_hdu.close()
    else:
        badpix_static = None

    # Do niterations passes of bad pixel flagging:
    nslide = 20
    nsigma_thresh = 7
    niterations = 2
    print '\nBad pixel flagging:'
    untainted_frames = np.ones( nimages )
    for i in range( niterations ):
        print ' ... iteration {0} of {1}'.format( i+1, niterations )

        for j in range( nslide, nimages-nslide ):

            # Load current image and measure dimensions:
            image_filename = science_images[j]
            image_root = image_filename[:image_filename.rfind('.')]
            image_filepath = os.path.join( stellar.ddir, image_filename )
            image_hdu_current = fitsio.FITS( image_filepath, 'rw' )
            header0 = image_hdu_current[0].read_header()
            header1 = image_hdu_current[1].read_header()
            current_data = image_hdu_current[1].read_image()
            image_hdu_current.close()

            print '\n', j+1, nimages
            print image_filename

            ixs_before = j - np.arange( nslide ) - 1
            ixs_after = j + np.arange( nslide ) + 1
            ixs_slide = np.concatenate( [ ixs_before, ixs_after ] )

            # If this is the first frame we need to construct
            # the slider frames to compare against:
            if j==nslide:
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
                slider_data = slider_data[:,:,1:]
                # Add a new slide to the leading edge:
                image_filename_lead = science_images[ixs_slide[-1]]
                image_root_lead = image_filename_lead[:image_filename_lead.rfind('.')]
                image_filepath_lead = os.path.join( stellar.ddir, image_filename_lead )
                image_hdu_lead = fitsio.FITS( image_filepath_lead )
                lead_data = image_hdu_lead[1].read_image()
                image_hdu_lead.close()
                slider_data = np.dstack( [ slider_data, lead_data ] )
            # Determine which slider frames are untainted:
            ixs_use = np.arange( 2*nslide )[ untainted_frames[ixs_slide]==1 ]
            # Loop over each star on the image:
            for k in range( stellar.nstars ):
                dl = stellar.disp_bounds[k][0]
                du = stellar.disp_bounds[k][1]
                cl = stellar.crossdisp_bounds[k][0]
                cu = stellar.crossdisp_bounds[k][1]
                crossdisp_ixs = ( crossdisp_pixrange>=cl )*( crossdisp_pixrange<=cu )
                disp_ixs = ( disp_pixrange>=dl )*( disp_pixrange<=du )
                if stellar.disp_axis==0:
                    subdarray = current_data[disp_ixs,:][:,crossdisp_ixs]
                    subslider = slider_data[disp_ixs,:,:][:,crossdisp_ixs,:]
                else:
                    subdarray = current_data[crossdisp_ixs,:][:,disp_ixs]
                    subslider = slider_data[crossdisp_ixs,:,:][:,disp_ixs,:]
                med_sub = np.median( subslider[:,:,ixs_use], axis=2 )
                sig_sub = np.std( subslider[:,:,ixs_use], axis=2 )
                delsigmas_sub = abs( ( subdarray - med_sub )/sig_sub )
            # If this is the last iteration, generate
            # a bad pixel map for the current image:
            if i<niterations-1:
                if delsigmas_sub.max()>nsigma_thresh:
                    untainted_frames[j] = 0
            else:
                med = np.median( slider[:,:,ixs_use], axis=2 )
                sig = np.std( slider[:,:,ixs_use], axis=2 )
                delsigmas = abs( ( current_data - med )/sig )
                if badpix_static!=None:
                    badpix_j = ( delsigmas>nsigma_thresh )*badpix_static
                else:
                    badpix_j = ( delsigmas>nsigma_thresh )
                if os.path.isfile( image_filepath ):
                    os.remove( image_filepath )
                image_hdu = fitsio.FITS( image_filepath, 'rw' )
                image_hdu.write( None, header=header0 )
                image_hdu.write( current_data, header=header1 )
                image_hdu.write( badpix_j )
                image_hdu.close()
                if i==niterations-1:
                    nbad = current_mask.sum()
                    print 'Flagged {0} bad pixels in image {1}'\
                          .format( nbad, image_filename )

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

    # Open files that will store a list of
    # the trace file names:
    science_traces_ofiles = []
    stellar.science_traces_list = []
    for i in range( stellar.nstars ):
        ext = 'science_traces_star{0}.lst'.format( i )
        science_trace_ofilepath = os.path.join( stellar.adir, ext )
        science_traces_ofiles += [ open( science_trace_ofilepath, 'w' ) ]
        stellar.science_traces_list += [ ext ]

    # Keep trace of the spectral trace widths
    # across all images:
    specwidths = np.zeros( [ nimages, stellar.nstars ] )

    # Loop over the science images, fitting
    # for the spectral traces:
    binw_disp = stellar.tracefit_kwargs['binw_disp']
    for j in range( nimages ):
        t1=time.time()
        # Load current image and measure dimensions:
        print '\nFitting traces in image {0} of {1}'.format( j+1, nimages )
        image_filename = science_images[j]
        image_root = image_filename[:image_filename.rfind('.')]
        image_filepath = os.path.join( stellar.ddir, image_filename )
        image_hdu = fitsio.FITS( image_filepath )
        darray = image_hdu[1].read_image()
        badpix = image_hdu[2].read_image()
        darray = np.ma.masked_array( darray, mask=badpix )
        image_hdu.close()

        arr_dims = np.shape( darray )
        disp_pixrange = np.arange( arr_dims[stellar.disp_axis] )
        crossdisp_pixrange = np.arange( arr_dims[stellar.crossdisp_axis] )

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
        s = [] # cross-dispersion profile widths

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
            trace_interps = np.zeros( nbins_disp )
            for i in range( nbins_disp ):

                # Extract the cross-dispersion profile form 
                # the current bin along the dispersion axis:
                dl_i = ledges[i]
                du_i = dl_i + binw_disp

                if stellar.disp_axis==0:
                    crossdisp_prof = np.mean( darray[dl_i:du_i,cl:cu+1], axis=0 )
                else:
                    crossdisp_prof = np.mean( darray[cl:cu+1,dl_i:du_i], axis=1 )

                # Fit the cross-dispersion profile with a
                # Gaussian using least squares:
                A0 = crossdisp_prof[0]
                B0 = 0.
                C0 = np.max( crossdisp_prof ) - A0
                crossdisp_prof_downshift = crossdisp_prof - A0 - 0.1*C0
                ixs = ( crossdisp_prof_downshift>0 )
                pixs = crossdisp_pixs[ixs]
                sig0 = 0.3*( pixs.max() - pixs.min() )
                ix = np.argmax( crossdisp_prof )
                crossdisp_coord0 = crossdisp_pixs[ix]
                pars0 = np.array( [ A0, B0, C0, sig0, crossdisp_coord0 ] )
                pars_optimised = scipy.optimize.leastsq( gauss_resids, \
                                                         pars0, \
                                                         args=( crossdisp_pixs, \
                                                                crossdisp_prof ) )[0]
                A, B, C, sig, trace_interps[i] = pars_optimised

                y_k += [ crossdisp_prof ]
                g_k += [ gauss_profile( crossdisp_pixs, pars_optimised ) ]
                s_k += [ abs( sig ) ]

            specwidths[j,k] = 0.8*np.median( s_k )
            
            # Now that we've fit for the centres of each bin along the
            # dispersion axis, we can interpolate these to a spectral
            # trace evaluated at each pixel along the dispersion axis:
            trace = np.zeros( npix_disp )
            if stellar.tracefit_kwargs['method']=='linear_interpolation':

                # Create the interpolating function:
                interpf = scipy.interpolate.interp1d( bincents_disp, \
                                                      trace_interps, \
                                                      kind='linear' )

                # Interpolate between the bin centers:
                ixs = ( ( disp_pixs>=bincents_disp.min() )\
                        *( disp_pixs<=bincents_disp.max() ) )
                trace[ixs] = interpf( disp_pixs[ixs] )

                # Linearly extrapolate at the edges:
                ixsl = ( disp_pixs<bincents_disp.min() )
                ixsu = ( disp_pixs>bincents_disp.max() )
                trace[ixsl] = linear_extrapolation( disp_pixs[ixsl], \
                                                    bincents_disp[0:2], \
                                                    trace_interps[0:2] )
                trace[ixsu] = linear_extrapolation( disp_pixs[ixsu], \
                                                    bincents_disp[-2:], \
                                                    trace_interps[-2:] )
            else:
                pdb.set_trace() # haven't implemented any other methods yet
            y += [ y_k ]
            g += [ g_k ]
            s += [ s_k ]
            ci += [ bincents_disp ]
            ti += [ trace_interps ]
            t += [ trace ]

            # Save the trace centers for the current image to an output file:
            ofolder_ext = 'trace_files/star{0}'.format( k )
            otracename = '{0}_trace_nbins{1}.npy'.format( image_root, nbins_disp )
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
            print ' ... saved trace fit star{0}/{1:s}'.format( k, otracename )
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
                axprof.set_ylabel( 'star{0}'.format( k ) )
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
            ofigname = '{0}.png'.format( image_root )
            ofigpath = os.path.join( tracedir, ofigname )
            plt.savefig( ofigpath )
            plt.close()
            print ' ... saved figure {0}'.format( ofigname )
        specwidths_str = ''
        for k in range( stellar.nstars ):
            specwidths_str += ' {0:.2f},'.format( specwidths[j,k] )
        print 'PSF widths (pixels) -', specwidths_str[:-1]
        t2=time.time()
        print t2-t1

    # Summarise the PSF width info:
    med = np.median( specwidths, axis=0 ) # median PSF width for each star
    std = np.std( specwidths, axis=0 ) # PSF width scatter for each star
    print '\nPSF widths across all images in units of pixels'
    print '# median, scatter'
    plt.figure()
    for k in range( stellar.nstars ):
        print 'star{0} = {1:.3f}, {2:.3f}'.format( k, med[k], std[k] )
        plt.plot( specwidths[:,k], '-', label='star{0}'.format( k ) )
    plt.ylabel( 'PSF width (pixels)' )
    plt.xlabel( 'Image number' )
    plt.legend()
    ofigname = os.path.join( stellar.adir, 'PSF_widths.png' )
    plt.savefig( ofigname )
    plt.close()
    m = np.median( specwidths, axis=1 ) # median PSF width for each image
    ix = np.argmax( m ) # image number with widest PSF
    print '\nThe frame with the largest median PSF is:'
    print science_images[ix]
    print 'with PSF width of {0:.3f} pixels'.format( m[ix] )
    z = np.std( specwidths, axis=1 ) # spread amongst stars per image
    
    print '\nSaved list of traces for each star in:'
    for k in range( stellar.nstars ):
        science_traces_ofiles[k].close()
        ofilename = os.path.basename( stellar.science_traces_list[k] )
        print 'star{0} --> {1}'.format( k, ofilename )

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
        ext = 'science_spectra_star{0}.lst'.format( k )
        science_spectrum_ofilepath = os.path.join( stellar.adir, ext )
        science_spectra_ofiles += [ open( science_spectrum_ofilepath, 'w' ) ]
        stellar.science_spectra_lists += [ ext ]

    # Read in the list of science images:
    science_images_list = os.path.join( stellar.adir, stellar.science_images_list )
    science_images = np.loadtxt( science_images_list, dtype=str )
    nimages = len( science_images )

    # Loop over each image, and extract the spectrum
    # for each star on each image:
    for j in range( nimages ):

        # Load in the image and header:
        print '\nExtracting spectra from image {0} of {1}'.format( j+1, nimages )
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
        jdobs = header[stellar.header_kws['JD']] \
                + 0.5*header[stellar.header_kws['EXPTIME_SECS']]/60./60./24.

        # Loop over each star:
        for k in range( stellar.nstars ):

            # Define filename for the output spectrum of
            # the current star for the current image:
            ospec_root = 'spec1d_{0}'.format( image_root )
            ospec_name = '{0}.fits'.format( ospec_root )
            ospec_ext = 'spectra/star{0}'.format( k )
            ospec_ext = os.path.join( ospec_ext, ospec_name )
            ospec_filepath = os.path.join( stellar.adir, ospec_ext )
            ofolder = os.path.dirname( ospec_filepath )
            if os.path.isdir( ofolder )!=True:
                os.makedirs( ofolder )
            
            # Identify the dispersion and cross-dispersion
            # pixels of the spectrum data:
            cl = stellar.crossdisp_bounds[k][0]
            cu = stellar.crossdisp_bounds[k][1]
            crossdisp_ixs = ( crossdisp_pixrange>=cl )*( crossdisp_pixrange<=cu )
            crossdisp_pixs = crossdisp_pixrange[crossdisp_ixs]
            dl = stellar.disp_bounds[k][0]
            du = stellar.disp_bounds[k][1]
            disp_ixs = ( disp_pixrange>=dl )*( disp_pixrange<=du )
            disp_pixs = disp_pixrange[disp_ixs]
            npix_disp = len( disp_pixs )

            # Cut out a subarray containing the spectrum data:
            if stellar.disp_axis==0:
                subarray = darray[disp_ixs,:][:,crossdisp_ixs]
            else:
                subarray = darray[:,disp_ixs][crossdisp_ixs,:]

            # Read in the array containing the trace fit:
            trarray = np.loadtxt( trace_files[k][j] )
            apflux = np.zeros( npix_disp )
            nappixs = np.zeros( npix_disp )
            skyppix = np.zeros( npix_disp )

            # Loop over each pixel column along the
            # dispersion axis:
            for i in range( npix_disp ):
                if stellar.disp_axis==0:
                    crossdisp_central_pix = trarray[i,1]
                    crossdisp_row = subarray[i,:]
                else:
                    crossdisp_central_pix = trarray[i,1]
                    crossdisp_row = subarray[i,:]

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
                if ( npix_full + nfracpix_l + nfracpix_u )>2*stellar.spectral_ap_radius:
                    pdb.set_trace() # this shouldn't happen
                elif ( npix_full + nfracpix_l + nfracpix_u )<2*stellar.spectral_ap_radius:
                    print ' - aperture overflowing cross-dispersion edge for star{0} in {1}'\
                          .format( k, image_filename )

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
                lsky_l = max( [ usky_l, crossdisp_pixs.min() ] )
                lsky_u = max( [ usky_u, crossdisp_pixs.min() ] )
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
                    pdb.set_trace() # no valid sky regions
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
                apflux[i] = np.sum( darray_full ) \
                            + darray_fraclower \
                            + darray_fracupper \
                            - sky

                nappixs[i] = npix_full + nfracpix_u + nfracpix_l
                skyppix[i] = sky/float( nappixs[i] )

                if 0*( i%20==0 ):
                    plt.ion()
                    plt.close('all')
                    plt.figure()
                    plt.plot( crossdisp_pixs, crossdisp_row )
                    plt.axvline( crossdisp_central_pix, ls='-' )
                    plt.axvline( crossdisp_central_pix+stellar.spectral_ap_radius, ls='--' )
                    plt.axvline( crossdisp_central_pix-stellar.spectral_ap_radius, ls='--' )
                    pdb.set_trace()

            # Save the spectra for the current star on 
            # the current image in a fits table:
            data = np.zeros( npix_disp, dtype=[ ( 'disp_pixs', np.int64 ), \
                                                ( 'apflux', np.float64 ), \
                                                ( 'nappixs', np.float64 ), \
                                                ( 'skyppix', np.float64 ) ] )
            data['disp_pixs'] = disp_pixs
            data['apflux'] = apflux
            data['nappixs'] = nappixs
            data['skyppix'] = skyppix
            if os.path.isfile( ospec_filepath ):
                os.remove( ospec_filepath )
            fits = fitsio.FITS( ospec_filepath, 'rw' )
            header = { 'IMAGE':image_filename, 'JD-OBS':jdobs }
            fits.write( data, header=header )
            fits.close()
            print ' ... saved {0}'.format( ospec_ext )
            science_spectra_ofiles[k].write( '{0}\n'.format( ospec_ext ) )

    # Save all the lists of spectra files:
    for k in range( stellar.nstars ):
        science_spectra_ofiles[k].close()
        print '\nSaved list of spectra in:\n{0}'.format( stellar.science_spectra_lists[k] )

    return None

    
def calibrate_wavelength_scale( stellar, make_plots=False ):
    """
    Use the pre-determined dispersion pixel coordinates of the fiducial
    lines from the master arc frame to fit a linear mapping from the
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
    pixel_coords = []
    wav_linfit_coeffs = []
    for k in range( stellar.nstars ):
        crossdisp_pixbounds = stellar.wcal_kws['crossdisp_pixbounds'][k]
        pixel_coords_k = []
        for j in range( nlines ):
            disp_pixbounds = stellar.wcal_kws['disp_pixbounds'][k][j]
            dl = disp_pixbounds[0]
            du = disp_pixbounds[1]
            cl = crossdisp_pixbounds[0]
            cu = crossdisp_pixbounds[1]
            if stellar.disp_axis==0:
                disp_prof = np.sum( marc[:,cl:cu+1][dl:du+1,:], axis=1 )
            else:
                disp_prof = np.sum( marc[cl:cu+1,:][:,dl:du+1], axis=0 )
            disp_pixs = np.arange( dl, du+1, 1 )
            
            A0 = disp_prof.min()
            B0 = 0.
            C0 = disp_prof.max() - disp_prof.min()
            ix = np.argmax( disp_prof )
            mu0 = disp_pixs[ix]
            disp_prof_shifted = disp_prof - 0.5*C0
            ixs = ( disp_prof_shifted>0 )
            sig0 = disp_pixs[ixs][-1] - disp_pixs[ixs][0]
            pars0 = np.array( [ A0, B0, C0, sig0, mu0 ] )
            pars_optimised = scipy.optimize.leastsq( gauss_resids, \
                                                     pars0, \
                                                     args=( disp_pixs, \
                                                            disp_prof ) )[0]
            A, B, C, sig, mu = pars_optimised
            pixel_coords_k += [ mu ]
        pixel_coords += [ pixel_coords_k ]

        # Compute the linear mapping from pixel
        # coordinates to wavelength scale:
        wav = np.array( fiducial_lines )
        pix = np.array( pixel_coords_k )
        offset = np.ones( len( pix ) )
        basis = np.column_stack( [ offset, pix ] )
        wav_linfit_coeffs += [ np.linalg.lstsq( basis, wav )[0] ]

        # Evaluate the wavelength solution across
        # the full dispersion axis:
        #offset = np.ones( len( disp_pixrange ) )
        #basis = np.column_stack( [ offset, disp_pixrange ] )
        #wav_solutions += [ np.dot( basis, coeffs ) ]

    # Now go through all the spectra that have already been generated
    # and add a column to the fits table containing the wavelengths:
    print '\nSuccessfully calibrated the wavelength scale'
    if stellar.science_spectra_lists==None:
        print '\nNo science_spectra_lists attribute provided -'
        print 'spectra not updated'
    else:
        print 'Adding wavelength solutions to the existing spectrum files...'
        science_images_list = os.path.join( stellar.adir, stellar.science_images_list )
        science_image_files = np.loadtxt( science_images_list, dtype=str )
        nimages = len( science_image_files )
        spectra_files = []
        for k in range( stellar.nstars ):
            print ' ... star{0}'.format( k )
            spectra_files += [ np.loadtxt( stellar.science_spectra_lists[k], dtype=str ) ]
            for j in range( nimages ):
                spectrum_hdu = fitsio.FITS( spectra_files[k][j], 'rw' )
                disp_pixs = spectrum_hdu[1]['disp_pixs'].read()
                offset = np.ones( len( disp_pixs ) )
                basis = np.column_stack( [ offset, disp_pixs ] )
                wav = np.dot( basis, wav_linfit_coeffs[k] )
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
            for j in range( nimages ):
                print ' ... image {0} of {1}'.format( j+1, nimages )
                plt.figure()
                for k in range( stellar.nstars ):
                    spectrum_file = spectra_files[k][j]
                    spectrum_hdu = fitsio.FITS( spectrum_file, 'r' )
                    wav = spectrum_hdu[1].read_column( 'wav' )
                    spectrum = spectrum_hdu[1].read_column( 'apflux' )
                    spectrum_hdu.close()
                    plt.plot( wav, spectrum, '-', lw=2, zorder=k, label='star{0}'.format( k ) )
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
    #print '\n\n\n{0}'.format(sig)
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


