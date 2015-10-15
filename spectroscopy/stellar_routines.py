import numpy as np
import time
import matplotlib.pyplot as plt
import os, pdb, sys
import scipy.optimize
import scipy.interpolate
import glob
import fitsio
import atpy
import cPickle

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
        if stellar.n_exts==1:
            for k in range( stellar.n_stars ):
                stellar.star_names += [ 'star{0}'.format( k ) ]
        else:
            counter = 0
            stellar.star_names = []
            for i in range( stellar.n_exts ):
                star_names_i = []
                for k in range( stellar.n_stars[i] ):
                    star_names_i += [ 'star{0}'.format( counter ) ]
                    counter += 1
                stellar.star_names += [ star_names_i ]
    
    # Read in the science images:
    science_images_list_path = os.path.join( stellar.adir, stellar.science_images_list )
    science_images = np.loadtxt( science_images_list_path, dtype=str )
    nimages = len( science_images )
    sciences = []
    badpix_statics = []

    if ( stellar.badpix_static!=None )*( stellar.badpix_static!='' ):
        badpix_path = os.path.join( stellar.adir, stellar.badpix_static )
        badpix_static_hdu = fitsio.FITS( badpix_path )
        badpix_static = badpix_static_hdu[1].read()
        badpix_static_hdu.close()
    else:
        badpix_static = None

    # Load the list of filenames that will be used to store
    # the bad pixel output:
    badpix_map_list_path = os.path.join( stellar.adir, stellar.badpix_maps_list )
    badpix_map_filenames = np.loadtxt( badpix_maps_list_path, dtype=str )
    if len( badpix_map_filenames )!=nimages:
        pdb.set_trace() # should be one bad pixel map per image

    # Do niterations passes of bad pixel flagging:
    nslide = 15
    nsigma_thresh = 10
    niterations = 2
    print '\nBad pixel flagging, using {0} iterations and a {1}-sigma threshold:'\
          .format( niterations, nsigma_thresh )
    # Loop over each FITS file:
    for j in range( nimages ):
        # Load current image and measure dimensions of current extension:
        image_filename = science_images[j]
        image_root = image_filename[:image_filename.rfind('.')]
        image_filepath = os.path.join( stellar.ddir, image_filename )
        image_hdu_current = fitsio.FITS( image_filepath, 'rw' )
        header0 = image_hdu_current[0].read_header()
        # Open a FITS HDU for the bad pixel map:
        badpix_filename = badpix_map_filenames[j]
        badpix_filepath = os.path.join( stellar.ddir, badpix_filename )
        if os.path.isfile( badpix_filepath ):
            os.remove( badpix_filepath )
        badpix_hdu = fitsio.FITS( badpix_filepath, 'rw' )
        badpix_hdu.write( None, header=header0 )
        # Loop over each FITS extension individually:
        for k in range( stellar.n_exts ):
            headerk = image_hdu_current[k+1].read_header()
            current_data = image_hdu_current[k+1].read()
            dims = np.shape( current_data )
            disp_pixrange = np.arange( dims[stellar.disp_axis] )
            crossdisp_pixrange = np.arange( dims[stellar.crossdisp_axis] )
            if (j+1)%20==0:
                print '... up to image {0} of {1}, chip {2} of {3}'\
                      .format( j+1, nimages, k+1, stellar.n_exts )
            # Array that keeps a record of those frames that have
            # not previously been flagged as bad, and hence can be
            # used in the slider:
            untainted_frames = np.ones( nimages )
            # Loop over each star on the image in order to create the
            # bad pixel map for the current HDU extension:
            badpix_jk = np.zeros( dims )
            for i in range( niterations ):
                #print ' ... iteration {0} of {1} (for chip {2} of {3})'\
                #      .format( i+1, niterations, k+1, stellar.n_exts )


                # Determine the indices of the frames in the slider before and
                # after the current frame:
                ixs_before = j - np.arange( nslide ) - 1
                ixs_after = j + np.arange( nslide ) + 1
                ixs_before = ixs_before[ixs_before>=0]
                ixs_after = ixs_after[ixs_after<nimages]
                ixs_before = ixs_before[np.argsort( ixs_before )]
                ixs_after = ixs_after[np.argsort( ixs_after )]
                ixs_slide = np.concatenate( [ ixs_before, [j], ixs_after ] )
                # The number of frames in the slider:
                ncontrol = len( ixs_before ) + len( ixs_after )

                # If this is the first frame we need to construct
                # the slider frames to compare against:
                if j==0:
                    slider_data = []
                    for jj in ixs_slide:
                        image_filename_jj = science_images[jj]
                        image_root_jj = image_filename_jj[:image_filename_jj.rfind('.')]
                        image_filepath_jj = os.path.join( stellar.ddir, image_filename_jj )
                        image_hdu_jj = fitsio.FITS( image_filepath_jj )
                        slider_data += [ image_hdu_jj[k+1].read() ]
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
                        lead_data = image_hdu_lead[k+1].read()
                        image_hdu_lead.close()
                        slider_data = np.dstack( [ slider_data, lead_data ] )
                # Determine which slider frames are untainted:
                untainted = ( untainted_frames[ixs_slide]==1 )*( ixs_slide!=j )
                ixs_use = np.arange( ncontrol+1 )[untainted]

                # Note that the number of stars depends on which
                # FITS extension we are analysing:
                if stellar.n_exts==1:
                    nstars_k = stellar.nstars
                else:
                    nstars_k = stellar.nstars[k]
                for l in range( nstars_k ):
                    if stellar.n_exts==1:
                        dl = stellar.disp_bounds[l][0]
                        du = stellar.disp_bounds[l][1]
                        cl = stellar.crossdisp_bounds[l][0]
                        cu = stellar.crossdisp_bounds[l][1]
                    else:
                        dl = stellar.disp_bounds[k][l][0]
                        du = stellar.disp_bounds[k][l][1]
                        cl = stellar.crossdisp_bounds[k][l][0]
                        cu = stellar.crossdisp_bounds[k][l][1]
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
                                  .format( image_filename, stellar.star_names[l] )
                    if stellar.disp_axis==0:
                        badpix_jk[dl:du+1,cl:cu+1][ixs_bad] = 1
                    else:
                        badpix_jk[cl:cu+1,dl:du+1][ixs_bad] = 1

                if i==niterations-1:
                    nbad_transient = badpix_jk.sum()
                    if badpix_static!=None:
                        badpix_jk += badpix_static

                    # If bad pixels have been flagged more than once,
                    # set their value to 1 in the bad pixel map:
                    ixs = ( badpix_jk!=0 )
                    badpix_jk[ixs] = 1
                    # Write the bad pixel map for the current extension
                    # to the open FITS HDU:
                    badpix_hdu.write( badpix_jk, header=headerk )

                    nbad_total = badpix_jk.sum()
                    if nbad_transient>0:
                        print 'Flagged {0:.0f} transient bad pixels on chip {1} in image {2} of {3} ({4})'\
                              .format( nbad_transient, k+1, j+1, nimages, image_filename )

        # Having looped over all HDU extensions, 
        # close the open FITS HDU and save the bad
        # pixel map:
        image_hdu_current.close()
        badpix_hdu.close()
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
    science_images_full_list_path = os.path.join( stellar.adir, stellar.science_images_full_list )
    science_images_full = np.loadtxt( science_images_full_list_path, dtype=str )
    nimages = len( science_images_full )
    badpix_maps_full_list_path = os.path.join( stellar.adir, stellar.badpix_maps_full_list )
    badpix_maps_full = np.loadtxt( badpix_maps_full_list_path, dtype=str )
    if len( badpix_maps_full )!=nimages:
        pdb.set_trace()


    # Load the list of filenames that will be used to store
    # the bad pixel output:
    #badpix_maps_list_path = os.path.join( stellar.adir, stellar.badpix_maps_list )
    #badpix_map_filenames = np.loadtxt( badpix_maps_list_path, dtype=str )
    #if len( badpix_map_filenames )!=nimages:
    #    pdb.set_trace() # should be one bad pixel map per image

    # If there is no goodbad array, assume all FITS files
    # are okay:
    if stellar.goodbad==None:
        stellar.goodbad = np.ones( nimages )
    nimages=5#delete

    # Open files that will store a list of
    # the trace file names:
    science_traces_ofiles = [] # to store list of trace files
    #science_images_ofiles = [] # to store list of corresponding images
    #science_badpix_ofiles = [] # to store list of corresponding badpix maps
    #stellar.science_traces_list = []
    #stellar.science_images_list = []
    if stellar.n_exts==1:
        for k in range( stellar.nstars ):
            #ext = 'science_traces_{0}.lst'.format( stellar.star_names[k] )
            science_trace_ofilepath = os.path.join( stellar.adir, stellar.science_traces_list[k] )
            science_traces_ofiles += [ open( science_trace_ofilepath, 'w' ) ]
            #stellar.science_traces_list += [ ext ]
            #ext = 'science_images_{0}.lst'.format( stellar.star_names[k] )
        science_image_ofilepath = os.path.join( stellar.adir, stellar.science_images_list )
        science_images_ofile = open( science_image_ofilepath, 'w' )
        badpix_maps_ofile = open( badpix_maps_ofilepath, 'w' )
    else:
        #stellar.science_traces_list = []
        science_traces_ofiles = []
        #stellar.science_images_list = []
        #science_images_ofiles = []
        for k in range( stellar.n_exts ):
            science_traces_list_k = []
            science_traces_ofiles_k = []
            science_images_list_k = []
            #science_images_ofiles_k = []
            for i in range( stellar.nstars[k] ):
                #ext = 'science_traces_{0}.lst'.format( stellar.star_names[k][i] )
                science_trace_ofilepath = os.path.join( stellar.adir, stellar.science_traces_list[k][i] )
                science_traces_ofiles_k += [ open( science_trace_ofilepath, 'w' ) ]
                #science_traces_list_k += [ ext ]
                #ext = 'science_images_{0}.lst'.format( stellar.star_names[k][i] )
                #science_images_ofilepath = os.path.join( stellar.adir, stellar.science_images_list[k][i] )
                #science_images_ofiles_k += [ open( science_images_ofilepath, 'w' ) ]
                #science_images_list_k += [ ext ]
            #stellar.science_traces_list += [ science_traces_list_k ]
            science_traces_ofiles += [ science_traces_ofiles_k ]
            #stellar.science_images_list += [ science_images_list_k ]
            #science_images_ofiles += [ science_images_ofiles_k ]
    science_image_ofilepath = os.path.join( stellar.adir, stellar.science_images_list )
    science_images_ofile = open( science_image_ofilepath, 'w' )
    badpix_maps_ofilepath = os.path.join( stellar.adir, stellar.badpix_maps_list )
    badpix_maps_ofile = open( badpix_maps_ofilepath, 'w' )

    # Loop over the science images, fitting the spectral traces:
    binw_disp = stellar.tracefit_kwargs['binw_disp']
    fwhms = []
    for k in range( stellar.n_exts ):
        fwhms_k = []
        if stellar.n_exts==1:
            nstars_k = stellar.nstars
        else:
            nstars_k = stellar.nstars[k]
        for i in range( nstars_k ):
            fwhms_k += [ np.zeros( nimages ) ]
        fwhms += [ fwhms_k ]

    for j in range( nimages ):

        # First check if the current image has been
        # flagged as bad:
        if stellar.goodbad[j]==1:
            print '\nFitting traces in image {0} of {1}'.format( j+1, nimages )
        else:
            print '\nImage {0} of {1} flagged as bad - skipping'.format( j+1, nimages )
            continue

        t1=time.time()
        # Load current image and associated bad pixel map:
        image_filename = science_images_full[j]
        badpix_map_filename = badpix_maps_full[j]
        science_images_ofile.write( '{0}\n'.format( image_filename ) ) 
        badpix_maps_ofile.write( '{0}\n'.format( badpix_map_filename ) ) 
        image_root = image_filename[:image_filename.rfind('.')]
        image_filepath = os.path.join( stellar.ddir, image_filename )
        image_hdu = fitsio.FITS( image_filepath )
        badpix_map_filepath = os.path.join( stellar.ddir, badpix_map_filename )
        badpix_hdu = fitsio.FITS( badpix_map_filepath )

        # Loop over each FITS extension individually:
        for k in range( stellar.n_exts ):

            darray = image_hdu[k+1].read()
            badpix = badpix_hdu[k+1].read()
            darray = np.ma.masked_array( darray, mask=badpix )

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

            if stellar.n_exts==1:
                nstars_k = stellar.nstars
            else:
                nstars_k = stellar.nstars[k]
            
            for i in range( nstars_k ):

                if stellar.n_exts==1:
                    star_name_i = stellar.star_names[i]
                    traces_ofile_i = science_traces_ofiles[i]
                    #images_ofile_i = science_images_ofiles[i]
                    crossdisp_bounds_i = stellar.crossdisp_bounds[i]
                    disp_bounds_i = stellar.disp_bounds[i]
                else:
                    star_name_i = stellar.star_names[k][i]
                    traces_ofile_i = science_traces_ofiles[k][i]
                    #images_ofile_i = science_images_ofiles[k][i]
                    crossdisp_bounds_i = stellar.crossdisp_bounds[k][i]
                    disp_bounds_i = stellar.disp_bounds[k][i]
                # Get the upper and lower edges of the window
                # in the cross-dispersion range that contains
                # the trace:
                cl = crossdisp_bounds_i[0]
                cu = crossdisp_bounds_i[1]
                dl = disp_bounds_i[0]
                du = disp_bounds_i[1]

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
                for v in range( nbins_disp ):

                    # Extract the cross-dispersion profile form 
                    # the current bin along the dispersion axis:
                    dl = ledges[v]
                    du = dl + binw_disp
                    if stellar.disp_axis==0:
                        binwindow = darray[dl:du,cl:cu+1]
                    else:
                        binwindow = darray[cl:cu+1,dl:du]

                    nbad = binwindow.mask.sum()
                    ntot = binwindow.mask.size
                    frac_bad = float( nbad )/ntot
                    if frac_bad>0.2:
                        print 'Too many bad pixels for bin - skipping'
                        continue
                    else:
                        if stellar.disp_axis==0:
                            sky_rough_l = np.median( darray[dl:du,cl:cl+2] )
                            sky_rough_u = np.median( darray[dl:du,cu-2:cu] )
                            sky_slope = ( sky_rough_u - sky_rough_l )/( cu-cl+1 )
                            sky_rough = sky_slope*np.arange( cu-cl+1 ) + sky_rough_l
                            raw_flux = np.median( darray[dl:du,cl:cu+1], axis=0 )
                            crossdisp_prof = raw_flux - sky_rough
                        else:
                            sky_rough_l = np.median( darray[cl:cl+2,dl:du].flatten() )
                            sky_rough_u = np.median( darray[cu-2:cu,dl:du].flatten() )
                            sky_slope = ( sky_rough_u - sky_rough_l )/( cu-cl+1 )
                            sky_rough = sky_slope*np.arange( cu-cl+1 ) + sky_rough_l
                            raw_flux = np.median( darray[cl:cu+1,dl:du], axis=1 )
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
                    A, B, C, sig, trace_interps_v = pars_optimised
                    bincents_disp_good += [ bincents_disp[v] ]
                    trace_interps += [ trace_interps_v ]

                    y_k += [ crossdisp_prof ]
                    g_k += [ gauss_profile( crossdisp_pixs, pars_optimised ) ]
                    s_k += [ abs( sig ) ]

                ninterp = len( trace_interps )
                if ninterp<2:
                    print 'Could not extract trace - skipping frame'
                    pdb.set_trace()
                    continue

                fwhm_jki = np.median( 2.355*np.array( s_k ) )

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
                    ixsl = disp_pixs<bincents_disp_good.min()
                    ixsu = disp_pixs>bincents_disp_good.max()

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

                # Save the trace centers for the current image to an output file:
                ofolder_ext = 'trace_files/{0}'.format( star_name_i )
                otracename = '{0}_nbins{1}.fits'.format( image_root, nbins_disp )
                ofile_ext = os.path.join( ofolder_ext, otracename )
                otrace_filepath = os.path.join( stellar.adir, ofile_ext )
                if os.path.isdir( os.path.dirname( otrace_filepath ) )!=True:
                    os.makedirs( os.path.dirname( otrace_filepath ) )
                if os.path.isfile( otrace_filepath ):
                    os.remove( otrace_filepath )
                otrace = fitsio.FITS( otrace_filepath, 'rw' )
                data = np.zeros( npix_disp, dtype=[ ( 'DISPPIXS', np.float64 ), \
                                                    ( 'TRACE', np.float64 ) ] )
                data['DISPPIXS'] = disp_pixs
                data['TRACE'] = np.array( trace )
                header = {}
                header['METHOD'] = stellar.tracefit_kwargs['method']
                header['FWHM'] = fwhm_jki
                header['IMAGE'] = image_filename
                otrace.write( data, header=header )
                otrace.close()
                print ' ... saved trace fit {0}/{1:s}'.format( star_name_i, otracename )
                # and write to open file:
                traces_ofile_i.write( '{0}\n'.format( ofile_ext ) )
                #images_ofile_i.write( '{0}\n'.format( image_filename ) )

            # once all the stars had been cycled through.... would make a plot
            # now will be edited to make a figure for each fits extension...
            if make_plots==True:
                try:
                    tracedir = os.path.join( stellar.adir, 'trace_pngs' )
                    if os.path.isdir( tracedir )!=True:
                        os.makedirs( tracedir )

                    fig = plt.figure( figsize = [ 15, 11 ] )
                    fig.suptitle( image_filename, fontsize=16 )

                    buff = 0.05
                    nrows = nstars_k
                    axh = ( 1. - nrows*buff - 2*buff )/float( nrows )
                    axw = ( 1. - 4*buff )/3.
                    xlow1 = 1.5*buff
                    xlow2 = xlow1 + axw + buff
                    xlow3 = xlow2 + axw + buff
                    for i in range( nstars_k ):
                        row_number = i%nrows + 1
                        ylow = 1. - buff - row_number*( axh + 0.5*buff )
                        if i==0:
                            axprof = fig.add_axes( [ xlow1, ylow, axw, axh ] )
                            axtr = fig.add_axes( [ xlow2, ylow, axw, axh ] )
                            axspec = fig.add_axes( [ xlow3, ylow, axw, axh ] )
                            axspec0 = axspec
                            axprof0 = axprof
                            axtr0 = axtr
                            axprof.set_title( 'cross-disp profile' )
                            axtr.set_title( 'trace fit' )
                            axspec.set_title( 'raw spectrum' )
                            ymax = np.concatenate( y[i] ).max()
                            specmax = spectra[i].max()
                        else:
                            axprof = fig.add_axes( [ xlow1, ylow, axw, axh ], sharey=axprof0 )
                            axtr = fig.add_axes( [ xlow2, ylow, axw, axh ], sharey=axtr0 )
                            axspec = fig.add_axes( [ xlow3, ylow, axw, axh ], sharey=axspec0 )
                        axprof.set_ylabel( '{0}'.format( star_name_i ) )
                        if i==nstars_k-1:
                            axprof.set_xlabel( 'cross-disp pixel coord' )
                            axtr.set_xlabel( 'disp pixel coord' )
                            axspec.set_xlabel( 'disp pixel coord' )
                        # Plot the raw stellar spectrum:
                        axspec.plot( xd[i], spectra[i], '-k', lw=2 )
                        # Plot the cross-dispersion profile:
                        for v in range( len( y[i] ) ):
                            axprof.plot( xc[i], y[i][v], '-k', lw=2 )
                            axprof.plot( xc[i], g[i][v], '-g', lw=1 )
                        xlow = max( [ np.median( np.array( ti[i] ) - 15*np.array( s[i] ) ), \
                                      crossdisp_bounds_i[0] ] )
                        xupp = min( [ np.median( np.array( ti[i] ) + 15*np.array( s[i] ) ), \
                                      crossdisp_bounds_i[1] ] )
                        axprof.set_xlim( [ xlow, xupp ] )
                        axprof.text( 0.05, 0.85, 'profile stdv = {0:.2f} pix'.format( np.median( s[i] ) ), \
                                     fontsize=8, horizontalalignment='left', transform=axprof.transAxes )
                        # Plot the trace fit:
                        axtr.plot( xd[i], t[i] - np.median( t[i] ), '-r', lw=2 )
                        axtr.plot( ci[i], ti[i] - np.median( t[i] ), 'o', mec='k', mfc='k', ms=7 )
                        axtr.fill_between( xd[i], \
                                           t[i] - np.median( t[i] ) - np.median( s[i] ), \
                                           t[i] - np.median( t[i] ) + np.median( s[i] ), \
                                           color=[0.8,0.8,0.8] )
                    axprof0.set_ylim( [ 0, 1.1*ymax ] )
                    axtr0.set_ylim( [ -10, +10 ] )
                    axspec0.set_ylim( [ 0, 1.1*specmax ] )
                    ofigname = 'traces_{0}_ext{1}.png'.format( image_root, k+1 )
                    ofigpath = os.path.join( tracedir, ofigname )
                    plt.savefig( ofigpath )
                    plt.close()
                    print ' ... saved figure: {0}'.format( ofigpath )
                except:
                    print 'Unable to generate figure for {0} - skipping'.format( ofigname )
                    continue 
            fwhm_str = ''
            for i in range( nstars_k ):
                fwhm_str += ' {0:.2f},'.format( fwhm_jki )
            print 'PSF FWHMs (pixels) -', fwhm_str[:-1]
            fwhms[k][i][j] = fwhm_jki
            t2=time.time()
            #print t2-t1
        # Having looped over all extensions 
        # in the current HDU, close it:
        image_hdu.close()

    # Summarise the PSF FWHM info:
    #med = np.median( fwhms, axis=0 ) # median PSF FWHM for each star
    #std = np.std( fwhms, axis=0 ) # PSF FWHM scatter for each star
    print '\nPSF FWHMs across all images in units of pixels'
    print '# median, scatter'
    plt.figure()
    fwhm_med_arrs = []
    for k in range( stellar.n_exts ):
        if stellar.n_exts==1:
            nstars_k = stellar.nstars
        else:
            nstars_k = stellar.nstars[k]
        fwhm_arr_k = np.zeros( [ nimages, nstars_k ] )
        for i in range( nstars_k ):
            fwhm_arr_k[:,i] = fwhms[k][i]
            # Statistics accross all nights
            med = np.median( fwhms[k][i] )
            std = np.std( fwhms[k][i] )
            if stellar.n_exts==1:
                star_name = stellar.star_names[i]
            else:
                star_name = stellar.star_names[k][i]
            print '{0} = {1:.3f}, {2:.3f}'.format( star_name, med, std )
            plt.plot( fwhms[k][i], '-', label='{0}'.format( star_name ) )
        fwhm_med_arrs += [ np.median( fwhm_arr_k, axis=1 ) ]
    fwhm_med_arrs = np.column_stack( fwhm_med_arrs )

    plt.ylabel( 'PSF FWHM (pixels)' )
    plt.xlabel( 'Image number' )
    plt.legend()
    ofigpath = os.path.join( stellar.adir, 'psf_fwhms.png' )
    plt.savefig( ofigpath )
    plt.close()
    ofilepath = os.path.join( stellar.adir, 'psf_fwhms.pkl' )
    #np.savetxt( ofilepath, fwhms )
    ofile = open( ofilepath, 'w' )
    cPickle.dump( fwhms, ofile )
    ofile.close()

    m = np.median( fwhm_med_arrs, axis=1 ) # median PSF FWHM for each image
    ix = np.argmax( m ) # image number with widest PSF
    print '\nThe frame with the largest median PSF is:\nImage {0} of {1} ({2}'\
          .format( ix+1, nimages, science_images_full[ix] )
    print 'with PSF FWHM of {0:.3f} pixels'.format( m[ix] )
    z = np.std( fwhm_med_arrs, axis=1 ) # spread amongst stars per image
    
    print '\nSaved PSF FWHMs for each star in each frame in:\n{0}'\
          .format( ofilepath )
    print 'Saved corresponding figure:\n{0}'.format( ofigpath )

    print '\nSaved list of traces for each star in:'
    if stellar.n_exts==1:
        for i in range( stellar.nstars ):
            science_traces_ofiles[i].close()
            print ' {0}. {1} --> {2}'\
                  .format( i+1, stellar.star_names[i], stellar.science_traces_list[i] )
    else:
        counter = 0
        for k in range( stellar.n_exts ):
            for i in range( stellar.nstars[k] ):
                counter += 1
                science_traces_ofiles[k][i].close()
                print ' {0}. {1} --> {2}'.format( counter, stellar.star_names[k][i], \
                                                  stellar.science_traces_list[k][i] )
    print 'with corresponding list of science images and badpix maps in:\n {0}\n {1}'\
          .format( stellar.science_images_list, stellar.badpix_maps_list )
    plt.ion()

    return None



def extract_spectra( stellar ):

    # Read in the list of science images and traces:
    science_images_list_path = os.path.join( stellar.adir, stellar.science_images_list )
    science_images = np.loadtxt( science_images_list_path, dtype=str )
    badpix_maps_list_path = os.path.join( stellar.adir, stellar.badpix_maps_list )
    badpix_maps = np.loadtxt( badpix_maps_list_path, dtype=str )
    # TRACE LISTS = 1 FOR EACH EXTENSION IN EACH IMAGE
    #science_traces_list_path = os.path.join( stellar.adir, stellar.science_traces_list )
    #science_traces = np.loadtxt( science_traces_list_path, dtype=str )
    nimages = len( science_images )
    if stellar.goodbad==None:
        stellar.goodbad = np.ones( nimages )
    if len( badpix_maps )!=nimages:
        pdb.set_trace()

    # Loop over each image, and extract the spectrum
    # for each star on each image:
    eps = 1e-10
    stellar.science_spectra_lists = []
    for j in range( nimages ):
        # First check if the current image has been
        # flagged as bad:
        if stellar.goodbad[j]==1:
            print '\nFitting traces in image {0} of {1}'.format( j+1, nimages )
        else:
            print '\nImage {0} of {1} flagged as bad - skipping'.format( j+1, nimages )
            continue
        # Load current image and associated bad pixel map:
        image_filename = science_images[j]
        #image_root = image_filename[:image_filename.rfind('.')]
        image_filepath = os.path.join( stellar.ddir, image_filename )
        image_hdu = fitsio.FITS( image_filepath )
        badpix_filename = badpix_maps[j]
        badpix_filepath = os.path.join( stellar.ddir, badpix_filename )
        badpix_hdu = fitsio.FITS( badpix_filepath )

        # Loop over each FITS extension individually:
        for k in range( stellar.n_exts ):

            if stellar.n_exts==1:
                nstars_k = stellar.nstars
            else:
                nstars_k = stellar.nstars[k]
            
            #for k in range( stellar.nstars ):
            for i in range( nstars_k ):

                #print '\nDoing spectra series for {0}:'.format( k+1, stellar.star_names[k] )

                # Read in the trace filenames for each star on each
                # image and open files to which lists of the extracted
                # spectra will be saved to:
                #trace_files = []
                #science_spectra_ofiles = []
                #stellar.science_spectra_lists = []
                if stellar_n_exts==1:
                    trace_list_file_ki = os.path.join( stellar.adir, stellar.science_traces_lists[i] )
                    star_name_ki = stellar.star_names[i]
                else:
                    trace_list_file_ki = os.path.join( stellar.adir, stellar.science_traces_lists[k][i] )
                    star_name_ki = stellar.star_names[k][i]
                trace_files_ki = np.loadtxt( trace_list_file_ki, dtype=str )
                science_spectra_list_ext_ki = 'science_spectra_{0}.lst'.format( star_name_ki )
                science_spectra_list_ofilepath_ki = os.path.join( stellar.adir, science_spectra_list_ext_ki )
                science_spectra_list_ofile_ki = open( science_spectra_list_ofilepath_ki, 'w' )
                stellar.science_spectra_lists += [ science_spectra_list_ext_ki ]
                ntraces = len( trace_files_ki )
                pdb.set_trace()

                # Identify the dispersion and cross-dispersion
                # pixels of the spectrum data:
                cl = stellar.crossdisp_bounds[k][0]
                cu = stellar.crossdisp_bounds[k][1]
                dl = stellar.disp_bounds[k][0]
                du = stellar.disp_bounds[k][1]

                for i in range( ntraces ):

                    # Read in the array containing the trace fit:
                    trace_filepath_ki = os.path.join( stellar.adir, trace_files_k[i] )
                    trace_hdu = fitsio.FITS( trace_filepath_ki )
                    fwhm_ki = trace_hdu[1].read_header()['FWHM']
                    image_filename = trace_hdu[1].read_header()['IMAGE']
                    disp_pixs = trace_hdu[1].read_column( 'DISPPIXS') 
                    trarray = trace_hdu[1].read_column( 'TRACE') 

                    # Load in the image and header:
                    print '\n ... extracting spectrum from image {0} of {1}'.format( i+1, ntraces )
                    image_root = image_filename[:image_filename.rfind('.')]
                    image_filepath = os.path.join( stellar.ddir, image_filename )
                    image_hdu = fitsio.FITS( image_filepath )
                    header = image_hdu[1].read_header()
                    darray = image_hdu[1].read()
                    badpix = image_hdu[2].read()
                    darray = np.ma.masked_array( darray, mask=badpix )
                    arr_dims = np.shape( darray )
                    disp_pixrange = np.arange( arr_dims[stellar.disp_axis] )
                    crossdisp_pixrange = np.arange( arr_dims[stellar.crossdisp_axis] )        
                    crossdisp_ixs = ( crossdisp_pixrange>=cl )*( crossdisp_pixrange<=cu )
                    disp_ixs = ( disp_pixrange>=disp_pixs.min() )*\
                               ( disp_pixrange<=disp_pixs.max() )
                    if ( stellar.header_kws['gain']==None ):
                        gain = 1.0
                    else:
                        gain = header[stellar.header_kws['gain']]
                    image_hdu.close()

                    # Cut out a subarray containing the spectrum data:
                    if stellar.disp_axis==0:
                        subarray = darray[disp_ixs,:][:,crossdisp_ixs]
                    else:
                        subarray = darray[:,disp_ixs][crossdisp_ixs,:]
                    nbad = subarray.mask.sum()
                    ntot = subarray.mask.size
                    frac_bad = float( nbad )/ntot
                    if frac_bad>0.2:
                        print '\nToo many bad pixels in:\n{0}\n(skipping)\n'\
                              .format( image_filepath )
                        stellar.goodbad[i] = 0
                        continue

                    # The pixels along the dispersion and cross-dispersion
                    # axes of the subarray:
                    crossdisp_pixs = crossdisp_pixrange[crossdisp_ixs]
                    disp_pixs = disp_pixrange[disp_ixs]
                    npix_disp = len( disp_pixs )

                    # Arrays to hold the spectral extraction output:
                    apflux = np.zeros( npix_disp )
                    nappixs = np.zeros( npix_disp )
                    skyppix = np.zeros( npix_disp )

                    # Loop over each pixel column along the dispersion axis
                    # to extract the spectrum:
                    for i in range( npix_disp ):

                        crossdisp_central_pix = trarray[i]
                        if stellar.disp_axis==0:
                            crossdisp_row = subarray[i,:]
                        else:
                            crossdisp_row = subarray[:,i]

                        # Before proceeding, make sure the fitted trace
                        # center is actually located within the defined
                        # cross-dispersion limits; if it's not, it suggests
                        # that something went wrong and this is probably
                        # an unusable frame, so skip it:
                        if ( crossdisp_central_pix<crossdisp_pixs.min() )+\
                           ( crossdisp_central_pix>crossdisp_pixs.max() ):
                            print '\nProblem with trace fit for dispersion column {0} of {1}'\
                                  .format( disp_pixs[i], os.path.basename( image_filepath ) )
                            print '(may be worth inspecting the corresponding image)\n'
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
                        nfracpix_u = ( crossdisp_central_pix + stellar.spectral_ap_radius ) - ( u_full + 1 )
                        nfracpix_l = l_full - ( crossdisp_central_pix - stellar.spectral_ap_radius )

                        # Simple sanity check:
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
                        apflux[i] = np.sum( darray_full ) \
                                    + darray_fraclower \
                                    + darray_fracupper \
                                    - sky
                        nappixs[i] = npix_full + nfracpix_u + nfracpix_l
                        skyppix[i] = sky/float( nappixs[i] )

                    # Save the spectra for the current star in the current
                    # image to a fits table:
                    data = np.zeros( npix_disp, dtype=[ ( 'disp_pixs', np.int64 ), \
                                                        ( 'apflux', np.float64 ), \
                                                        ( 'nappixs', np.float64 ), \
                                                        ( 'skyppix', np.float64 ) ] )
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

                    data['disp_pixs'] = disp_pixs
                    data['apflux'] = apflux*gain
                    data['nappixs'] = nappixs
                    data['skyppix'] = skyppix*gain
                    if os.path.isfile( ospec_filepath ):
                        os.remove( ospec_filepath )
                    fits = fitsio.FITS( ospec_filepath, 'rw' )
                    trace_filename = os.path.basename( trace_files_k[i] )
                    header = { 'IMAGE':image_filename, 'TRACE':trace_filename, 'FWHM':fwhm_kj }
                    fits.write( data, header=header )
                    fits.close()
                    print ' ... saved {0}'.format( ospec_filepath )
                    science_spectra_list_ofile_k.write( '{0}\n'.format( ospec_ext ) )

            # Save the current list of spectra files:
            science_spectra_list_ofile_k.close()
            print '\nSaved list of spectra in:\n{0}'.format( science_spectra_list_ofilepath_k )

    return None

    
def extract_spectra_BACKUP( stellar ):

    # Read in the list of science images:
    science_images_list = os.path.join( stellar.adir, stellar.science_images_list )
    science_images = np.loadtxt( science_images_list, dtype=str )
    nimages = len( science_images )
    if stellar.goodbad==None:
        stellar.goodbad = np.ones( nimages )
    
    # Loop over each image, and extract the spectrum
    # for each star on each image:
    eps = 1e-10
    stellar.science_spectra_lists = []
    for k in range( stellar.nstars ):

        print '\nDoing spectra series for {0}:'.format( k+1, stellar.star_names[k] )

        # Read in the trace filenames for each star on each
        # image and open files to which lists of the extracted
        # spectra will be saved to:
        #trace_files = []
        #science_spectra_ofiles = []
        #stellar.science_spectra_lists = []
        trace_list_file_k = os.path.join( stellar.adir, stellar.science_traces_list[k] )
        trace_files_k = np.loadtxt( trace_list_file_k, dtype=str )
        science_spectra_list_ext_k = 'science_spectra_{0}.lst'.format( stellar.star_names[k] )
        science_spectra_list_ofilepath_k = os.path.join( stellar.adir, science_spectra_list_ext_k )
        science_spectra_list_ofile_k = open( science_spectra_list_ofilepath_k, 'w' )
        stellar.science_spectra_lists += [ science_spectra_list_ext_k ]
        ntraces = len( trace_files_k )

        # Identify the dispersion and cross-dispersion
        # pixels of the spectrum data:
        cl = stellar.crossdisp_bounds[k][0]
        cu = stellar.crossdisp_bounds[k][1]
        dl = stellar.disp_bounds[k][0]
        du = stellar.disp_bounds[k][1]

        for j in range( ntraces ):

            # Read in the array containing the trace fit:
            trace_filepath_kj = os.path.join( stellar.adir, trace_files_k[j] )
            trace_hdu = fitsio.FITS( trace_filepath_kj )
            fwhm_kj = trace_hdu[1].read_header()['FWHM']
            image_filename = trace_hdu[1].read_header()['IMAGE']
            disp_pixs = trace_hdu[1].read_column( 'DISPPIXS') 
            trarray = trace_hdu[1].read_column( 'TRACE') 

            # Load in the image and header:
            print '\n ... extracting spectrum from image {0} of {1}'.format( j+1, ntraces )
            image_root = image_filename[:image_filename.rfind('.')]
            image_filepath = os.path.join( stellar.ddir, image_filename )
            image_hdu = fitsio.FITS( image_filepath )
            header = image_hdu[1].read_header()
            darray = image_hdu[1].read()
            badpix = image_hdu[2].read()
            darray = np.ma.masked_array( darray, mask=badpix )
            arr_dims = np.shape( darray )
            disp_pixrange = np.arange( arr_dims[stellar.disp_axis] )
            crossdisp_pixrange = np.arange( arr_dims[stellar.crossdisp_axis] )        
            crossdisp_ixs = ( crossdisp_pixrange>=cl )*( crossdisp_pixrange<=cu )
            disp_ixs = ( disp_pixrange>=disp_pixs.min() )*\
                       ( disp_pixrange<=disp_pixs.max() )
            if ( stellar.header_kws['gain']==None ):
                gain = 1.0
            else:
                gain = header[stellar.header_kws['gain']]
            image_hdu.close()

            # Cut out a subarray containing the spectrum data:
            if stellar.disp_axis==0:
                subarray = darray[disp_ixs,:][:,crossdisp_ixs]
            else:
                subarray = darray[:,disp_ixs][crossdisp_ixs,:]
            nbad = subarray.mask.sum()
            ntot = subarray.mask.size
            frac_bad = float( nbad )/ntot
            if frac_bad>0.2:
                print '\nToo many bad pixels in:\n{0}\n(skipping)\n'\
                      .format( image_filepath )
                stellar.goodbad[j] = 0
                continue

            # The pixels along the dispersion and cross-dispersion
            # axes of the subarray:
            crossdisp_pixs = crossdisp_pixrange[crossdisp_ixs]
            disp_pixs = disp_pixrange[disp_ixs]
            npix_disp = len( disp_pixs )
            
            # Arrays to hold the spectral extraction output:
            apflux = np.zeros( npix_disp )
            nappixs = np.zeros( npix_disp )
            skyppix = np.zeros( npix_disp )

            # Loop over each pixel column along the dispersion axis
            # to extract the spectrum:
            for i in range( npix_disp ):

                crossdisp_central_pix = trarray[i]
                if stellar.disp_axis==0:
                    crossdisp_row = subarray[i,:]
                else:
                    crossdisp_row = subarray[:,i]

                # Before proceeding, make sure the fitted trace
                # center is actually located within the defined
                # cross-dispersion limits; if it's not, it suggests
                # that something went wrong and this is probably
                # an unusable frame, so skip it:
                if ( crossdisp_central_pix<crossdisp_pixs.min() )+\
                   ( crossdisp_central_pix>crossdisp_pixs.max() ):
                    print '\nProblem with trace fit for dispersion column {0} of {1}'\
                          .format( disp_pixs[i], os.path.basename( image_filepath ) )
                    print '(may be worth inspecting the corresponding image)\n'
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
                nfracpix_u = ( crossdisp_central_pix + stellar.spectral_ap_radius ) - ( u_full + 1 )
                nfracpix_l = l_full - ( crossdisp_central_pix - stellar.spectral_ap_radius )

                # Simple sanity check:
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
                apflux[i] = np.sum( darray_full ) \
                            + darray_fraclower \
                            + darray_fracupper \
                            - sky
                nappixs[i] = npix_full + nfracpix_u + nfracpix_l
                skyppix[i] = sky/float( nappixs[i] )

            # Save the spectra for the current star in the current
            # image to a fits table:
            data = np.zeros( npix_disp, dtype=[ ( 'disp_pixs', np.int64 ), \
                                                ( 'apflux', np.float64 ), \
                                                ( 'nappixs', np.float64 ), \
                                                ( 'skyppix', np.float64 ) ] )
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

            data['disp_pixs'] = disp_pixs
            data['apflux'] = apflux*gain
            data['nappixs'] = nappixs
            data['skyppix'] = skyppix*gain
            if os.path.isfile( ospec_filepath ):
                os.remove( ospec_filepath )
            fits = fitsio.FITS( ospec_filepath, 'rw' )
            trace_filename = os.path.basename( trace_files_k[i] )
            header = { 'IMAGE':image_filename, 'TRACE':trace_filename, 'FWHM':fwhm_ki }
            fits.write( data, header=header )
            fits.close()
            print ' ... saved {0}'.format( ospec_filepath )
            science_spectra_list_ofile_k.write( '{0}\n'.format( ospec_ext ) )
                
    # Save the current list of spectra files:
    science_spectra_list_ofile_k.close()
    print '\nSaved list of spectra in:\n{0}'.format( science_spectra_list_ofilepath_k )

    return None

    
def calibrate_wavelength_scale( stellar, poly_order=1, make_plots=False ):
    """
    Use the pre-determined dispersion pixel coordinates of the fiducial
    lines from the master arc frame to fit a polynomial mapping from the
    dispersion axis to the wavelength scale for each star.
    """

    plt.ioff()
    
    marc_hdu = fitsio.FITS( stellar.wcal_kws['marc_file'], 'r' )
    marc = marc_hdu[1].read()
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
        print '  Max wavelength discrepancy = {0} (units of input fiducial wavelengths; not pixels)'\
              .format( np.abs( residuals ).max() )
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


