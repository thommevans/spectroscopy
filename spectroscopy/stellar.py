import pdb, os, sys, shutil
import pyfits
import numpy as np
import stellar_routines

class stellar():
    """
    """
    def __init__( self ):
        """
        """

        self.star_names = None

        # Image location keywords:
        self.ddir = None
        self.adir = None
        self.science_images_list = ''
        self.badpix_static = ''
        self.science_traces_list = ''
        # Option to provide list of names for output spectra:
        self.science_spectra_filenames = [] 

        # Keywords for some basic variables contained
        # in the image headers:
        self.header_kws = {}
        self.header_kws['gain'] = None

        # Trace fitting keywords:
        self.tracefit_kwargs = { 'method':'linear_interpolation', \
                                 'binw_disp':50 }
        self.crossdisp_bounds = []

        # Spectra extraction keywords:
        self.spectral_ap_width = None
        self.sky_inner_radius = None
        self.sky_band_width = None
        self.sky_method = 'linear_interpolation'

        # Record of frames that have been flagged as good or bad:
        self.goodbad = None
        
        # The number of extensions per fits file containing science data:
        self.next = 1

        return None

    def identify_bad_pixels( self ):
        """
        """
        stellar_routines.identify_bad_pixels( self )
        return None

    def fit_traces( self, make_plots=False ):
        """
        """
        stellar_routines.fit_traces( self, make_plots=make_plots )
        return None

    def extract_spectra( self ):
        """
        """
        stellar_routines.extract_spectra( self )
        return None

    def calibrate_wavelength_scale( self, poly_order=1, make_plots=False ):
        """
        """
        stellar_routines.calibrate_wavelength_scale( self, poly_order=poly_order, make_plots=make_plots )
        return None
