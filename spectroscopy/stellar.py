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

        # Image location keywords:
        self.ddir = None
        self.adir = None
        self.science_images_list = ''
        self.badpix_static = ''
        self.science_traces_list = ''

        # Header keywords:
        self.header_kws = {}
        self.header_kws['JD'] = ''
        self.header_kws['EXPTIME_SECS'] = ''
        self.header_kws['GAIN'] = ''

        # Trace fitting keywords:
        self.tracefit_kwargs = { 'method':'linear_interpolation', \
                                 'binw_disp':50 }
        self.crossdisp_bounds = []

        # Spectra extraction keywords:
        self.spectral_ap_width = None
        self.sky_inner_radius = None
        self.sky_band_width = None
        self.sky_method = 'linear_interpolation'
        
        return None

    def identify_badpixels( self ):
        """
        """
        stellar_routines.identify_badpixels( self )
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

    def calibrate_wavelength_scale( self, make_plots=False ):
        """
        """
        stellar_routines.calibrate_wavelength_scale( self, make_plots=make_plots )
        return None
