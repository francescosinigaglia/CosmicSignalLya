from astropy.io import fits
import input_params as inpars

nreal = inpars.nreal
version = inpars.version

# Input filenames                                                                                                                                               
output_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/' + version + '/skewers-%d/' %nreal

hdulm = fits.open(output_dir + 'master_fiducial.fits')
data = hdulm[1].data
c1 = fits.Column(name='RA', array=data['RA'], format='D')
c2 = fits.Column(name='DEC', array=data['DEC'], format='D')
c3 = fits.Column(name='Z_noRSD', array=data['Z_noRSD'], format='D')
c4 = fits.Column(name='Z', array=data['Z'], format='D')
c5 = fits.Column(name='MOCKID', array=data['MOCKID'], format='K')
c6 = fits.Column(name='THING_ID', array=data['MOCKID'], format='K')
c7 = fits.Column(name='PLATE', array=data['MOCKID'], format='K')
c8 = fits.Column(name='MJD', array=data['MOCKID'], format='K')
c9 = fits.Column(name='FIBERID', array=data['MOCKID'], format='K')
hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9], name='CATALOG')

hdulm[1] = hdu
hdulm.writeto(output_dir + 'master.fits', overwrite=True)
