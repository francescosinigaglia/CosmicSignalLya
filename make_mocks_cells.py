import numpy as np
from numba import njit, prange
import os
import time
import bias_parameters_lya as biaspars
import input_params as inpars

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

# I/O
do_cic_real_space = False
do_cic_redshift_space = True
do_tweb_redshift_space = True

# General parameters

# I/O files    
nreal = inpars.nreal
version = inpars.version

# Input filenames                                                                                                                                               
input_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_boxes/v0/box%d/' %nreal
output_aux_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/' + version + '/skewers-%d/aux/' %nreal

dm_filename =  input_dir + 'dmLCOM0.314OL0.686G1800V10000.0.dat'  # Real space                               
tweb_filename = input_dir + 'Tweb_OM0.307OL0.693G1800V10000.0lth0.000.dat'  # Real space         

vx_filename = input_dir + 'VEULxOM0.314OL0.686G1800V10000.0.dat' 
vy_filename = input_dir + 'VEULyOM0.314OL0.686G1800V10000.0.dat'
vz_filename = input_dir + 'VEULzOM0.314OL0.686G1800V10000.0.dat'

posx_filename = input_dir + 'partposxOM0.314OL0.686G1800V10000.0.dat'
posy_filename = input_dir + 'partposyOM0.314OL0.686G1800V10000.0.dat'
posz_filename = input_dir + 'partposzOM0.314OL0.686G1800V10000.0.dat'

zarr_filename = 'zarr.DAT'
darr_filename = 'dcomOM0.314OL0.686.DAT'

dmout_filename_rspace = output_aux_dir + 'dm_lc_rspace.dat'  # Output filename    
dmout_filename_zspace = output_aux_dir + 'dm_lc_zspace_cells.dat'  # Output filename

tweb_zspace_filename = output_aux_dir + 'tweb_zspace.dat'
fluxout_filename = output_aux_dir + 'flux_zspace.dat'

# General parameters

lbox = 10000.
ngrid = 1800 

zmin = 1.77
zmax = 3.8

# Observer positions
obspos = [5000.,5000.,5000.]

# NL FGPA parameters                                                                                                                                             
zarrbias = np.array(biaspars.zarrbias)
aapars = np.array(biaspars.aapars)
alphapars = np.array(biaspars.alphapars)
delta1pars = np.array(biaspars.delta1pars)
delta2pars = np.array(biaspars.delta2pars)

negbinpars1 = np.array(biaspars.negbinpars1)
negbinpars2 = np.array(biaspars.negbinpars2)
negbinpars3 = np.array(biaspars.negbinpars3)

# RSD parameters                                                                                                                                                 
bbpars = np.array(biaspars.bbpars)
betapars = np.array(biaspars.betapars)
bvpars = 0.35 * np.array(biaspars.bvpars) # 0.35 fiducial

# Cosmological parameters (Abacus)
h = 0.6736
H0 = 100
Om = 0.314
Orad = 0.
Ok = 0.
N_eff = 3.046
w_eos = -1
Ol = 1-Om-Ok-Orad

# Random seed for stochasticity reproducibility
np.random.seed(123456)

# **********************************************
# **********************************************
# **********************************************
def fftr2c(arr):
    arr = np.fft.rfftn(arr, norm='ortho')

    return arr

def fftc2r(arr):
    arr = np.fft.irfftn(arr, norm='ortho')

    return arr

# **********************************************
def measure_spectrum(signal):

    nbin = round(ngrid/2)
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))
    dk = kmax/nbin  # Bin width

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_power(fsignal, nbin, kmax, dk, kmode, power, nmode)
    
    return kmode[1:], power[1:]

# **********************************************                                                                                    
def cross_spectrum(signal1, signal2):

    nbin = round(ngrid/2)

    fsignal1 = np.fft.fftn(signal1) #np.fft.fftn(signal)                                             
    fsignal2 = np.fft.fftn(signal2) #np.fft.fftn(signal)                                             

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))            
    dk = kmax/nbin  # Bin width                                                                                                    

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_cross_power(fsignal1, fsignal2, nbin, kmax, dk, kmode, power, nmode)

    return kmode[1:], power[1:]

# **********************************************                                                                                        
def compute_cross_correlation_coefficient(cross, power1,power2):
    ck = cross/(np.sqrt(power1*power2))
    return ck

# **********************************************                                                                                         
@njit(parallel=True, cache=True)
def get_power(fsignal, Nbin, kmax, dk, kmode, power, nmode):
    
    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl = fsignal.real[i,j,k]
                    bkl = fsignal.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl*akl+bkl*bkl)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode

# **********************************************                                                                            
@njit(parallel=False, cache=True)
def get_cross_power(fsignal1, fsignal2, Nbin, kmax, dk, kmode, power, nmode):

    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl1 = fsignal1.real[i,j,k]
                    bkl1 = fsignal1.imag[i,j,k]
                    akl2 = fsignal2.real[i,j,k]
                    bkl2 = fsignal2.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl1*akl2+bkl1*bkl2)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode
  
# **********************************************
@njit(parallel=True, cache=True)
def get_cic(posx, posy, posz, weight, lbox, ngrid):

    weight = weight.flatten() + 1.
    
    lcell = lbox / ngrid

    print('Lcell CIC:', lcell)

    delta = np.zeros((ngrid,ngrid,ngrid))

    dummylen = int(ngrid**3)

    for ii in range(dummylen):

        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
        indxc = int(xx/lcell)
        indyc = int(yy/lcell)
        indzc = int(zz/lcell)

        wxc = xx/lcell - indxc
        wyc = yy/lcell - indyc
        wzc = zz/lcell - indzc

        if indxc >= ngrid:
            indxc -= ngrid
        if indyc >= ngrid:
            indyc -= ngrid
        if indzc >= ngrid:
            indzc -= ngrid

        if wxc <=0.5:
            indxl = indxc - 1
            if indxl<0:
                indxl += ngrid
            wxc += 0.5
            wxl = 1. - wxc
        elif wxc >0.5:
            indxl = indxc + 1
            if indxl>=ngrid:
                indxl -= ngrid
            wxl = wxc - 0.5
            wxc = 1 - wxl

        if wyc <=0.5:
            indyl = indyc - 1
            if indyl<0:
                indyl += ngrid
            wyc += 0.5
            wyl = 1. - wyc
        elif wyc >0.5:
            indyl = indyc + 1
            if indyl>=ngrid:
                indyl -= ngrid
            wyl = wyc - 0.5
            wyc = 1 - wyl

        if wzc <=0.5:
            indzl = indzc - 1
            if indzl<0:
                indzl += ngrid
            wzc += 0.5
            wzl = 1. - wzc
        elif wzc >0.5:
            indzl = indzc + 1
            if indzl>=ngrid:
                indzl -= ngrid
            wzl = wzc - 0.5
            wzc = 1 - wzl

        ww = weight[ii]
        
        delta[indxc,indyc,indzc] += ww * wxc * wyc * wzc
        delta[indxl,indyc,indzc] += ww * wxl * wyc * wzc
        delta[indxc,indyl,indzc] += ww * wxc * wyl * wzc
        delta[indxc,indyc,indzl] += ww * wxc * wyc * wzl
        delta[indxl,indyl,indzc] += ww * wxl * wyl * wzc
        delta[indxc,indyl,indzl] += ww * wxc * wyl * wzl
        delta[indxl,indyc,indzl] += ww * wxl * wyc * wzl
        delta[indxl,indyl,indzl] += ww * wxl * wyl * wzl
        

    #delta = delta/np.mean(delta) - 1.
    
    return delta

# **********************************************
# Real to redshift space mapping
@njit(parallel=True, fastmath=True, cache=True)
def real_to_redshift_space(delta, tweb, posx, posy, posz, vx, vy, vz, ngrid, lbox, zarr, darr, zmin, zmax, xobs, yobs, zobs, bvpars, bbpars, betapars):
    
    HH = 100.
    lcell = lbox/ ngrid

    print('Lcell: ', lcell)

    posxnew = np.zeros(ngrid*ngrid*ngrid)
    posynew = np.zeros(ngrid*ngrid*ngrid)
    posznew = np.zeros(ngrid*ngrid*ngrid)

    # Parallelize the outer loop
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                ind3d = kk + ngrid*(jj + ngrid*ii)

                # Initialize positions at the centre of the cell
                xtmp = (ii+0.5) * lcell
                ytmp = (jj+0.5)	* lcell
                ztmp = (kk+0.5)	* lcell

                # Compute redshift
                dtmp = np.sqrt((xtmp-xobs)**2 + (ytmp-yobs)**2 + (ztmp-zobs)**2)

                if dtmp<= lbox/2:
                    redshift = np.interp(dtmp, darr, zarr)
                    ascale = 1./(1.+redshift)

                    if redshift>zmin and redshift<zmax:

                        if tweb[ii,jj,kk]==1:
                            bv = bvpars[0]
                            bb = bbpars[0]
                            beta = betapars[0]

                        elif tweb[ii,jj,kk]==2:
                            bv = bvpars[1]
                            bb = bbpars[1]
                            beta = betapars[1]

                        elif tweb[ii,jj,kk]==3:
                            bv = bvpars[2]
                            bb = bbpars[2]
                            beta = betapars[2]

                        elif tweb[ii,jj,kk]==4:
                            bv = bvpars[3]
                            bb = bbpars[3]
                            beta = betapars[3]

                        sigma = bb*(1. + delta[ii,jj,kk])**beta

                        vxrand = np.random.normal(0,sigma)
                        vyrand = np.random.normal(0,sigma)
                        vzrand = np.random.normal(0,sigma)

                        vxtmp = vx[ii,jj,kk] #trilininterp(xtmp, ytmp, ztmp, vx, lbox, ngrid)
                        vytmp = vy[ii,jj,kk] #trilininterp(xtmp, ytmp, ztmp, vy, lbox, ngrid)
                        vztmp = vz[ii,jj,kk] #trilininterp(xtmp, ytmp, ztmp, vz, lbox, ngrid)

                        #print(vxtmp, vxrand, ascale * HH, redshift)
                        #print('')

                        vxtmp += vxrand
                        vytmp += vyrand
                        vztmp += vzrand

                        # Go from cartesian to sky coordinates
                        vxtmp, vytmp, vztmp = project_vector_los(xtmp, ytmp, ztmp, vxtmp, vytmp, vztmp, zarr, darr, xobs, yobs, zobs)

                        xtmp = xtmp + bv * vxtmp / (ascale * HH)
                        ytmp = ytmp + bv * vytmp / (ascale * HH)
                        ztmp = ztmp + bv * vztmp / (ascale * HH)

                        # Impose boundary conditions
                        if xtmp<0.:
                            xtmp += lbox
                        elif xtmp>lbox:
                            xtmp -= lbox

                        if ytmp<0.:
                            ytmp += lbox
                        elif ytmp>lbox:
                            ytmp -= lbox

                        if ztmp<0.:
                            ztmp += lbox
                        elif ztmp>lbox:
                            ztmp -= lbox

                # Allocate new positions (outside if condition)
                posxnew[ind3d] = xtmp
                posynew[ind3d] = ytmp
                posznew[ind3d] = ztmp

    return posxnew, posynew, posznew

# **********************************************
@njit(parallel=True, cache=True)
def divide_by_k2(delta,ngrid, lbox):
    for ii in prange(ngrid):
        for jj in prange(ngrid):
            for kk in prange(round(ngrid/2.)): 
                k2 = k_squared(lbox,ngrid,ii,jj,kk) 
                if k2>0.:
                    delta[ii,jj,kk] /= -k_squared(lbox,ngrid,ii,jj,kk) 

    return delta

# **********************************************
@njit(cache=True)
def k_squared(lbox,ngrid,ii,jj,kk):
    
      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)
      
      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)
      
      #if kk <= nc/2:
      kz = kfac*kk
      #else:
      #  kz = -kfac*np.float64(nc-k)
      
      k2 = kx**2+ky**2+kz**2

      return k2

@njit(cache=True)
def k_squared_nohermite(lbox,ngrid,ii,jj,kk):

      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)

      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)

      if kk <= ngrid/2:
          kz = kfac*kk
      else:
          kz = -kfac*(ngrid-kk)                                                                                                           

      k2 = kx**2+ky**2+kz**2

      return k2

# **********************************************
@njit(parallel=True, fastmath=True, cache=True)
def extract_skewers(posx, posy, posz, zmin, zmax, zarr, darr, hrbinw, flux, ngrid, lbox):

    lcell = lbox / ngrid

    # Initialize a matrix of skewers
    # (NxM), where N=number of QSO, M=number of bins per spectrum

    # Let's first determine maximum and minimum distance, given the redshift range
    dmax = np.interp(zmax, zarr, darr)
    dmin = np.interp(zmin, zarr, darr)

    # Determine the number of bins in the spectra
    nbins = int((dmax-dmin) / hrbinw)

    # Allocate the matrix
    skmat = np.zeros((len(posx), nbins))

    #print(skmat.shape)

    # Set up a template of distances
    dtemplate = np.linspace(dmin, dmax, nbins+1)
    dtemplate = 0.5*(dtemplate[1:] + dtemplate[:-1])

    ztemplate = np.interp(dtemplate, darr, zarr)

    for ii in prange(len(posx)):

        ra, dec, zz = cartesian_to_sky(posx[ii],posy[ii],posz[ii], zarr, darr)

        for jj in range(len(dtemplate)):

            ztemp = ztemplate[jj]
            posxlya, posylya, poszlya = sky_to_cartesian(ra,dec,ztemp, zarr, darr)

            indx = int(posxlya/lcell)
            indy = int(posylya/lcell)
            indz = int(poszlya/lcell)

            skmat[ii,jj] = flux[indx,indy,indz] 

    return skmat

# **********************************************
@njit(parallel=False, fastmath=True, cache=True)
def sky_to_cartesian(ra,dec,zz, zarr, darr, xobs, yobs, zobs):

    dd = np.interp(zz, zarr, darr)

    ra = ra / 180. * np.pi
    dec = dec / 180. * np.pi

    posx = dd * np.cos(dec) * np.cos(ra)
    posy = dd * np.cos(dec) * np.sin(ra)
    posz = dd * np.sin(dec)

    posx += xobs
    posy += yobs
    posz += zobs

    return posx, posy, posz

# **********************************************

@njit(parallel=False, fastmath=True, cache=True)
def cartesian_to_sky(posx, posy, posz, zarr, darr, xobs, yobs, zobs):

    posx -= xobs
    posy -= yobs
    posz -= zobs
    
    dd = np.sqrt(posx**2 + posy**2 + posz**2)
    zz = np.interp(dd, darr, zarr)
    #dec = np.arccos(posz / dd) / np.pi * 180.

    #if np.sin(dec)!=0:
    #    ra = np.arccos( posx / (dd * np.sin(dec))) / np.pi * 180.
    #else:
    #    ra = 0.
    ss = np.hypot(posx, posy)
    ra = np.arctan2(posy, posx) / np.pi * 180.
    #dec = np.arcsin(posz/dd) / np.pi * 180.
    dec = np.arctan2(posz, ss) / np.pi * 180.

    # convert to degrees
    #lon = da.rad2deg(lon)
    #lat = da.rad2deg(lat)

    #ra = np.mod(ra-360., 360.)

    return ra, dec, zz

# **********************************************

@njit(parallel=False, fastmath=True, cache=True)
def project_vector_los(posx, posy, posz, vecx, vecy, vecz, zarr, darr, xobs, yobs, zobs):

    # Determine the line of sight angles ra0 and dec0
    ra0, dec0, zz0 = cartesian_to_sky(posx, posy, posz, zarr, darr, xobs, yobs, zobs)

    ra0 = ra0 / 180. * np.pi
    dec0 = dec0 / 180. * np.pi
    
    # Find the l.o.s. unit vector
    versx = np.cos(dec0) * np.cos(ra0)
    versy = np.cos(dec0) * np.sin(ra0)
    versz = np.sin(dec0)
    
    # Project the velocity vector along the l.o.s. direction
    norm = vecx*versx + vecy*versy + vecz*versz
    
    vecx = norm * versx
    vecy = norm * versy
    vecz = norm * versz

    return vecx, vecy, vecz

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def trilininterp(xx, yy, zz, arrin, lbox, ngrid):

    lcell = lbox/ngrid

    indxc = int(xx/lcell)
    indyc = int(yy/lcell)
    indzc = int(zz/lcell)

    wxc = xx/lcell - indxc
    wyc = yy/lcell - indyc
    wzc = zz/lcell - indzc

    if wxc <=0.5:
        indxl = indxc - 1
        if indxl<0:
            indxl += ngrid
        wxc += 0.5
        wxl = 1 - wxc
    elif wxc >0.5:
        indxl = indxc + 1
        if indxl>=ngrid:
            indxl -= ngrid
        wxl = wxc - 0.5
        wxc = 1 - wxl

    if wyc <=0.5:
        indyl = indyc - 1
        if indyl<0:
            indyl += ngrid
        wyc += 0.5
        wyl = 1 - wyc
    elif wyc >0.5:
        indyl = indyc + 1
        if indyl>=ngrid:
            indyl -= ngrid
        wyl = wyc - 0.5
        wyc = 1 - wyl

    if wzc <=0.5:
        indzl = indzc - 1
        if indzl<0:
            indzl += ngrid
        wzc += 0.5
        wzl = 1 - wzc
    elif wzc >0.5:
        indzl = indzc + 1
        if indzl>=0:
            indzl -= ngrid
        wzl = wzc - 0.5
        wzc = 1 - wzl

    wtot = wxc*wyc*wzc + wxl*wyc*wzc + wxc*wyl*wzc + wxc*wyc*wzl + wxl*wyl*wzc + wxl*wyc*wzl + wxc*wyl*wzl + wxl*wyl*wzl

    out = 0.

    out += arrin[indxc,indyc,indzc] * wxc*wyc*wzc
    out += arrin[indxl,indyc,indzc] * wxl*wyc*wzc
    out += arrin[indxc,indyl,indzc] * wxc*wyl*wzc
    out += arrin[indxc,indyc,indzl] * wxc*wyc*wzl
    out += arrin[indxl,indyl,indzc] * wxl*wyl*wzc
    out += arrin[indxc,indyl,indzl] * wxc*wyl*wzl
    out += arrin[indxl,indyc,indzl] * wxl*wyc*wzl
    out += arrin[indxl,indyl,indzl] * wxl*wyl*wzl

    return out

# **********************************************
# **********************************************
# **********************************************                                                                                                                
# Compute overdensity using only the cells which are really used
@njit(parallel=True, fastmath=True, cache=True)
def calc_overdens_with_mean(delta, lbox, ngrid, zmin, zmax, zarr, darr):

    HH = 100.
    lcell = lbox/ ngrid

    posxnew = np.zeros(ngrid*ngrid*ngrid)
    posynew = np.zeros(ngrid*ngrid*ngrid)
    posznew = np.zeros(ngrid*ngrid*ngrid)

    cnt = 0
    val = 0.

    # Parallelize the outer loop                 
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                ind3d = kk + ngrid*(jj + ngrid*ii)

                # Initialize positions at the centre of the cell    
                xtmp = (ii+0.5) * lcell
                ytmp = (jj+0.5) * lcell
                ztmp = (kk+0.5) * lcell

		# Compute redshift   
                dtmp = np.sqrt((xtmp-xobs)**2 + (ytmp-xobs)**2 + (ztmp-zobs)**2)

                if dtmp<= lbox/2:
                    redshift = np.interp(dtmp, darr, zarr)

                    if redshift>zmin and redshift<zmax:
                        val += delta[ii,jj,kk]
                        cnt += 1

    mean = val / cnt

    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                ind3d = kk + ngrid*(jj + ngrid*ii)

                # Initialize positions at the centre of the cell                                                                       
                xtmp = (ii+0.5) * lcell
                ytmp = (jj+0.5) * lcell
                ztmp = (kk+0.5) * lcell

		# Compute redshift                                                                                                                              
                dtmp = np.sqrt((xtmp-xobs)**2 + (ytmp-xobs)**2 + (ztmp-zobs)**2)

                if dtmp<= lbox/2:
                    redshift = np.interp(dtmp, darr, zarr)

                    if redshift>zmin and redshift<zmax:
                        delta[ii,jj,kk] = delta[ii,jj,kk]/mean - 1.
                    else:
                        delta[ii,jj,kk] = 0.

                else:
                    delta[ii,jj,kk] = 0.

    return delta

# **********************************************
# ********************************************** 
@njit(parallel=True, fastmath=True, cache=True)
def apply_nl_fgpa(delta, tweb, ngrid, lbox, zzarrbias, aapars, alphapars, delta1pars, delta2pars, negbinpars1, negbinpars2, negbinpars3, zmin, zmax):

    lcell = lbox/ ngrid

    # Allocate flux field (may be replaced with delta if too memory consuming)  
    flux = np.zeros((ngrid,ngrid,ngrid))

    # Parallelize the outer loop                         
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                # Initialize positions at the centre of the cel                      
                xtmp = (ii+0.5) * lcell
                ytmp = (jj+0.5) * lcell
                ztmp = (kk+0.5) * lcell

                # Compute redshift      
                dtmp = np.sqrt((xtmp-lbox/2.)**2 + (ytmp-lbox/2.)**2 + (ztmp-lbox/2.)**2)
                redshift = np.interp(dtmp, darr, zarr)

                if dtmp<lbox/2. and redshift>zmin and redshift<zmax:

                    indd = int(tweb[ii,jj,kk] - 1)

                    aaparstmp = aapars[indd,:]
                    alphaparstmp = alphapars[indd,:]
                    delta1parstmp = delta1pars[indd,:]
                    delta2parstmp = delta2pars[indd,:]

                    aa = np.interp(redshift, zzarrbias, aaparstmp)
                    alpha = np.interp(redshift, zzarrbias, alphaparstmp)
                    delta1 = np.interp(redshift, zzarrbias, delta1parstmp)
                    delta2 = np.interp(redshift, zzarrbias, delta2parstmp)

                    deltatmp = delta[ii,jj,kk]

                    #if tweb[ii,jj,kk]==4:
                    #    negbin1 = np.interp(redshift, zzarrbias, negbinpars1)
                    #    negbin2 = np.interp(redshift, zzarrbias, negbinpars2)
                    #    negbin3 = np.interp(redshift, zzarrbias, negbinpars3)

                    #    deltatmp += negbin1*negative_binomial(negbin2, negbin3)

                    #    if deltatmp<-1:
                    #        deltatmp = -1.

                    tau = aa * (1.+deltatmp)**alpha #* np.exp(-deltatmp/delta1) * np.exp(deltatmp/delta2)

                    flux[ii,jj,kk] = np.exp(-tau)

                else:
                    flux[ii,jj,kk] = 0.5

    return flux

# **********************************************      
@njit(fastmath=True, cache=True)
def negative_binomial(n, p):
    if n>0:
        if p > 0. and p < 1.:
            gfunc = np.random.gamma(n, (1. - p) / p)
            Y = np.random.poisson(gfunc)

    else:
        Y = 0

    return Y

# **********************************************
def run_cwc():
    
    #os.system('cp density_dm_zspace.dat')
    os.system('cp %s .' %dmout_filename_zspace)
    os.system('cp webon/webonx .')
    os.system('cp webon/*par .')
    #os.system('cp %s .' %dm_filename)
    os.system('./webonx')
    os.system('mv Tweb_* %s' %tweb_zspace_filename)
    os.system('rm %s .' %dmout_filename_zspace)

# **********************************************
# **********************************************
# **********************************************
print('--------------------------------')
print('RSD code for lightcone DM fields')
print('--------------------------------')

ti = time.time()

lcell = lbox/ngrid

xobs = obspos[0]
yobs = obspos[1]
zobs = obspos[2]

print('')
print('Reading input ...')

if do_cic_real_space ==False:
    delta = np.fromfile(dm_filename, dtype=np.float32)  # In real space
    delta = np.reshape(delta, (ngrid,ngrid,ngrid))

tweb = np.fromfile(tweb_filename, dtype=np.float32)  # In real space      

# Positions
#posx = np.fromfile(posx_filename, dtype=np.float32)   
#posy = np.fromfile(posy_filename, dtype=np.float32) 
#posz = np.fromfile(posz_filename, dtype=np.float32)

#posx += xobs
#posy += yobs
#posz += zobs

# Now they are velocity vectors
vx = np.fromfile(vx_filename, dtype=np.float32)  
vy = np.fromfile(vy_filename, dtype=np.float32) 
vz = np.fromfile(vz_filename, dtype=np.float32) 

# Reshape arrays from 1D to 3D --> reshape only arrays which have mesh structure, e.g. NOT positions
#delta = np.reshape(delta, (ngrid,ngrid,ngrid))
tweb = np.reshape(tweb, (ngrid,ngrid,ngrid))   

#posx = np.reshape(posx, (ngrid,ngrid,ngrid))
#posy = np.reshape(posy, (ngrid,ngrid,ngrid))
#posz = np.reshape(posz, (ngrid,ngrid,ngrid))

vx = np.reshape(vx, (ngrid,ngrid,ngrid))
vy = np.reshape(vy, (ngrid,ngrid,ngrid))
vz = np.reshape(vz, (ngrid,ngrid,ngrid))

# Read the tabulated redshift and comoving distance arrays
zarr = np.fromfile(zarr_filename, dtype=np.float32) 
darr = np.fromfile(darr_filename, dtype=np.float32) 

print('... done!')
print('')

if do_cic_real_space == True:
    print('Interpolating particles to mesh (real space) ...')
    delta = get_cic(posx, posy, posz, delta, lbox, ngrid)
    delta = calc_overdens_with_mean(delta, lbox, ngrid, zmin, zmax)
    print('... done!')
    print('')

    # Write real space DM field to file
    delta.astype('float32').tofile(dmout_filename_rspace)

print('Diagnostics DM field real space (min, max, mean)', np.amin(delta), np.amax(delta), np.mean(delta))
print('')

print('Mapping DM field from real to redshift space ...')
# Now the containers become redshift space positions
posx = 0.
posy = 0.
posz = 0.

if do_cic_redshift_space == True:
    posxnew, posynew, posznew = real_to_redshift_space(delta, tweb, posx, posy, posz, vx, vy, vz, ngrid,  lbox, zarr, darr, zmin, zmax, xobs, yobs, zobs, bvpars, bbpars, betapars)
print('... done!')
print('')

print('POSZNEW: ', posznew[:10])

if do_cic_redshift_space == True:
    # Do CIC, delta becomes the redshift space density field
    print('Interpolating particles to mesh (redshift space) ...')
    delta = get_cic(posxnew, posynew, posznew, delta, lbox, ngrid)
    delta = calc_overdens_with_mean(delta, lbox, ngrid, zmin, zmax, zarr, darr)
    print('... done!')

    # Write redshift space DM field to file
    delta.astype('float32').tofile(dmout_filename_zspace)

else:
    delta = np.fromfile(dmout_filename_zspace, dtype=np.float32) # TEMPORARY
    delta = np.reshape(delta, (ngrid,ngrid,ngrid))

print('Diagnostics DM field redshift space (min, max, mean)', np.amin(delta), np.amax(delta), np.mean(delta))
print('')

#delta = np.reshape(delta, (ngrid,ngrid,ngrid))
# Deallocate all the arrays (but delta) if needed
#del tweb
if do_cic_redshift_space ==True:
    del posxnew, posynew, posznew, posx, posy, posz

"""
# Run T-web classification
print('Doing T-web classification ...')
if do_tweb_redshift_space==True:
    run_cwc()
    print('... done!')
    print('')
"""

# Read the T-web classification (in zspace)
#delta = np.fromfile('density_dm_zspace.dat', dtype=np.float32) # TEMPORARY
#delta = np.reshape(delta, (ngrid,ngrid,ngrid)) # TEMPORARY
#tweb = np.fromfile(tweb_zspace_filename ,dtype=np.float32)
#tweb = np.reshape(tweb, (ngrid,ngrid,ngrid))

print('Applying NL FGPA ...')
# Predict the LyA flux via NL FGPA
flux = apply_nl_fgpa(delta, tweb, ngrid, lbox, zarrbias, aapars, alphapars, delta1pars, delta2pars, negbinpars1, negbinpars2, negbinpars3, zmin, zmax)
flux.astype('float32').tofile(fluxout_filename)
print('... done!')
print('')

tf = time.time()
print('Elapsed %s seconds' %str(tf-ti))
