import numpy as np
from numba import njit, prange, int64
import numba as nb
import os
import time
import bias_parameters_qso as pars
import random
import input_params as inpars

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

nreal = inpars.nreal
version = inpars.version

# Input filenames
input_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_boxes/v0/box%d/' %nreal
output_aux_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/' + version + '/skewers-%d/aux/' %nreal

dm_filename =  input_dir + 'ridged_dmLCOM0.314OL0.686G1800V10000.0.dat'
tweb_filename = input_dir + 'Tweb_OM0.307OL0.693G1800V10000.0lth0.000.dat'
twebdelta_filename = input_dir + 'TwebDelta_OM0.307OL0.693G1800V10000.0lth0.000.dat'

vx_filename = input_dir + 'VEULxOM0.314OL0.686G1800V10000.0.dat'
vy_filename = input_dir + 'VEULyOM0.314OL0.686G1800V10000.0.dat'
vz_filename = input_dir + 'VEULzOM0.314OL0.686G1800V10000.0.dat'

posx_filename = input_dir + 'partposxOM0.314OL0.686G1800V10000.0.dat'
posy_filename =	input_dir + 'partposyOM0.314OL0.686G1800V10000.0.dat'
posz_filename = input_dir + 'partposzOM0.314OL0.686G1800V10000.0.dat'

zarr_filename = 'zarr.DAT'
darr_filename = 'dcomOM0.314OL0.686.DAT'

# General parameters

lbox = 10000.
ngrid = 1800 

zmin = 1.65 #1.77
zmax = 3.8

# Observer positions
obspos = [5000., 5000., 5000.]

# Bias parameters
zzarrbias = np.array(pars.zzarrbias) 

meandensarr = 0.4*np.array(pars.meandensarr)

nmean_arr = np.array(pars.nmean_arr) 
alpha_arr = np.array(pars.alpha_arr) 
beta_arr = np.array(pars.beta_arr)
dth_arr = np.array(pars.dth_arr)
rhoeps_arr = np.array(pars.rhoeps_arr) 
eps_arr = np.array(pars.eps_arr)

# RSD parameters
bb_arr = np.array(pars.bb_arr)
betarsd_arr = np.array(pars.betarsd_arr)
bv_arr = np.array(pars.bv_arr)
gamma_arr = np.array(pars.gamma_arr)

radsearch = 2

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
@njit(parallel=False, cache=True)
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
def get_cic(posx, posy, posz, lbox, ngrid):

    lcell = lbox / ngrid

    delta = np.zeros((ngrid,ngrid,ngrid))

    dummylen = len(posx)

    for ii in prange(dummylen):

        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
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
            if indzl>=ngrid:
                indzl -= ngrid
            wzl = wzc - 0.5
            wzc = 1 - wzl

        delta[indxc,indyc,indzc] += wxc * wyc * wzc
        delta[indxl,indyc,indzc] += wxl * wyc * wzc
        delta[indxc,indyl,indzc] += wxc * wyl * wzc
        delta[indxc,indyc,indzl] += wxc * wyc * wzl
        delta[indxl,indyl,indzc] += wxl * wyl * wzc
        delta[indxc,indyl,indzl] += wxc * wyl * wzl
        delta[indxl,indyc,indzl] += wxl * wyc * wzl
        delta[indxl,indyl,indzl] += wxl * wyl * wzl

    delta = delta/np.mean(delta) - 1.
    
    return delta

# *********************************************
# **********************************************
# Real to redshift space mapping
@njit(parallel=True, fastmath=True, cache=True)
def real_to_redshift_space(delta, tweb, posx, posy, posz, vx, vy, vz, ngrid, lbox, zarr, darr, zmin, zmax, xobs, yobs, zobs, bv_arr, bb_arr, beta_arr, gamma_arr, zzarrbias):
    
    H0 = 100.

    lcell = lbox/ ngrid

    posxnew = np.zeros(len(posx))
    posynew = np.zeros(len(posy))
    posznew = np.zeros(len(posz))

    # Parallelize the outer loop
    for ii in prange(len(posx)):

        # Initialize positions at the centre of the cell
        xtmp = posx[ii]
        ytmp = posy[ii]
        ztmp = posz[ii]

        indx = int(xtmp/lcell)
        indy = int(ytmp/lcell)
        indz = int(ztmp/lcell)

        ind3d = indz+ngrid*(indy+ngrid*indx) 

        # Compute redshift
        dtmp = np.sqrt((xtmp-xobs)**2 + (ytmp-xobs)**2 + (ztmp-zobs)**2)
        redshift = np.interp(dtmp, darr, zarr)
        ascale = 1./(1.+redshift)
        HH = H0 * np.sqrt(Om*(1.+redshift)**3 + Ol)

        indtmp = int(tweb[indx,indy,indz]-1) 

        bv_tmp = bv_arr[:,indtmp]
        bb_tmp = bb_arr[:,indtmp]
        beta_tmp = betarsd_arr[:,indtmp]
        gamma_tmp = betarsd_arr[:,indtmp]

        bv = np.interp(redshift, zzarrbias, bv_tmp)
        bb = np.interp(redshift, zzarrbias, bb_tmp)
        betarsd = np.interp(redshift, zzarrbias, beta_tmp)
        gamma = np.interp(redshift, zzarrbias, gamma_tmp)
        

        sigma = bb*(1. + delta[indx,indy,indz])**betarsd

        vxrand = np.random.normal(0,sigma)
        vyrand = np.random.normal(0,sigma)
        vzrand = np.random.normal(0,sigma)

        vxrand = np.sign(vxrand) * abs(vxrand) ** gamma
        vyrand = np.sign(vyrand) * abs(vyrand) ** gamma
        vzrand = np.sign(vzrand) * abs(vzrand) ** gamma

        vxtmp = trilininterp(xtmp, ytmp, ztmp, vx, lbox, ngrid)
        vytmp = trilininterp(xtmp, ytmp, ztmp, vy, lbox, ngrid)
        vztmp = trilininterp(xtmp, ytmp, ztmp, vz, lbox, ngrid)

        vxtmp += vxrand
        vytmp += vyrand
        vztmp += vzrand

        # Go from cartesian to sky coordinates
        vxtmp, vytmp, vztmp = project_vector_los(xtmp, ytmp, ztmp, vxtmp, vytmp, vztmp, zarr, darr, xobs, yobs, zobs)

        xtmp = xtmp + bv * vxtmp / (ascale * HH)
        ytmp = ytmp + bv * vytmp / (ascale * HH)
        ztmp = ztmp + bv * vztmp / (ascale * HH)

        # Impose boundary conditions
        if xtmp<0:
            xtmp += lbox
        elif xtmp>lbox:
            xtmp -= lbox

        if ytmp<0:
            ytmp += lbox
        elif ytmp>lbox:
            ytmp -= lbox

        if ztmp<0:
            ztmp += lbox
        elif ztmp>lbox:
            ztmp -= lbox

        posxnew[ii] = xtmp
        posynew[ii] = ytmp
        posznew[ii] = ztmp

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

    print(skmat.shape)

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
@njit(parallel=True, cache=True, fastmath=True)
def apply_bias(delta, tweb, zzarrbias, aapars, alphapars, delta1pars, delta2pars, zmin, zmax, xobs, yobs, zobs, lbox, ngrid, darr, zarr):

    lcell = lbox/ ngrid

    # Allocate flux field (may be replaced with delta if too memory consuming)
    ncounts = np.zeros((ngrid,ngrid,ngrid))

    # Parallelize the outer loop
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                # Initialize positions at the centre of the cell
                xtmp = (ii+0.5) * lcell
                ytmp = (jj+0.5) * lcell
                ztmp = (kk+0.5) * lcell

                # Compute redshift
                dtmp = np.sqrt((xtmp-xobs)**2 + (ytmp-yobs)**2 + (ztmp-zobs)**2)
                redshift = np.interp(dtmp, darr, zarr)

                if redshift>=zmin and redshift<zmax:

                    indd = int(tweb[ii,jj,kk] - 1.)

                    aaparstmp = aapars[indd,:]
                    alphaparstmp = alphapars[indd,:]
                    delta1parstmp = delta1pars[indd,:]
                    delta2parstmp = delta2pars[indd,:]

                    aa = np.interp(redshift, zzarrbias, aaparstmp)
                    alpha = np.interp(redshift, zzarrbias, alphaparstmp)
                    delta1 = np.interp(redshift, zzarrbias, delta1parstmp)
                    delta2 = np.interp(redshift, zzarrbias, delta2parstmp)

                    counttmp = (1.+delta[ii,jj,kk])**alpha #* np.exp(-delta[ii,jj,kk]/delta1) * np.exp(delta[ii,jj,kk]/delta2)

                    ncounts[ii,jj,kk] = np.round(counttmp)

                else:
                    ncounts[ii,jj,kk] = 0.

    ncounts = ncounts.astype('int')
                
    return ncounts

# **********************************************
# Compute bias model
@njit(parallel=True, cache=True, fastmath=True)
#def biasmodel_exp(ngrid, lbox, delta, tweb, dweb, zarr, darr, meandensarr, nmean11, alpha11,beta11,rhoeps11,eps11,nmean12,alpha12,beta12,rhoeps12,eps12,nmean13,alpha13,beta13,rhoeps13,eps13,nmean14,alpha14,beta14,rhoeps14,eps14,nmean21, alpha21,beta21,rhoeps21,eps21,nmean22,alpha22,beta22,rhoeps22,eps22,nmean23,alpha23,beta23,rhoeps23,eps23,nmean24,alpha24,beta24,rhoeps24,eps24,nmean31, alpha31,beta31,rhoeps31,eps31,nmean32,alpha32,beta32,rhoeps32,eps32,nmean33,alpha33,beta33,rhoeps33,eps33,nmean34,alpha34,beta34,rhoeps34,eps34,nmean41, alpha41,beta41,rhoeps41,eps41,nmean42,alpha42,beta42,rhoeps42,eps42,nmean43,alpha43,beta43,rhoeps43,eps43,nmean44,alpha44,beta44,rhoeps44,eps44,dth11,dth12,dth13,dth14,dth21,dth22,dth23,dth24,dth31,dth32,dth33,dth34,dth41,dth42,dth43,dth44,xobs,yobs,zobs, zmin,zmax):
def biasmodel_exp(ngrid, lbox, delta, tweb, dweb, zarr, darr, meandensarr, nmean_arr, alpha_arr, beta_arr, dth_arr, rhoeps_arr, eps_arr, xobs,yobs,zobs, zmin,zmax):
     
    lcell = lbox/ ngrid

    # Allocate tracer field (may be replaced with delta if too memory consuming)
    ncounts = np.zeros((ngrid,ngrid,ngrid))

    # FIRST LOOP: deterministic bias
    # Parallelize the outer loop
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                # Initialize positions at the centre of the cell
                xtmp = (ii+0.5) * lcell
                ytmp = (jj+0.5) * lcell
                ztmp = (kk+0.5) * lcell

                # Compute redshift
                dtmp = np.sqrt((xtmp-xobs)**2 + (ytmp-yobs)**2 + (ztmp-zobs)**2)
                redshift = np.interp(dtmp, darr, zarr)

                if redshift>=zmin and redshift<zmax:

                    indtweb = int(tweb[ii,jj,kk])-1
                    inddweb = int(dweb[ii,jj,kk])-1

                    nmeanpars = nmean_arr[:,indtweb,inddweb]
                    alphapars = alpha_arr[:,indtweb,inddweb]
                    betapars = beta_arr[:,indtweb,inddweb]
                    dthpars = dth_arr[:,indtweb,inddweb]
                    rhoepspars = rhoeps_arr[:,indtweb,inddweb]
                    epspars = eps_arr[:,indtweb,inddweb]

                    meandens = np.interp(redshift, zzarrbias, meandensarr) 
                    nmean = np.interp(redshift, zzarrbias, nmeanpars)
                    alpha = np.interp(redshift, zzarrbias, alphapars)
                    beta = np.interp(redshift, zzarrbias, betapars)
                    rhoeps = np.interp(redshift, zzarrbias, rhoepspars)
                    eps = np.interp(redshift, zzarrbias, epspars)
                    dth = np.interp(redshift, zzarrbias, dthpars)

                    nmean *= 0.1

                    if delta[ii,jj,kk]<dth:
                        ncounts[ii,jj,kk] = 0.
                    else:
                        ncounts[ii,jj,kk] = (1.+delta[ii,jj,kk])**alpha * np.exp(-((1 + delta[ii,jj,kk])/rhoeps)**eps)

                    ncounts[ii,jj,kk] = nmean * meandens * ncounts[ii,jj,kk]
                    pnegbin = 1 - ncounts[ii,jj,kk]/(ncounts[ii,jj,kk] + beta)

                    ncounts[ii,jj,kk] = negative_binomial(beta, pnegbin)

                else:
                    ncounts[ii,jj,kk] = 0.
    
    return ncounts

# **********************************************
@njit(parallel=True, cache=True, fastmath=True)
def assign_positions(posx, posy, posz, ncounts, ngrid, lbox, radsearch):
    
    lcell = lbox / ngrid

    indstart = np.zeros((len(posx)))

    # Loop over the indstart array to establish the starting index - the loop is NOT parallelizable
    cc = 0
    ncountsflat = ncounts.flatten().astype('int')
    for ii in range(len(indstart)):
        indstart[ii] = cc
        cc += ncountsflat[ii]

    countstot = cc
    posxnew = np.zeros((countstot))
    posynew = np.zeros((countstot))
    posznew = np.zeros((countstot))
    #posxnew = np.array([0.], dtype=np.float64)
    #posynew = np.array([0.], dtype=np.float64)
    #posznew = np.array([0.], dtype=np.float64)

    posx = np.reshape(posx, (ngrid,ngrid,ngrid))
    posy = np.reshape(posy, (ngrid,ngrid,ngrid))
    posz = np.reshape(posz, (ngrid,ngrid,ngrid))

    # Now loop over the grid
    # 1) search for particles which are within the cell
    # 2) assing them until the number counts are matched
    # 3) if not enough throw random
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                # Assign positions only if the number of counts is >0
                if ncounts[ii,jj,kk]>0:

                    xlow = lcell*ii
                    xtop = lcell*(ii+1)
                    ylow = lcell*jj
                    ytop = lcell*(jj+1)
                    zlow = lcell*kk
                    ztop = lcell*(kk+1)

                    # Initialize empty arrays of positions inside the cell
                    #xtmparr = np.array([0.], dtype=nb.float64)
                    #ytmparr = np.array([0.], dtype=nb.float64)
                    #ztmparr = np.array([0.], dtype=nb.float64)

                    ind3d = kk + ngrid*(jj + ngrid*ii)
                    indini = int(indstart[ind3d])

                    gg = 0

                    # Initialize arrays of particle positiong within the search radius
                    xtmprad = np.zeros((2*radsearch+1)**3)
                    ytmprad = np.zeros((2*radsearch+1)**3)
                    ztmprad = np.zeros((2*radsearch+1)**3)

                    # Now loop on particles close to the grid position
                    for ll in range(ii-radsearch,ii+radsearch+1):
                        for mm in range(jj-radsearch,jj+radsearch+1):
                            for nn in range(kk-radsearch,kk+radsearch+1):

                                # Apply BCs for the indices
                                if ll<0:
                                    ll+=ngrid
                                elif ll>=ngrid:
                                    ll-=ngrid

                                if mm<0:
                                    mm+=ngrid
                                elif mm>=ngrid:
                                    mm-=ngrid

                                if nn<0:
                                    nn+=ngrid
                                elif nn>=ngrid:
                                    nn-=ngrid

                                xtmp = posx[ll,mm,nn]
                                ytmp = posy[ll,mm,nn]
                                ztmp = posz[ll,mm,nn]

                                # Evaluate if the particle is within the cell
                                if xtmp>=xlow and xtmp<xtop and ytmp>=ylow and ytmp<ytop and ztmp>=zlow and ztmp<ztop:
                                    xtmprad[gg] = xtmp
                                    ytmprad[gg] = ytmp
                                    ztmprad[gg] = ztmp
                                else:
                                    xtmprad[gg] = -1.
                                    ytmprad[gg] = -1.
                                    ztmprad[gg] = -1.

                                gg += 1

                    # Keep only valid positions
                    xtmprad = xtmprad[xtmprad>=0.]
                    ytmprad = ytmprad[ytmprad>=0.]
                    ztmprad = ztmprad[ztmprad>=0.]

                    lenarr = len(xtmprad)
                    
                    # Now evalute if we have enough DM particles or if we have to sample some new one
                    
                    if lenarr == ncounts[ii,jj,kk]:

                        for ww in range(lenarr):
                            posxnew[indini+ww] = xtmprad[ww]
                            posynew[indini+ww] = ytmprad[ww]
                            posznew[indini+ww] = ztmprad[ww]

                    elif lenarr > ncounts[ii,jj,kk]:

                        indsh = np.arange(len(xtmprad))
                        np.random.shuffle(indsh)

                        xtmprad = xtmprad[indsh]
                        ytmprad = ytmprad[indsh]
                        ztmprad = ztmprad[indsh]
                        
                        for ww in range(len(indsh)):
                            posxnew[indini+ww] = xtmprad[ww]
                            posynew[indini+ww] = ytmprad[ww]
                            posznew[indini+ww] = ztmprad[ww]
                    
                    elif lenarr < ncounts[ii,jj,kk]:

                        cnt = 0
                        
                        for ww in range(lenarr):
                            posxnew[indini+ww] = xtmprad[ww]
                            posynew[indini+ww] = ytmprad[ww]
                            posznew[indini+ww] = ztmprad[ww]

                            cnt += 1
                            
                        while cnt<ncounts[ii,jj,kk]:
                            xrand = np.random.uniform(xlow, xtop)
                            yrand = np.random.uniform(ylow, ytop)
                            zrand = np.random.uniform(zlow, ztop)

                            posxnew[indini+cnt] = xrand
                            posynew[indini+cnt] = yrand
                            posznew[indini+cnt] = zrand

                            cnt += 1
                            
                else: # This closes if ncounts>0
                    pass
    
    return posxnew, posynew, posznew

# **********************************************
# **********************************************     
@njit(parallel=True, cache=True, fastmath=True)
def assign_positions_random(posx, posy, posz, ncounts, ngrid, lbox, radsearch):

    lcell = lbox / ngrid

    indstart = np.zeros((len(posx)))

    # Loop over the indstart array to establish the starting index - the loop is NOT parallelizable
    cc = 0
    ncountsflat = ncounts.flatten().astype('int')
    for ii in range(len(indstart)):
        indstart[ii] = cc
        cc += ncountsflat[ii]

    countstot = cc
    posxnew = np.zeros((countstot))
    posynew = np.zeros((countstot))
    posznew = np.zeros((countstot))
    #posxnew = np.array([0.], dtype=np.float64)   
    #posynew = np.array([0.], dtype=np.float64)      
    #posznew = np.array([0.], dtype=np.float64)   

    posx = np.reshape(posx, (ngrid,ngrid,ngrid))
    posy = np.reshape(posy, (ngrid,ngrid,ngrid))
    posz = np.reshape(posz, (ngrid,ngrid,ngrid))

    # Now loop over the grid                            
    # 1) search for particles which are within the cell           
    # 2) assing them until the number counts are matched        
    # 3) if not enough throw random
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                # Assign positions only if the number of counts is >0 
                if ncounts[ii,jj,kk]>0:

                    xlow = lcell*ii
                    xtop = lcell*(ii+1)
                    ylow = lcell*jj
                    ytop = lcell*(jj+1)
                    zlow = lcell*kk
                    ztop = lcell*(kk+1)

                    # Initialize empty arrays of positions inside the cell 
                    #xtmparr = np.array([0.], dtype=nb.float64)       
                    #ytmparr = np.array([0.], dtype=nb.float64)  
                    #ztmparr = np.array([0.], dtype=nb.float64)
                    ind3d = kk + ngrid*(jj + ngrid*ii)
                    indini = int(indstart[ind3d])

                    cnt = 0

                    while cnt<ncounts[ii,jj,kk]:
                        xrand = np.random.uniform(xlow, xtop)
                        yrand = np.random.uniform(ylow, ytop)
                        zrand = np.random.uniform(zlow, ztop)

                        posxnew[indini+cnt] = xrand
                        posynew[indini+cnt] = yrand
                        posznew[indini+cnt] = zrand

                        cnt += 1

                else: # This closes if ncounts>0  
                    pass

    return posxnew, posynew, posznew
    

# **********************************************
# **********************************************
# **********************************************
print('---------------------------------------------------------')
print('Code to populate lightcone DM fields with galaxies/haloes')
print('---------------------------------------------------------')

ti = time.time()

lcell = lbox/ngrid

xobs = obspos[0]
yobs = obspos[1]
zobs = obspos[2]

print('')
print('Reading input ...')

delta = np.fromfile(dm_filename, dtype=np.float32)  # In real space
delta = np.reshape(delta, (ngrid,ngrid,ngrid))

tweb = np.fromfile(tweb_filename, dtype=np.float32)  # In real space
twebdelta = np.fromfile(twebdelta_filename, dtype=np.float32)  # In real space 

# Positions
posx = np.fromfile(posx_filename, dtype=np.float32)   
posy = np.fromfile(posy_filename, dtype=np.float32) 
posz = np.fromfile(posz_filename, dtype=np.float32)

posx += xobs
posy += yobs
posz += zobs

# Now they are velocity vectors
vx = np.fromfile(vx_filename, dtype=np.float32)  
vy = np.fromfile(vy_filename, dtype=np.float32) 
vz = np.fromfile(vz_filename, dtype=np.float32) 

# Reshape arrays from 1D to 3D --> reshape only arrays which have mesh structure, e.g. NOT positions
#delta = np.reshape(delta, (ngrid,ngrid,ngrid))
tweb = np.reshape(tweb, (ngrid,ngrid,ngrid))
twebdelta = np.reshape(twebdelta, (ngrid,ngrid,ngrid))

vx = np.reshape(vx, (ngrid,ngrid,ngrid))
vy = np.reshape(vy, (ngrid,ngrid,ngrid))
vz = np.reshape(vz, (ngrid,ngrid,ngrid))

# Read the tabulated redshift and comoving distance arrays
zarr = np.fromfile(zarr_filename, dtype=np.float32) 
darr = np.fromfile(darr_filename, dtype=np.float32) 

print('... done!')
print('')

# Apply the bias and get halo/galaxy number counts

print('Getting number counts via parametric bias ...')
#ncounts = apply_bias(delta, tweb, zzarrbias, aapars, alphapars, delta1pars, delta2pars, zmin, zmax, xobs, yobs, zobs, lbox, ngrid, darr, zarr)
ncounts = biasmodel_exp(ngrid, lbox, delta, tweb, twebdelta, zarr, darr, meandensarr, nmean_arr, alpha_arr, beta_arr, dth_arr, rhoeps_arr, eps_arr, xobs,yobs,zobs, zmin,zmax)

#ncounts.astype('float32').tofile('number_counts.dat')
# Compute the total number of objects
ncountstot = np.sum(ncounts)
print('Number counts diagnostics (min, max, mean): ', np.amin(ncounts), np.amax(ncounts), np.mean(ncounts))
print('... done!')
print('')

# Now populate the lightcone with positions
print('Populating lightcone with objects ...')
#ncounts = np.fromfile('number_counts.dat', dtype=np.float32).astype('int')
ncounts = np.reshape(ncounts, (ngrid,ngrid,ngrid))

# Now assign positions
posx, posy, posz = assign_positions(posx, posy, posz, ncounts, ngrid, lbox, radsearch)

posx.astype('float32').tofile(output_aux_dir + 'posxtr_rspace.dat')
posy.astype('float32').tofile(output_aux_dir + 'posytr_rspace.dat')
posz.astype('float32').tofile(output_aux_dir + 'posztr_rspace.dat')

#deltatr = get_cic(posx, posy, posz, lbox, ngrid)
#deltatr.astype('float32').tofile('deltatr_rspace.dat')

print('Tracer positions diagnostics: ', np.amin(posx), np.amax(posx), np.amin(posy), np.amax(posy), np.amin(posz), np.amax(posz))

print('... done!')
print('')

print('Mapping tracers positions from real to redshift space ...')
# Now the containers become redshifft space positions
posx, posy, posz = real_to_redshift_space(delta, tweb, posx, posy, posz, vx, vy, vz, ngrid,  lbox, zarr, darr, zmin, zmax, xobs, yobs, zobs, bv_arr, bb_arr, beta_arr, gamma_arr, zzarrbias)

posx.astype('float32').tofile(output_aux_dir + 'posxtr_zspace.dat')
posy.astype('float32').tofile(output_aux_dir + 'posytr_zspace.dat')
posz.astype('float32').tofile(output_aux_dir + 'posztr_zspace.dat')

#deltatr = get_cic(posx, posy, posz, lbox, ngrid)

#deltatr.astype('float32').tofile(output_aux_dir + 'deltatr_zspace.dat')

print('... done!')
print('')

print('The end')
