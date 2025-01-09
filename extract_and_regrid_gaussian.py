import numpy as np
from numba import njit, prange
import os
import time
import bias_parameters_lya as biaspars
import astropy.io.fits as fits
import healpy
import astropy.constants as const
from astropy.table import Table
from multiprocessing import Pool, cpu_count
import input_params as inpars

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

# I/O files           
nreal = inpars.nreal
version = inpars.version

# Input filenames     
input_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_boxes/v0/box%d/' %nreal
output_aux_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/' + version + '/skewers-%d/aux/' %nreal
output_dir = '/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/' + version + '/skewers-%d/' %nreal

# General parameters
posx_qso_filename = output_aux_dir + 'posxtr_zspace.dat'
posy_qso_filename = output_aux_dir + 'posytr_zspace.dat'
posz_qso_filename = output_aux_dir + 'posztr_zspace.dat'

fits_filename = output_dir + 'master.fits'

fluxout_filename = output_aux_dir + 'flux_zspace.dat'

zarr_filename = 'zarr.DAT'
darr_filename = 'dcomOM0.314OL0.686.DAT'

nside = 16
pixmax = 3072

num_processes = 96

# General parameters

lbox = 10000.
ngrid = 1800 

zmin = 1.77
zmax = 3.8

zmin_extr = 1.77#2.

hrbinw = 0.1

lammin = 3469.9
lammax = 6500.1
dlam = 0.2

# Subgrid parameters
lss_fact = 0.4
gamma = 0.3
scale = 5.
norm = 10.0
zknee = 2.6
zexp = 2.

alpha_lss = 1.0

# large scale bias
norm_lss = 1.5
gamma_lss = 2.4
zpivot = 4.0

mm_lss = 2.
qq_lss = 0.

# Rest-frame frequencies
lam_lya = 1215.67
lam_lyb = 1025.72
lam_SiII_1260 = 1260.42
lam_SiIII_1207 = 1206.50
lam_SiII_1193 = 1193.29
lam_SiII_1190 = 1190.42

# Absorption strenghts - Lya by definition is 1 (Numbers from Tab. 2 of Farr et al. 2020, https://arxiv.org/abs/1912.02763)
Ax_lya = 1.
Ax_lyb = 0.1901
Ax_SiII_1260 = 3.542e-4
Ax_SiIII_1207 = 1.8919e-3
Ax_SiII_1193 = 9.0776e-4
Ax_SiII_1190 = 1.28478e-4

# Observer positions
obspos = [5000.,5000.,5000.]


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

smallnum = 1e-6

saturate_flux = True

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

    weight = weight.flatten()
    
    lcell = lbox / ngrid

    delta = np.zeros((ngrid,ngrid,ngrid))

    dummylen = int(ngrid**3)

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
            wxl = 1 - wxc

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
            wyl = 1 - wyc

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
            wzl = 1 - wzc

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
                dtmp = np.sqrt((xtmp-xobs)**2 + (ytmp-xobs)**2 + (ztmp-zobs)**2)

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
def extract_skewers(posx, posy, posz, zmin, zmax, zarr, darr, hrbinw, flux, ngrid, lbox, xobs, yobs, zobs):

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
    indmat = np.zeros((len(posx), nbins))

    # Set up a template of distances                                                                                                                            
    dtemplate = np.linspace(dmin, dmax, nbins+1)
    dtemplate = 0.5*(dtemplate[1:] + dtemplate[:-1])

    ztemplate = np.interp(dtemplate, darr, zarr)

    for ii in range(len(posx)):

        ra, dec, zz = cartesian_to_sky(posx[ii],posy[ii],posz[ii], zarr, darr, xobs, yobs, zobs)
        #print(zz)

        for jj in range(len(dtemplate)):

            ztemp = ztemplate[jj]

            if ztemp<=zz and ztemp>zmin_extr and ztemp<zmax:
                posxlya, posylya, poszlya = sky_to_cartesian(ra,dec,ztemp, zarr, darr, xobs, yobs, zobs)
                indx = int(posxlya/lcell)
                indy = int(posylya/lcell)
                indz = int(poszlya/lcell)

                ind3d = int(indx + ngrid*(indy + ngrid*indz))

                skmat[ii,jj] = flux[indx,indy,indz]
                indmat[ii,jj] = ind3d

            else:
                skmat[ii,jj] = 1.
                indmat[ii,jj] = -99

    return ztemplate, dtemplate, skmat, indmat


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
        wxl = 1 - wxc

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
        wyl = 1 - wyc

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
        wzl = 1 - wzc

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
def flux_to_tau(ngrid, lbox, skmat):

    for ii in prange(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            skmat[ii,jj] = -np.log10(skmat[ii,jj])

    return skmat

# **********************************************     
@njit(parallel=True, cache=True, fastmath=True)
def tau_to_flux(ngrid, lbox, skmat):

    for	ii in prange(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            skmat[ii,jj] = np.exp(-skmat[ii,jj])

            if saturate_flux == True:
                if skmat[ii,jj]>1.:
                    skmat[ii,jj] = 1.

    return skmat

# **********************************************   
@njit(parallel=False, cache=True, fastmath=True)
def regrid_skewers(ngrid, lbox, zmin, zmax, lammin, lammax, dlam, ztemplate, lam_rf, skmat, Ax, is_flux_flag, return_flux_flag):

    if is_flux_flag==True:
        for kk in prange(skmat.shape[0]):
            for ll in range(skmat.shape[1]):
                skmat[kk,ll] = -np.log10(skmat[kk,ll])

    numbin = int((lammax-lammin)/dlam)
    lam_template = np.linspace(lammin, lammax, num=numbin+1)
    lam_template_cen = 0.5*(lam_template[1:]+lam_template[:-1])

    lam_arr = (1+ztemplate)*lam_rf

    nobj = skmat.shape[0]
    numbinold = skmat.shape[1]

    indarr = np.zeros(numbinold)
    indlarr = np.zeros(numbinold)

    warr = np.zeros(numbinold)
    wlarr = np.zeros(numbinold)

    # Do a loop to determine the indices of the regridding
    for jj in prange(len(indarr)):

        if lam_arr[jj]>lammin and lam_arr[jj]<lammax:
            indtmp = int((lam_arr[jj] - lammin)/dlam)
            indarr[jj] = indtmp
            ww = (lam_arr[jj]-lam_template[indtmp])/dlam

            #print(ww, indtmp, lam_arr[jj], lammin, lam_template[indtmp], dlam)

            if lam_arr[jj]>=(lam_template[indtmp]+0.5*dlam):
                indlarr[jj] = int(indtmp+1)
                warr[jj] = ww
                wlarr[jj] = 1-ww
            else:
                warr[jj] = ww+0.5
                wlarr[jj] = 1-ww
                indlarr[jj] = int(indtmp-1)
        else:
            indarr[jj] = -99
            indlarr[jj] = -99
            warr[jj] = 0.
            wlarr[jj] = 0.
            
    
    skmat_new = np.zeros((nobj, numbin))

    for ii in prange(nobj):
        for jj in range(numbinold):
            
            indd = int(indarr[jj])
            indl = int(indlarr[jj])
            wgt = warr[jj]
            wgtl = wlarr[jj]

            if indd>=0:
                skmat_new[ii,indd] +=  wgt * skmat[ii,jj]
                skmat_new[ii,indl] += wgtl * skmat[ii,jj]
                #skmat_new[ii,indd] +=  skmat[ii,jj] 

    if is_flux_flag==False:
        for kk in prange(skmat_new.shape[0]):
            for	ll in range(skmat_new.shape[1]):
                skmat_new[kk,ll] *= Ax #np.exp(-Ax * skmat_new[kk,ll])

                if return_flux_flag==True:
                    skmat_new[kk,ll] = np.exp(-skmat_new[kk,ll])

    return skmat_new, lam_template_cen

# ********************************************** 
@njit(parallel=True, cache=True, fastmath=True)
def combine_metals(SiII_1260, SiIII_1207, SiII_1193, SiII_1190):

    out = np.zeros((SiII_1260.shape[0], SiII_1260.shape[1]))

    for ii in prange(SiII_1260.shape[0]):
        for jj in range(SiII_1260.shape[1]):

            out[ii,jj] = np.exp(-(SiII_1260[ii,jj] + SiIII_1207[ii,jj] + SiII_1193[ii,jj] + SiII_1190[ii,jj]))

    return out

# **********************************************  
@njit(parallel=True, cache=True, fastmath=True)
def sum_metals(skmat_metals, skmat_dummy):

    for ii in prange(skmat_metals.shape[0]):
        for jj in range(skmat_metals.shape[1]):

            skmat_metals[ii,jj] += skmat_dummy[ii,jj]

    return skmat_metals

# *********************************************
# *********************************************   
@njit(parallel=False, cache=True, fastmath=True)
def tau_to_gaussian(arr, mu, sigma):

    arr = (np.log(arr) - mu) / sigma

    return arr

# *********************************************        
@njit(parallel=False, cache=True, fastmath=True)
def gaussian_to_tau(arr, mu, sigma):

    arr = np.exp(mu + sigma * arr)

    return arr

# *********************************************
def find_lognormal_parameters(tau, indd):

    mask = np.where(indd>0)

    mut = np.mean(tau[mask])
    sigmat = np.std(tau[mask])

    mu = np.log(mut**2 / np.sqrt(mut**2 + sigmat**2))
    sigma = np.sqrt(np.log(1. + sigmat**2/mut**2))

    return mu, sigma

# *********************************************
@njit(parallel=False, cache=True, fastmath=True)
def add_small_scale_power(ngrid, lbox, skmat, indmat, ztemplate, dtemplate, gamma, mug, sigmag):

    skmatnew = 0. * skmat.copy()
    
    for ii in range(skmat.shape[0]):

        indstart=0
        indend=0
        cnt = 0
        total = 0.

        # Pass from tau to Gaussian space
        skmat[ii,:] = tau_to_gaussian(skmat[ii,:], mug, sigmag)
        
        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]>=0.:
                if cnt == 0:
                    indstart=jj
                    mean = skmat[ii,jj]

                #print(skmat[ii,jj])
                #rand = np.random.gamma(smallnum+norm*skmat[ii,jj]**gamma*(ztemplate[jj]/zknee)**zexp, scale=scale)#*0.1
                #rand = np.random.gamma(smallnum+norm*skmat[ii,jj]**gamma, scale=scale*(ztemplate[jj]/zknee)**zexp)#*0.1
                thsafe = skmat[ii,jj] / 6.
                sigmatmp = norm * skmat[ii,jj]**gamma * (ztemplate[jj]/zknee)**2.
                sigma = np.amin(np.array([thsafe, sigmatmp])) 
                rand = np.random.normal(0., sigma)
                #rand = np.random.gamma(smallnum+norm*skmat[ii,jj]**gamma, scale=scale*(ztemplate[jj]/zknee)**zexp)
                #print(rand)
                skmatnew[ii,jj] = skmatnew[ii,jj] + skmat[ii,jj] + rand #np.random.gamma(1.+skmat[ii,jj])*0.1

                total += rand
                cnt += 1
                
                if jj<skmat.shape[1]-1:
                    if indmat[ii,jj+1]!=indmat[ii,jj]:
                        indend=jj
                        meannew = total / cnt
                        #print(skmat[ii,jj], meannew, mean)
                        for kk in range(indstart,indend+1):
                            #print(indstart,indend+1)
                            skmatnew[ii,kk] = 0.5 * (skmatnew[ii,kk] - meannew + mean)

                            #print(skmat[ii,jj], skmatnew[ii,jj], meannew, mean)

                        #print('CHECK: ', np.mean(skmat[ii,indstart:indend+1]), np.mean(skmatnew[ii,indstart:indend+1]))
                        #print('CHECK IND: ', indmat[ii,indstart-1], indmat[ii,indstart],  indmat[ii,indend], indmat[ii,indend+1])
                        #print('')

                        cnt = 0
                        total = 0.
                
            else:
                skmatnew[ii,jj] = 1.#np.nan
        
        skmatnew[ii,:] = gaussian_to_tau(skmatnew[ii,:], mug, sigmag)

    # Consistency check
    for ii in range(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]<0 or np.isnan(skmatnew[ii,jj])==True or np.isinf(skmatnew[ii,jj])==True:
                skmatnew[ii,jj] = 0.
                
    return skmatnew

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def add_small_scale_power_fluxtau(ngrid, lbox, skmat, indmat, ztemplate, dtemplate, gamma, mug, sigmag):

    skmatnew = 0. * skmat.copy()

    for ii in range(skmat.shape[0]):

        indstart=0
        indend=0
        cnt = 0
        total = 0.

        # Pass from tau to Gaussian space                                                                                                                                               
        #skmat[ii,:] = tau_to_gaussian(skmat[ii,:], mug, sigmag)                                                                                                                        

        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]>=0.:
                if cnt == 0:
                    indstart=jj
                    mean = skmat[ii,jj]

                #if skmat[ii,jj]<=0.5:                                                                                                                                                  
                #    sigma = skmat[ii,jj] / 6.                       
                #elif skmat[ii,jj]>0.5:    
                #    sigma = (1.-skmat[ii,jj]) / 6.   

                skmat[ii,jj] = -np.log(skmat[ii,jj])
                sigma = skmat[ii,jj]**alpha_lss

                #rand = np.random.normal(0., sigma)
                #mu = np.log10(mu) - sigma**2/2. 
                rand = 10. * (np.random.lognormal(0., sigma) - np.exp((sigma**2)/2.))
                #rand = np.random.gamma(smallnum+norm*skmat[ii,jj]**gamma*(ztemplate[jj]/zknee)**zexp, scale=scale)#*0.1 
                
                #print(rand)
                
                skmatnew[ii,jj] = np.exp(-(skmatnew[ii,jj] + skmat[ii,jj] + rand))

                skmat[ii,jj] = np.exp(-skmat[ii,jj])

                total = total + (skmatnew[ii,jj]-skmat[ii,jj])
                #total += rand                                                                                                                                                          
                cnt += 1

                if jj<skmat.shape[1]-1:
                    if indmat[ii,jj+1]!=indmat[ii,jj]:
                        indend=jj
                        meannew = total / cnt
                        #print(skmat[ii,jj], meannew, mean)                                                                                                                             
                        for kk in range(indstart,indend+1):
                            #print(indstart,indend+1)                                                                                                                                   
                            skmatnew[ii,kk] = 0.5 * (skmatnew[ii,kk] - meannew + mean)
                            #print(skmat[ii,jj], skmatnew[ii,jj], meannew, mean)                                                                                                        

                        if np.isnan(np.mean(skmatnew[ii,indstart:indend+1]))==True:
                            print('CHECK: ', np.mean(skmat[ii,indstart:indend+1]), np.mean(skmatnew[ii,indstart:indend+1]))  
                        #print('CHECK IND: ', indmat[ii,indstart-1], indmat[ii,indstart],  indmat[ii,indend], indmat[ii,indend+1])              
                        #print('')                                               
                        cnt = 0
                        total = 0.

            else:
                skmatnew[ii,jj] = 1.#np.nan       
        #skmatnew[ii,:] = gaussian_to_tau(skmatnew[ii,:], mug, sigma

        # Consistency check                     
    for ii in range(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]<0 or np.isnan(skmatnew[ii,jj])==True or np.isinf(skmatnew[ii,jj])==True:
                skmatnew[ii,jj] = 1.
            if skmatnew[ii,jj]<0.:
                skmatnew[ii,jj] = 0.

    return skmatnew

        
# **********************************************                                                                                                                     
@njit(parallel=False, cache=True, fastmath=True)
def add_small_scale_power_flux(ngrid, lbox, skmat, indmat, ztemplate, dtemplate, gamma, mug, sigmag):

    skmatnew = 0. * skmat.copy()

    for ii in range(skmat.shape[0]):

        indstart=0
        indend=0
        cnt = 0
        total = 0.

        # Pass from tau to Gaussian space                                                                                                                            
        #skmat[ii,:] = tau_to_gaussian(skmat[ii,:], mug, sigmag)

        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]>=0.:
                if cnt == 0:
                    indstart=jj
                    mean = skmat[ii,jj]

                if skmat[ii,jj]<=0.5:
                    sigma = skmat[ii,jj] / 6.
                elif skmat[ii,jj]>0.5:
                    sigma = (1.-skmat[ii,jj]) / 6.
                
                rand = np.random.normal(0., sigma)
                #rand = np.random.gamma(smallnum+norm*skmat[ii,jj]**gamma, scale=scale*(ztemplate[jj]/zknee)**zexp)                                                  
                #print(rand)
                skmatnew[ii,jj] = skmatnew[ii,jj] + skmat[ii,jj] + rand #np.random.gamma(1.+skmat[ii,jj])*0.1

                total += rand
                cnt += 1

                if jj<skmat.shape[1]-1:
                    if indmat[ii,jj+1]!=indmat[ii,jj]:
                        indend=jj
                        meannew = total / cnt
                        #print(skmat[ii,jj], meannew, mean)                                                                                                                            
                        for kk in range(indstart,indend+1):
                            #print(indstart,indend+1)
                            skmatnew[ii,kk] = 0.5 * (skmatnew[ii,kk] - meannew + mean)
                            #print(skmat[ii,jj], skmatnew[ii,jj], meannew, mean)                                                                                                       

                        print('CHECK: ', np.mean(skmat[ii,indstart:indend+1]), np.mean(skmatnew[ii,indstart:indend+1]))                                                              
                        #print('CHECK IND: ', indmat[ii,indstart-1], indmat[ii,indstart],  indmat[ii,indend], indmat[ii,indend+1])                                                    
                        #print('')                                                                                                                                                    
                        cnt = 0
                        total = 0.

            else:
                skmatnew[ii,jj] = 1.#np.nan                                                                                                                                            
        #skmatnew[ii,:] = gaussian_to_tau(skmatnew[ii,:], mug, sigma

    # Consistency check                                                                                                                                                                
    for ii in range(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]<0 or np.isnan(skmatnew[ii,jj])==True or np.isinf(skmatnew[ii,jj])==True:
                skmatnew[ii,jj] = 1.

    return skmatnew

# **********************************************
@njit(parallel=False, cache=True, fastmath=True)
def correct_large_scale_bias(skmat, indmat, ztemplate):

    for ii in range(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            if indmat[ii,jj]>=0.:

                #factor = norm_lss * (ztemplate[jj] / zpivot ) ** gamma_lss
                factor = 0.4
                #factor = mm_lss * (ztemplate[jj] / zpivot ) + qq_lss #** gamma_lss
                #print(factor)
                skmat[ii,jj] *= factor

    return skmat

# **********************************************
def ensure_regularity(skmat):

    for ii in range(skmat.shape[0]):
        for jj in range(skmat.shape[1]):

            if np.isnan(skmat[ii,jj])==True:
                skmat[ii,jj] = 1.
            elif np.isinf(skmat[ii,jj])==True:
                skmat[ii,jj] = 1.
            elif skmat[ii,jj]>1.:
                skmat[ii,jj] = 1.
            elif skmat[ii,jj]<0:
                skmat[ii,jj]<0.
            else:
                pass
    return skmat

# **********************************************
# **********************************************
# **********************************************
print('--------------------------------')
print('Extract and regrid Lya skewers')
print('--------------------------------')

ti = time.time()

lcell = lbox/ngrid

xobs = obspos[0]
yobs = obspos[1]
zobs = obspos[2]

# Read the tabulated redshift and comoving distance arrays                                                                                                      
zarr = np.fromfile(zarr_filename, dtype=np.float32)
darr = np.fromfile(darr_filename, dtype=np.float32)


print('Read QSO positions ...')
# Now read QSO positions in redshift space                                        
# The containers are used first for RA,DEC,z     
cx = np.fromfile(open(posx_qso_filename, 'r'), dtype=np.float32)
cy = np.fromfile(open(posy_qso_filename, 'r'), dtype=np.float32)
cz = np.fromfile(open(posz_qso_filename, 'r'), dtype=np.float32)
print('... done!')
print('') 

# Now open the fits catalog                                                                                                                                     
rawfits = fits.open(fits_filename)
catfits = rawfits[1].data
ra = catfits['RA']
dec = catfits['DEC']
zz = catfits['Z']
mockid = catfits['MOCKID']

# Read flux field                                                                                                                                               
flux = np.fromfile(open(fluxout_filename, 'r'), dtype=np.float32)
flux = np.reshape(flux, (ngrid,ngrid,ngrid))

# Cut the QSO positions - keep only QSOs relevant for lya                                                                                                       
cx = cx[np.logical_and(zz>zmin, zz<zmax)]
cy = cy[np.logical_and(zz>zmin, zz<zmax)]
cz = cz[np.logical_and(zz>zmin, zz<zmax)]

ra = ra[np.logical_and(zz>zmin, zz<zmax)]
dec = dec[np.logical_and(zz>zmin, zz<zmax)]
mockid = mockid[np.logical_and(zz>zmin, zz<zmax)]
zz = zz[np.logical_and(zz>zmin, zz<zmax)]

#ra_rad = (ra+180.) / 180. * np.pi
#dec_rad = (dec+90.) / 180. * np.pi

healpix = healpy.ang2pix(nside, np.radians(90.-dec), np.radians(ra), nest=True)

# Define the DESI footprint
footprint = fits.open('DESI_footprint_nside16.fits')
dark = footprint[1].data['DESI_DARK']

#print('Extracting skewers ...')

# ****************************************
def extract_and_regrid_parallel(ii):

    dirnum = int(ii/100)

    if dark[ii]==True:

        print(ii)

        if not os.path.exists(output_dir + '/%d/%d/' %(dirnum,ii)):
            os.mkdir(output_dir + '%d/%d/' %(dirnum,ii))
            
        xx = cx[healpix==ii]
        yy = cy[healpix==ii]
        zz = cz[healpix==ii]
        mockidd = mockid[healpix==ii]

        ztemplate, dtemplate, skmat, indmat = extract_skewers(xx, yy, zz, zmin, zmax, zarr, darr, hrbinw, flux, ngrid, lbox, xobs, yobs, zobs)

        # First, pass from flux to tau for further computations
        #skmat = flux_to_tau(ngrid, lbox, skmat)

         # Compute parameters of the lognormal transform      
        #mug, sigmag = find_lognormal_parameters(skmat, indmat)
        mug = 0.
        sigmag = 0.

        # Correct large-scale bias
        skmat = flux_to_tau(ngrid, lbox, skmat)
        #skmat = correct_large_scale_bias(skmat, indmat, ztemplate)
        #skmat = tau_to_flux(ngrid, lbox, skmat)

        # ADD SMALL_SCALE FLUCTUATIONS  
        #skmat = add_small_scale_power_fluxtau(ngrid, lbox, skmat, indmat, ztemplate, dtemplate, gamma, mug, sigmag)

        #print('Diagnostics (min, max, mean): ', np.amin(np.exp(-skmat[indmat>=0])), np.amax(np.exp(-skmat[indmat>=0])), np.mean(np.exp(-skmat[indmat>=0])), np.median(np.exp(-skmat[indmat>=0])))

        #skmat = flux_to_tau(ngrid, lbox, skmat)

        #print(np.where(np.isnan(skmat)==True))

        # REGRIDDING
        # Lya
        skmat_lya, lam_template = regrid_skewers(ngrid, lbox, zmin, zmax, lammin, lammax, dlam, ztemplate, lam_lya, skmat, Ax_lya, False, False)

        # RESCALE TO MATCH THE BIAS
        #skmat_lya *= lss_fact

        # Lyb
        skmat_lyb, lam_template = regrid_skewers(ngrid, lbox, zmin, zmax, lammin, lammax, dlam, ztemplate, lam_lyb, skmat, Ax_lyb, False, False)

        # Metals
        skmat_metals = np.zeros((skmat_lya.shape[0], skmat_lya.shape[1]))

        skmat_dummy, lam_template = regrid_skewers(ngrid, lbox, zmin, zmax, lammin, lammax, dlam, ztemplate, lam_SiII_1260, skmat, Ax_SiII_1260, False, False)
        skmat_metals = sum_metals(skmat_metals, skmat_dummy)

        skmat_dummy, lam_template = regrid_skewers(ngrid, lbox, zmin, zmax, lammin, lammax, dlam, ztemplate, lam_SiIII_1207, skmat, Ax_SiIII_1207, False, False)
        skmat_metals = sum_metals(skmat_metals, skmat_dummy)

        skmat_dummy, lam_template = regrid_skewers(ngrid, lbox, zmin, zmax, lammin, lammax, dlam, ztemplate, lam_SiII_1193, skmat, Ax_SiII_1193, False, False)
        skmat_metals = sum_metals(skmat_metals, skmat_dummy)

        skmat_dummy, lam_template = regrid_skewers(ngrid, lbox, zmin, zmax, lammin, lammax, dlam, ztemplate, lam_SiII_1190, skmat, Ax_SiII_1190, False, False)
        skmat_metals = sum_metals(skmat_metals, skmat_dummy)

        # PASS FROM TAU TO FLUX
        skmat_lya = tau_to_flux(ngrid, lbox, skmat_lya)
        skmat_lyb = tau_to_flux(ngrid, lbox, skmat_lyb)
        skmat_metals = tau_to_flux(ngrid, lbox, skmat_metals)

        #for ll in range(skmat_lya.shape[0]):
        #    print(np.amin(skmat_lya[ll,:]), np.amax(skmat_lya[ll,:]), np.mean(skmat_lya[ll,:]))

        # Flux diagnostics
        skmat_lya = ensure_regularity(skmat_lya)
        print('Min, max, mean flux lya: ', np.amin(skmat_lya), np.amax(skmat_lya), np.mean(skmat_lya))
        
        # Now construct the fits file
        # Read QSO catalog and build HDU
        rawqso = fits.open(output_dir + 'master.fits')

        qsotab = rawqso[1].data
        arr, inddx, inddx2 = np.intersect1d(qsotab['MOCKID'], mockidd, return_indices=True)
        data = qsotab[inddx]
        
        c1 = fits.Column(name='RA', array=data['RA'], format='D')
        c2 = fits.Column(name='DEC', array=data['DEC'], format='D')
        c3 = fits.Column(name='Z_noRSD', array=data['Z_noRSD'], format='D')
        c4 = fits.Column(name='Z', array=data['Z'], format='D')
        c5 = fits.Column(name='MOCKID', array=data['MOCKID'], format='D')
        hdu1 = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5], name='METADATA')

        # Read DLA catalog and build HDU
        rawdla = fits.open(output_dir + 'master_DLA.fits')

        dlatab = rawdla[1].data
        arr, inddx, inddx2 = np.intersect1d(dlatab['MOCKID'], mockidd, return_indices=True)
        data = dlatab[inddx]

        c1 = fits.Column(name='RA', array=data['RA'], format='D')
        c2 = fits.Column(name='DEC', array=data['DEC'], format='D')
        c3 = fits.Column(name='Z_DLA_NO_RSD', array=data['Z_DLA_NO_RSD'], format='D')
        c4 = fits.Column(name='Z_DLA_RSD', array=data['Z_DLA_RSD'], format='D')
        c5 = fits.Column(name='MOCKID', array=data['MOCKID'], format='D')
        c6 = fits.Column(name='DLAID', array=data['DLAID'], format='D')
        c7 = fits.Column(name='N_HI_DLA', array=data['NHI_DLA'], format='D')
        hdudla = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7], name='DLA')

        hdu_list = fits.HDUList([
            fits.PrimaryHDU(),
            hdu1,
            fits.ImageHDU(lam_template),
            fits.ImageHDU(skmat_lya),
            fits.ImageHDU(skmat_lyb),
            fits.ImageHDU(skmat_metals),
            hdudla])

        hdu_list[1].name = 'METADATA'
        hdu_list[2].name = 'WAVELENGTH'
        hdu_list[3].name = 'F_LYA'
        hdu_list[4].name = 'F_LYB'
        hdu_list[5].name = 'F_METALS'
        hdu_list[6].name = 'DLA'

        hdu_list[1].header['HPXNSIDE'] = 16
        hdu_list[1].header['HPXPIXEL'] = ii
        hdu_list[1].header['HPXNEST'] = True
        hdu_list[1].header['LYA'] = 1215.67

        hdu_list.writeto(output_dir + '%d/%d/transmission-%d-%d.fits.gz' %(dirnum,ii,nside,ii), overwrite=True)

        #break
        

# ******************************************************************

tin = time.time()

# First check if the master directory exists. If not, create it, together with the subdirectories
if not os.path.exists('/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/%s/' %version):
    os.mkdir('/global/cfs/cdirs/desi/mocks/lya_forest/develop/cs-alpt/alpt_skewers/%s/' %version)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for ii in range(pixmax):
    if not os.path.exists(output_dir + '%d/' %(ii//100)):
        os.mkdir(output_dir + '%d/' %(ii//100))

# Now start extraction and regridding
ii_list = [(ii) for ii in range(pixmax)]

if num_processes<0:
    num_processes = cpu_count()
else:
    num_processes = num_processes

with Pool(processes=num_processes) as pool:
    pool.map(extract_and_regrid_parallel, ii_list)

tfin = time.time()

dt = (tfin-tin)/60.

print('Elapsed ' + str(dt) + ' minutes ...')
