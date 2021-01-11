#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 08:40:17 2017

@author: hmonteiro
"""

import numpy as np
from scipy.interpolate import interp1d,LinearNDInterpolator,griddata
from scipy.integrate import trapz,cumtrapz
import matplotlib.pyplot as plt
from astropy import units as u
import sys
from scipy import stats
from sklearn.neighbors.dist_metrics import DistanceMetric
from scipy.spatial.distance import cdist
from scipy import stats

###############################################
# Save binary file with full isochrone grid
def save_mod_grid(dir, isoc_set='UBVRI'):
    if(isoc_set == 'UBVRI'):
        mod_grid = np.genfromtxt(dir+'parsec1.2sc-full.txt',names=True)
        np.save(dir+'full_isoc_UBVRI.npy',mod_grid)
        
    if(isoc_set == 'GAIA'):
        mod_grid = np.genfromtxt(dir+'parsec1.2s-tycho-full.txt',names=True)
        use = ['Zini','Age','Mini','Gmag','G_BPmag','G_RPmag','B_Tmag',
               'V_Tmag','Jmag','Hmag','Ksmag']
        np.save(dir+'full_isoc_GAIA.npy',mod_grid[use])
        
        
    return

###############################################
# Load binary file with full isochrone grid
# and returns array of data and arrays of unique age and Z values
#
def load_mod_grid(dir, isoc_set='UBVRI'):
    global mod_grid
    global age_grid
    global z_grid

    if(isoc_set == 'UBVRI'):
        mod_grid = np.load(dir+'full_isoc_UBVRI.npy')
        
    if(isoc_set == 'GAIA'):
        mod_grid = np.load(dir+'full_isoc_GAIA_CMD33.npy')

    if(isoc_set == 'MIST-UBVRI'):
        mod_grid = np.load(dir+'full_isoc_MIST-UBVRI.npy')
        
    if(isoc_set == 'MIST-GAIA'):
        mod_grid = np.load(dir+'full_isoc_MIST-GAIA.npy')
            
    age_grid = np.unique(mod_grid['Age'])
    z_grid = np.unique(mod_grid['Zini'])
    
    return mod_grid, age_grid, z_grid
###############################################
# truncated PAreto for Salpeter IMF

def salpeter(alpha, nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin), 1000),axis=0)

    ind_low = np.where(mass_int <= 1.)    
    imf_val = mass_int**(-alpha)
    
    #normalize
    imf_norm =  imf_val / (trapz(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumtrapz(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses

###############################################
# tapered deMarchi IMF

def deMarchi(alpha, beta, nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin), 1000),axis=0)

    ind_low = np.where(mass_int <= 1.)    
    imf_val = mass_int**(-alpha)*(1.-np.exp(-(mass_int/1.)**(-beta)))
    
    #normalize
    imf_norm =  imf_val / (trapz(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumtrapz(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses

###############################################
#     
def MillerScalo(alpha,nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin), 100),axis=0)

    ind_low = np.where(mass_int <= 1.)    
    imf_val = mass_int**(-alpha)
    imf_val[ind_low] = mass_int[ind_low]**(0.)
    
    #normalize
    imf_norm =  imf_val / (trapz(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumtrapz(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses

###############################################
# Chabrier (2001) exponential form of the IMF.
# http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
def chabrier(alpha,nstars, Mmin, Mmax,seed=None):
    
    #mass_int = np.linspace(Mmin,Mmax,1000)
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin),1000),axis=0)
    # Chabrier (2001) exponential form of the IMF.
    # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
#    imf_val = 3. * mass_int ** (-3.3) * np.exp(-(716.4 / mass_int) ** 0.25)
    
    #Chabrier (2003) - http://adsabs.harvard.edu/abs/2003PASP..115..763C
    ind_low = np.where(mass_int <= 1.)    
    imf_val = 4.43e-2*mass_int**(-alpha)
    imf_val[ind_low] = 0.158*np.exp(-0.5*(np.log10(mass_int[ind_low])-
           np.log10(0.08))**2/0.69**2)
    
    #normalize
    imf_norm =  imf_val / (trapz(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumtrapz(imf_norm,mass_int, initial=0)
    
    r = np.random.RandomState(seed)
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(r.rand(nstars))

    return gen_masses

###############################################
# Chabrier (2001) exponential form of the IMF.
# http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
def chabrier_bin(nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.linspace(Mmin,Mmax,100)
#    mass_int = np.logspace(np.log10(Mmax),np.log10(Mmin),base=10.)
    # Chabrier (2001) exponential form of the IMF.
    # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
#    imf_val = 3. * mass_int ** (-3.3) * np.exp(-(716.4 / mass_int) ** 0.25)
    
    #Chabrier (2003) - http://adsabs.harvard.edu/abs/2003PASP..115..763C
    ind_low = np.where(mass_int <= 1.)    
    imf_val = 4.43e-2*mass_int**(-2.1)
    imf_val[ind_low] = 0.086*np.exp(-0.5*(np.log10(mass_int[ind_low])-
           np.log10(0.22))**2/0.57**2)
    
    #normalize
    imf_norm =  imf_val / (trapz(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumtrapz(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses


###############################################
# sample from given isochrone
    
def sample_from_isoc(rawisoc,bands,refMag,nstars,imf='chabrier',alpha=2.3,
                     beta=-3,Mcut=False,seed=None,binmasses=None):
    
#    if (Mcut and Abscut): sys.exit('Both Mcut and Abscut are not allowed simultaneaously!')
    
    if(Mcut):
        ind = np.abs(rawisoc[refMag]-Mcut).argmin()
        Mmin, Mmax = rawisoc['Mini'][ind], np.max(rawisoc['Mini'])
        if (Mmin==Mmax): Mmin=0.99*Mmin
    else:
        Mmin, Mmax = np.min(rawisoc['Mini']),np.max(rawisoc['Mini'])
        
    # get mass vector according to IMF
    
    if imf:
        if imf == 'salpeter':
            # print 'Using Salpeter IMF with alpha=2.35 in mass interval: [',Mmin,',',Mmax,']'
            masses = salpeter(alpha, nstars, Mmin, Mmax)
        if imf == 'chabrier':
            # print 'Using Chabrier(2001) IMF in mass interval: [',Mmin,',',Mmax,']'
            masses = chabrier(alpha,nstars, Mmin, Mmax,seed=seed)
        if imf == 'MillerScalo':
            # print MillerScalo
            masses = MillerScalo(alpha,nstars, Mmin, Mmax,seed=seed)
        if imf == 'deMarchi':
            masses = deMarchi(alpha,beta, nstars, Mmin, Mmax,seed=seed)
    else:
        raise ValueError('The IMF was not specified')

    if (binmasses is not None):
        masses = binmasses
        masses[masses<Mmin]=Mmin
        masses[masses>Mmax]=Mmax

    # interpolate photometry on the grid for the given masses
    photint = []

    for filter in bands:
        aux = interp1d(rawisoc['Mini'],rawisoc[filter])
        photint.append(aux(masses))
        
    photint.append(masses)
    
    cols = bands[:]
    cols.append('Mini')
    
    return np.core.records.fromarrays(photint, names=cols)
    
###############################################
# function to get isochrone from grid given an age, metalicity

def get_iso_from_grid(age,met,bands,refMag,Abscut=False, nointerp=False):
    
    global mod_grid, age_grid, z_grid
    # check to see if grid is loaded
    if 'mod_grid' not in globals(): 
        raise NameError('Isochrone grid not loaded!')
        
    # find closest values to given age and Z
    dist_age = np.abs(age - age_grid)#/age
    ind_age = dist_age.argsort()
    dist_z = np.abs(met - z_grid)#/met
    ind_z = dist_z.argsort()
    
    dist0 = np.sqrt(dist_age[ind_age[0]]**2 + dist_z[ind_z[0]]**2)
    dist1 = np.sqrt(dist_age[ind_age[1]]**2 + dist_z[ind_z[1]]**2)

#    dist0 = np.sqrt(dist_age[ind_age[0]]**2)
#    dist1 = np.sqrt(dist_age[ind_age[1]]**2)
    
    dist_age_0 = dist_age[ind_age[0]]/(dist_age[ind_age[0]]+dist_age[ind_age[1]])
    dist_age_1 = dist_age[ind_age[1]]/(dist_age[ind_age[0]]+dist_age[ind_age[1]])
    dist_z_0 = dist_z[ind_z[0]]/(dist_z[ind_z[0]]+dist_z[ind_z[1]])
    dist_z_1 = dist_z[ind_z[1]]/(dist_z[ind_z[0]]+dist_z[ind_z[1]])
    
    dist0 = np.sqrt(dist_age_0**2 + dist_z_0**2)
    dist1 = np.sqrt(dist_age_1**2 + dist_z_1**2)
    
    # get the closest isochrone to the given age and Z
    #apply absolute mag cut if set
    if(Abscut):
        iso1 = mod_grid[(mod_grid['Age'] == age_grid[ind_age[0]]) & 
                       (mod_grid['Zini'] == z_grid[ind_z[0]]) & 
                       (mod_grid[refMag] < Abscut)]
        iso2 = mod_grid[(mod_grid['Age'] == age_grid[ind_age[1]]) & 
                       (mod_grid['Zini'] == z_grid[ind_z[1]]) & 
                       (mod_grid[refMag] < Abscut)]
    else:
        iso1 = mod_grid[(mod_grid['Age'] == age_grid[ind_age[0]]) &
                       (mod_grid['Zini'] == z_grid[ind_z[0]])]
        iso2 = mod_grid[(mod_grid['Age'] == age_grid[ind_age[1]]) &
                       (mod_grid['Zini'] == z_grid[ind_z[1]])]   
        
    photint = []
    
    for filter in bands:
        mass_int = []
        f_int = []
        
        for n in np.unique(iso1['label']):
            
            f1 = iso1[filter][iso1['label'] == n]
            f2 = iso2[filter][iso2['label'] == n]
            
            m1 = iso1['Mini'][iso1['label'] == n]
            m2 = iso2['Mini'][iso2['label'] == n]

            if(f1.size < 2 or f2.size < 2):
                continue

            if(f1.size > f2.size):
                npoints = f2.size
                
                f1i = interp1d(np.arange(f1.size),f1)
                f1 = f1i(np.linspace(0,f1.size-1,npoints))
                
                m1i = interp1d(np.arange(m1.size),m1)
                m1 = m1i(np.linspace(0,m1.size-1,npoints))
            else:
                npoints = f1.size

                f2i = interp1d(np.arange(f2.size),f2)
                f2 = f2i(np.linspace(0,f2.size-1,npoints))
                
                m2i = interp1d(np.arange(m2.size),m2)
                m2 = m2i(np.linspace(0,m2.size-1,npoints))
                
            t = dist0/(dist0+dist1)
            
            mass_int = np.concatenate([mass_int, (1.-t)*m1+t*m2])
            f_int = np.concatenate([f_int, (1.-t)*f1+t*f2 ])
            

        photint.append(f_int)

    # keep mass field for future use
    photint.append(mass_int)

##########################################################
    if nointerp:
        # get the closest isochrone to the given age and Z
        #apply absolute mag cut if set
        if(Abscut):
            iso = mod_grid[(mod_grid['Age'] == age_grid[ind_age[0]]) & 
                           (mod_grid['Zini'] == z_grid[ind_z[0]]) & 
                           (mod_grid[refMag] < Abscut)]
        else:
            iso = mod_grid[(mod_grid['Age'] == age_grid[ind_age[0]]) &
                          (mod_grid['Zini'] == z_grid[ind_z[0]])]
            
        photint = []

        for filter in bands:
            photint.append(iso[filter])
        photint.append(iso['Mini'])
##########################################################
        
    
    cols = bands[:]
    cols.append('Mini')
    
    return np.core.records.fromarrays(photint, names=cols)
    
###############################################
# function to get CCM model coeficients
# for the GAIA filters we used: https://arxiv.org/pdf/1008.0815.pdf

def ccm_coefs(band):
    # CCM coefficients revised by O'Donnell (1994)
    dict_94 = {'Bmag': [4460.62, 0.9999253931841896, 0.94553192328962365],
            'Hmag': [16369.53, 0.2596053235545497, -0.23834844166071026],
            'Imag': [8036.57, 0.76735566136864775, -0.51126852210308293],
            'Jmag': [12314.46, 0.4105283397212145, -0.37691364988341475],
            'Kmag': [21937.18, 0.16203573610362962, -0.14876800161430803],
            'Rmag': [6557.09, 0.90991266273182836, -0.29970863780329793],
            'Umag': [3641.89, 0.96420188342499813, 1.784213363585738],
            'Vmag': [5501.7, 0.99974902186052628, -0.0046292182005786527],
            'B_Tmag': [4350.0, 1.0017962252392263, 1.0362801277999945],
            'V_Tmag': [5050.0, 1.0044713895752799, 0.36691338631832937],
            'G_BPmag': [5320.0, 1.0042005025756102, 0.12595552528038209],
            'G_RPmag': [7970.0, 0.77261277538121242, -0.49646921986943959],
            'Gmag': [6730.0, 0.89044911179059461, -0.31133471454740169]}
    
    # Original Cardelli, Clayton, and Mathis (1989 ApJ. 345, 245)
    dict_ccm_ori = {'Bmag': [4460.62, 1.0025759394309195, 0.92908077063668137],
                    'Hmag': [16369.53, 0.2596053235545497, -0.23834844166071026],
                    'Imag': [8036.57, 0.77833673604251075, -0.57683220088641463],
                    'Jmag': [12314.46, 0.4105283397212145, -0.37691364988341475],
                    'Kmag': [21937.18, 0.16203573610362962, -0.14876800161430803],
                    'Rmag': [6557.09, 0.90937658249478737, -0.28122644122534407],
                    'Umag': [3641.89, 0.95941873606501926, 1.8578498871346709],
                    'Vmag': [5501.7, 0.99957590813034558, -0.0033509151940263101],
                    'B_Tmag': [4350.0, 0.99736986014984541, 1.0711315721870396],
                    'V_Tmag': [5050.0, 1.015771886403408, 0.28589221136305393],
                    'G_BPmag': [5320.0, 1.0087722119558464, 0.092674041614019612],
                    'G_RPmag': [7970.0, 0.78420684534377993, -0.56542798221110957],
                    'Gmag': [6730.0, 0.89312322728995797, -0.31352377819769739]}

    return dict_94[band]
   
###############################################

def gaia_ext_coefs(band):

    dict_gaia = {'G_BPmag': [1.1517,-0.0871,-0.0333,0.0173,-0.0230,0.0006,0.0043],
                 'G_RPmag': [0.6104,-0.0170,-0.0026,-0.0017,-0.0078,0.00005,0.0006],
                 'Gmag': [0.9761,-0.1704,0.0086,0.0011,-0.0438,0.0013,0.0099]}
    return dict_gaia[band]

###############################################
def gaia_ext_Hek(color, Av,band):
    
    # polynomial values for FITZPATRICK & MASSA (2007) and Evans+, (2018) revised bands
#    if (band == 'G_BPmag'):
#        c0_0, c1_0, c2_0, c3_0 = 1.0283865, 0.03007388, 0.00487814, -0.00047627 
#        c0_1, c0_2, c0_3 =-0.09923966, 0.0270383, 0.00574691
#        c1_1, c1_2, c2_1 =-0.02490741, -0.00601982, 0.00283257
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
#    
#    if (band == 'G_RPmag'):
#        c0_0, c1_0, c2_0, c3_0 = 0.58664197, 0.00367414, -0.00002256, 0.00038784
#        c0_1, c0_2, c0_3 = -0.02428025, -0.00372679, -0.00345797
#        c1_1, c1_2, c2_1 = 0.00120703, 0.00549521, -0.00260023
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
#            
#    if (band == 'Gmag'):
#        c0_0, c1_0, c2_0, c3_0 = 0.78389355, 0.03531745, 0.00171237, -0.00014117
#        c0_1, c0_2, c0_3 = -0.14335868, 0.01376157, 0.00494197
#        c1_1, c1_2, c2_1 = -0.01106363, -0.00361002, 0.00121742
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
#            

    # polynomial values for FITZPATRICK & MASSA (2007) and Weiler+, (2018) revised bands
    if (band == 'G_BPmag'):
        c0_0, c1_0, c2_0, c3_0 = 1.02327769, 0.0270509, 0.00502619, -0.00046956 
        c0_1, c0_2, c0_3 =-0.09041839, 0.02504953, 0.00400657
        c1_1, c1_2, c2_1 =-0.02440485, -0.00460445, 0.00252245
        
        k = (c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2)
    
    if (band == 'G_RPmag'):
        c0_0, c1_0, c2_0, c3_0 = 0.58184356, 0.00385562, 0.00008391, 0.00021834
        c0_1, c0_2, c0_3 = -0.02412338, -0.00198959, -0.00131952
        c1_1, c1_2, c2_1 = 0.00042228, 0.00260914, -0.00137943
        
        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
            
    if (band == 'Gmag'):
        c0_0, c1_0, c2_0, c3_0 = 0.78510666, 0.03356769, 0.00209894, -0.00005242
        c0_1, c0_2, c0_3 = -0.13439024, 0.01637635, 0.00178928
        c1_1, c1_2, c2_1 = -0.01353545, -0.00071825, 0.0003951
        
        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
            

            
#    # polynomial values for O’Donnell (1994) with Rv=3.1 and Weiler+, (2018) revised bands
#    if (band == 'G_BPmag'):
#        c0_0, c1_0, c2_0, c3_0 = 1.03751421, 0.02633363, 0.00330665, -0.00055461
#        c0_1, c0_2, c0_3 =-0.09782787, 0.02467949, 0.00834803
#        c1_1, c1_2, c2_1 =-0.01948578, -0.0091578, 0.00390453
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
#    
#    if (band == 'G_RPmag'):
#        c0_0, c1_0, c2_0, c3_0 = 0.62881907, 0.00244174, 0.00003922, 0.00031353
#        c0_1, c0_2, c0_3 = -0.02342176, -0.00309773, -0.00354599
#        c1_1, c1_2, c2_1 = 0.00059264, 0.0052238, -0.00226387
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
#            
#    if (band == 'Gmag'):
#        c0_0, c1_0, c2_0, c3_0 = 0.81784342, 0.02914206, 0.00153169, -0.00007233
#        c0_1, c0_2, c0_3 = -0.1285682, 0.01562673, 0.00301747
#        c1_1, c1_2, c2_1 = -0.01122163, -0.00185401, 0.00066998
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
            
    
    # polynomial values for Wang (2019) with Rv=3.1 and Weiler+, (2018) revised bands
#    if (band == 'G_BPmag'):
#        c0_0, c1_0, c2_0, c3_0 = 1.02097048, 0.01365521, 0.00503549, 0.00003211
#        c0_1, c0_2, c0_3 =-0.06089457, 0.02604143, -0.0028939
#        c1_1, c1_2, c2_1 =-0.02506906, 0.00363858, -0.00088837
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
#    
#    if (band == 'G_RPmag'):
#        c0_0, c1_0, c2_0, c3_0 = 0.59018567, 0.00593959, -0.00039767, 0.00001157
#        c0_1, c0_2, c0_3 = -0.02796109, -0.00254418, 0.00068733
#        c1_1, c1_2, c2_1 = 0.00158512, -0.00043657, 0.00005208
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
#            
#    if (band == 'Gmag'):
#        c0_0, c1_0, c2_0, c3_0 = 0.78392579, 0.0301596, 0.00187428, 0.00012547
#        c0_1, c0_2, c0_3 = -0.12131921, 0.01973744, -0.00146647
#        c1_1, c1_2, c2_1 = -0.01561832, 0.00238119, -0.00069678
#        
#        k = c0_0 + c1_0*Av + c2_0*Av**2 + c3_0*Av**3 + c0_1*color + c0_2*color**2 + \
#            c0_3*color**3 + c1_1*Av*color + c1_2*Av*color**2 + c2_1*color*Av**2
            

    return k
    
###############################################
# Make an observed synthetic cluster given an isochrone,
# distance, E(B-V) and Rv
# Al/Av = a + b/rv
# Av = rv*ebv

def make_obs_iso(bands, iso, dist, Av, gaia_ext = False):
    #redden and move isochrone
    obs_iso = np.copy(iso)
    
    color = iso['G_BPmag'] - iso['G_RPmag']

    for filter in bands:
        
        if gaia_ext:
            # get coeficients
#            c1, c2, c3, c4, c5, c6, c7 = gaia_ext_coefs(filter)
#            AloAv = c1 + c2*color + c3*color**2 + c4*color**3 + c5*Av + c6*Av**2 + c7*color*Av
            
            AloAv = gaia_ext_Hek(color, Av,filter)
            
            # apply correction
            obs_iso[filter] = iso[filter] + 5.*np.log10(dist*1.e3) - 5.+ AloAv*Av
            
        else:
            # get CCm coeficients
            wav,a,b = ccm_coefs(filter)
        
            # apply ccm model and make observed iso
            obs_iso[filter] = iso[filter] + 5.*np.log10(dist*1.e3) - 5.+ ( (a + b/3.1)*Av )
        
    return obs_iso


    
###############################################
# add some binaries
# generate probability of being binary

def add_binaries(bin_frac,isoc,isoc_bin,bands,refMag,imf='chabrier',alpha=2.3,
                 beta=-3,seed=None,binratio=0.8):
    
    if(seed != None): 
        r = np.random.RandomState(seed+1)
    else:
        r = np.random.RandomState(seed)

    prob_bin = r.rand(isoc.size)
    nbin = isoc[prob_bin < bin_frac].size
    isoc = np.copy(isoc)
    isoc_bin = np.copy(isoc_bin)
    
#    masses = binratio*isoc['Mini'][prob_bin < bin_frac]
#    masses = r.beta(35,10.,nbin)*isoc['Mini'][prob_bin < bin_frac]
#    masses = r.rand(isoc.size)*isoc['Mini']
#    masses = r.uniform(0,1,nbin)*isoc['Mini'][prob_bin < bin_frac]


#    lower, upper = 0., 1.
#    mu, sigma = 1., 0.05
#    fac = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)    
#    masses = fac.rvs(nbin)*isoc['Mini'][prob_bin < bin_frac]
    
#    indlow = np.where(isoc['Mini'][prob_bin < bin_frac] < 1.5)
#    indhi = np.where(isoc['Mini'][prob_bin < bin_frac] >= 3.)
#    indmid = np.where((isoc['Mini'][prob_bin < bin_frac] >= 1.5) & (isoc['Mini'][prob_bin < bin_frac] < 3))
#    masses=np.zeros(nbin)
#    masses[indlow] = salpeter(0.01,indlow[0].size,0.01,1.,seed)*isoc['Mini'][prob_bin < bin_frac][indlow]
#    masses[indhi] = salpeter(0.15,indhi[0].size,0.01,1.,seed)*isoc['Mini'][prob_bin < bin_frac][indhi]
#    masses[indmid] = salpeter(0.45,indmid[0].size,0.01,1.,seed)*isoc['Mini'][prob_bin < bin_frac][indmid]
#
#    bin_comp_stars = sample_from_isoc(isoc_bin,bands,refMag,nbin,imf=imf,alpha=alpha,
#                                      beta=beta, seed=seed,binmasses=masses)

    bin_comp_stars = sample_from_isoc(isoc_bin,bands,refMag,nbin,imf=imf,alpha=alpha,
                                      beta=beta,seed=seed)
    

    for filter in bands:
        m1 = isoc[filter][prob_bin < bin_frac]
        m2 = bin_comp_stars[filter]
        mcomb = m1 - 2.5 * np.log10(1.0 + 10**(-0.4*(m2-m1)))
        isoc[filter][prob_bin < bin_frac] = mcomb
        
    return isoc



###############################################
# add some errors
# IMPORTANT: the error coeficients for J,H, K are just copies of Imag 
# since we dont have data for them

def add_phot_errors(isoc,bands):
    # gaia errors from https://www.cosmos.esa.int/web/gaia/science-performance
    
    er_isoc = np.copy(isoc)
    gaia_bands = ['Gmag','G_BPmag','G_RPmag']
    
    er_coefs = {'Umag': [0.03093, 3.93086E-13, 0.79563],
                'Bmag': [0.03078, 6.92477E-13, 0.84029],
                'Vmag': [0.02869, 4.70556E-12, 0.86897],
                'Rmag': [0.02400, 3.77483E-12, 0.82648],
                'Imag': [0.02831, 8.05872E-13, 0.73068],
                'Jmag': [0.02831, 8.05872E-13, 0.73068],
                'Hmag': [0.02831, 8.05872E-13, 0.73068],
                'Kmag': [0.02831, 8.05872E-13, 0.73068],
                'B_Tmag': [0.03078, 6.92477E-13, 0.84029],
                'V_Tmag': [0.02869, 4.70556E-12, 0.86897]}

                
    for filter in bands:
        
        if filter == 'Gmag':
            z=[]
            for mag in isoc[filter]:
                z.append(np.max([10**0.4*(12 - 15), 10**0.4*(mag - 15.)]))
            er = 1.e-3*np.sqrt(np.abs(0.04895*np.array(z)**2 + 1.8633*np.array(z) + 0.0001985))
                        
            er_isoc[filter] = isoc[filter] * np.random.normal(1., np.abs(er/isoc[filter]), isoc.size)
            
        elif (filter == 'G_BPmag' or filter == 'G_RPmag'):
            # WARNING !!!! this needs to be updated to correct formulas!!!
            z=[]
            for mag in isoc['Gmag']:
                z.append(np.max([10**0.4*(11 - 15), 10**0.4*(mag - 15.)]))
            er = 1.e-3*np.sqrt(np.abs(0.04895*np.array(z)**2 + 1.8633*np.array(z) + 0.0001985))
            er_isoc[filter] = isoc[filter] * np.random.normal(1., np.abs(er/isoc[filter]), isoc.size)
            
        else:
            er = er_coefs[filter][0] +  er_coefs[filter][1] * np.exp(isoc[filter]/er_coefs[filter][2])
            er[er > 0.5*3.] = 0.5*3.        
            er_isoc[filter] = isoc[filter] * np.random.normal(1., np.abs(er/isoc[filter])/3., isoc.size)
    
        
    return er_isoc

###############################################
# return the vector of errors
# IMPORTANT: the error coeficients for J,H, K are just copies of Imag 
# since we dont have data for them

def get_phot_errors(isoc,bands):
    
    errors = np.copy(isoc)
    
    er_coefs = {'Umag': [0.03093, 3.93086E-13, 0.79563],
                'Bmag': [0.03078, 6.92477E-13, 0.84029],
                'Vmag': [0.02869, 4.70556E-12, 0.86897],
                'Rmag': [0.02400, 3.77483E-12, 0.82648],
                'Imag': [0.02831, 8.05872E-13, 0.73068],
                'Jmag': [0.02831, 8.05872E-13, 0.73068],
                'Hmag': [0.02831, 8.05872E-13, 0.73068],
                'Kmag': [0.02831, 8.05872E-13, 0.73068],
                'B_Tmag': [0.03078, 6.92477E-13, 0.84029],
                'V_Tmag': [0.02869, 4.70556E-12, 0.86897]}
                
    for filter in bands:
        
        if filter == 'Gmag':
            z=[]
            for mag in isoc[filter]:
                z.append(np.max([10**0.4*(12 - 15), 10**0.4*(mag - 15.)]))
            er = 1.e-3*np.sqrt(0.04895*np.array(z)**2 + 1.8633*np.array(z) + 0.0001985)
            
        elif (filter == 'G_BPmag' or filter == 'G_RPmag'):
            # WARNING !!!! this needs to be updated to correct formulas!!!
            z=[]
            for mag in isoc['Gmag']:
                z.append(np.max([10**0.4*(12 - 15), 10**0.4*(mag - 15.)]))
            er = 1.e-3*np.sqrt(0.04895*np.array(z)**2 + 1.8633*np.array(z) + 0.0001985)
            
        else:
            er = er_coefs[filter][0] +  er_coefs[filter][1] * np.exp(isoc[filter]/er_coefs[filter][2])
            er[er > 0.5*3.] = 0.5*3.        
    
        
    return er


###############################################
# king profile: y = Nbg +(Nc/(1+(R/Rcore)^2)) 
# where rcore is the cluster core radius, Ncore is the core density,
# and Nbg the background density level, respectively. The core radius 
# was defined as a distance where density drops to half of Ncore. 
    
def king_prof(radius, rcore, Ncore, Nbg):
    '''
    Three parameters King profile fit.
    '''
    return Nbg + Ncore / (1.0 + (radius/rcore)**2)


###############################################
# Sample from King profile to get radius vector
    
def sample_king(nstars, rcore, Ncore, mass):
    
    rad_int = np.arange(0.,2.5*rcore,2.5*rcore/100.)
    king_val = king_prof(rad_int, rcore, Ncore, 0.)
    
    #normalize
    king_norm =  king_val / trapz(king_val,rad_int)
    
    # get cumulative distribution
    cum_king = cumtrapz(king_norm,rad_int, initial=0)
    
    # sample from King profile
    gen_radius = ( (interp1d(cum_king,rad_int))(np.random.rand(nstars)) )
            
    plt.plot(mass,gen_radius,'o')

    return np.array(gen_radius)

###############################################
# Sample from King profile to get radius vector
    
def sample_king_test(nstars, rcore, Ncore, mass):
    
    # define 4 zones to sample radius
    rout = 2.*rcore
    rad_int_1 = np.arange(0.,rout*0.5,rout*0.5/100.)
    rad_int_2 = np.arange(0.,rout*0.8,rout*0.8/100.)
    rad_int_3 = np.arange(0.,rout,rout/100.)
    rad_int_4 = np.arange(0.,rout,rout/100.)
        
    Mmin, Mmax = np.min(mass), np.max(mass)
    
    gen_radius=[]
    
    dm = (Mmax-Mmin)/4.
    for m in mass:
        
        if (m>=Mmin and m<Mmin+dm):
            king_val = king_prof(rad_int_4, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapz(king_val,rad_int_4)
            # get cumulative distribution
            cum_king = cumtrapz(king_norm,rad_int_4, initial=0)
            # sample from King profile
            gen_radius.append( (interp1d(cum_king,rad_int_4))(np.random.rand()) )
            
        if (m>=Mmin+dm and m<Mmin+2*dm):
            king_val = king_prof(rad_int_3, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapz(king_val,rad_int_3)
            # get cumulative distribution
            cum_king = cumtrapz(king_norm,rad_int_3, initial=0)
            # sample from King profile
            gen_radius.append( (interp1d(cum_king,rad_int_3))(np.random.rand()) )
            
        if (m>=Mmin+2*dm and m<Mmin+3*dm):
            king_val = king_prof(rad_int_2, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapz(king_val,rad_int_2)
            # get cumulative distribution
            cum_king = cumtrapz(king_norm,rad_int_2, initial=0)
            # sample from King profile
            gen_radius.append( (interp1d(cum_king,rad_int_2))(np.random.rand()) )
            
        if (m>=Mmin+3*dm and m<Mmin+4*dm or m==Mmax):
            king_val = king_prof(rad_int_1, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapz(king_val,rad_int_1)
            # get cumulative distribution
            cum_king = cumtrapz(king_norm,rad_int_1, initial=0)
            # sample from King profile
            aux = (interp1d(cum_king,rad_int_1))(np.random.rand()) 
            gen_radius.append( aux )

    return np.array(gen_radius)

##############################################################################################
# convert radius vector in arcmin coordinate pair 
# in decimal degree assming angular symetry
    
def gen_cluster_coordinates(ra0,dec0,nstars, rcore, Ncore,mass):
    
    # sample radius from King profile
    radius = sample_king_test(nstars, rcore, Ncore, mass)*u.arcmin
    
    #convert arcmin to degree
    radius = radius.to(u.deg)
    
    
    # get uniform distribution of angle [0,2Pi]
    phi = np.random.uniform(0,2*np.pi,nstars)
    
    # convert (r,phi) to (ra,dec)
    ra = radius*np.cos(phi) + ra0*u.deg
    dec = radius*np.sin(phi) + dec0*u.deg
    
    return ra.value,dec.value


###############################################
# generate a sample of field star coordinates
# assuming uniform distribution and given a field size
    # in arcmin
    
def gen_field_coordinates(ra0,dec0,field_sz, nstars):
    
    field_sz = (field_sz*u.arcmin).to(u.deg)
    ra0, dec0 = ra0*u.deg, dec0*u.deg

    ra = np.random.uniform((ra0-field_sz/2.).value,(ra0+field_sz/2.).value,nstars)
    dec = np.random.uniform((dec0-field_sz/2.).value,(dec0+field_sz/2.).value,nstars)

    return ra,dec


###############################################
# generate a synthetic cluster
    
def model_cluster(age,dist,FeH,Av,bin_frac,nstars,bands,refMag,Mcut=False,error=True,
                  seed=None,Abscut=False,imf='chabrier',alpha=2.3,beta=-3):
    
    # check if binarity is ok
    if( bin_frac < 0. or bin_frac > 1.):
        #print 'binarity out of intervl, setting it to 1...'
        bin_frac = 1.
    
    # check if distance is ok    
    if( dist < 0.01 ):
        dist = 0.01
        
    # check age is ok    
    if( age < 6.65):
        age = 6.65
    elif (age>10.3):
        age = 10.3
    
    # check metalicity is ok    
    if( FeH < -0.9):
        FeH = -0.9
    elif (FeH>0.7):
        FeH = 0.7
    
    # check Av is ok    
    if( Av < 0.01):
        Av = 0.01
    elif (Av>6.0):
        Av = 6.0
    
    
    # get isochrone
    met = (10.**FeH)*0.0152
    grid_iso = get_iso_from_grid(age,met,bands,refMag,Abscut=Abscut)

    # make an observed isochrone
    obs_iso = make_obs_iso(bands, grid_iso, dist, Av, gaia_ext = True)

#    use_imf = 'salpeter' 
#    use_imf = 'chabrier' 
    #use_imf = 'MillerScalo' 
#    use_imf = 'deMarchi'

    #sample from isochrone
    gen_iso = sample_from_isoc(obs_iso,bands,refMag,nstars,imf=imf,alpha=alpha, 
                               beta=beta, Mcut=Mcut,seed=seed)
    gen_iso_bin = sample_from_isoc(obs_iso,bands,refMag,nstars,imf=imf,alpha=alpha, 
                               beta=beta,Mcut=False,seed=seed)

    # add some binaries
    gen_iso = add_binaries(bin_frac,gen_iso,gen_iso_bin,bands,refMag,imf=imf,
                           alpha=alpha,beta=beta,seed=seed,binratio=0.8)

    # add some errors
    if error:
        gen_iso = add_phot_errors(gen_iso,bands)
        gen_iso_er = get_phot_errors(gen_iso,bands)


    return gen_iso

##############################################################################
# Define the log likelihood
# theta -> vector of model parameters [age, dist, z, ebv, Rv]
# 
def lnlikelihood(theta,obs_iso,obs_iso_er,bands,refMag,prange,weight,prior=[[1.],[1.]],seed=None):
    
    # generate synth cluster from input parameters
    age, dist, FeH, Av = theta
    bin_frac = 0.5
    # number of stars to generate
    nstars = obs_iso.size
    Mlim = obs_iso[refMag].max()
    
    if (age<prange[0,0] or age>prange[0,1] or dist<prange[1,0] or dist>prange[1,1] 
        or FeH<prange[2,0] or FeH>prange[2,1] or Av<prange[3,0] or Av>prange[3,1]):
        return -np.inf
    
    else:
        # get synth isochrone
        mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,2000,bands,
                                    refMag,error=False,Mcut=Mlim,seed=seed,
                                    imf='chabrier',alpha=2.1, beta=-3.)
        
        # get distance of each observed star to the model isochrone
        obs = np.array(obs_iso[bands].tolist())
        obs_er = np.array(obs_iso_er[bands].tolist())
        mod = np.array(mod_cluster[bands].tolist())
                
        p_iso = []
        for i in range(nstars):
    
            aux = np.prod(1./np.sqrt(2.*np.pi*obs_er[i,:]**2)*
                          np.exp(-0.5*(obs[i,:]-mod)**2/obs_er[i,:]**2),axis=1)

#            aux = np.prod(1./np.sqrt(2.*np.pi)*
#                          np.exp(-0.5*(obs[i,:]-mod)**2),axis=1)

            p_iso.append(np.max(aux))
    
        p_iso = np.array(p_iso)*np.prod(np.exp(-0.5*( (theta-prior[0,:])/prior[1,:] )**2))
            
        p_iso[p_iso< 1.e-307] = 1.e-307
        res = np.log(p_iso) + np.log(weight) 
        
        return np.mean(res)
            

##############################################################################
##############################################################################
# Define the log likelihood
# theta -> vector of model parameters [age, dist, z, ebv, Rv]
# 
def lnlikelihoodCE(theta,obs_iso,obs_iso_er,bands,refMag,prange,weight,prior=[[1.],[1.e3]],seed=None):
    
    # generate synth cluster from input parameters
    age, dist, FeH, Av = theta
    
    bin_frac = 0.5

    # number of stars to generate
    nstars = obs_iso.size
    Mlim = obs_iso[refMag].max()
    
    # get synth isochrone

    mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,2000,bands,
                                refMag,error=False,Mcut=Mlim,seed=seed,
                                imf='chabrier',alpha=2.1, beta=-3.)

    # get distance of each observed star to the model isochrone
    obs = np.array(obs_iso[bands].tolist())
    obs_er = np.array(obs_iso_er[bands].tolist())

    mod = np.array(mod_cluster[bands].tolist())

    p_iso = []
    for i in range(nstars):

#        aux = np.prod(1./np.sqrt(2.*np.pi)*
#                      np.exp(-0.5*(obs[i,:]-mod)**2),axis=1)
        
        aux = np.prod(1./np.sqrt(2.*np.pi*obs_er[i,:]**2)*
                      np.exp(-0.5*(obs[i,:]-mod)**2/obs_er[i,:]**2),axis=1)
        
#        aux = np.prod(np.exp(-0.5*np.abs(obs[i,:]-mod)),axis=1)

        p_iso.append(np.max(aux))

    p_iso = np.array(p_iso)*np.prod(np.exp(-0.5*( (theta-prior[0,:])/prior[1,:] )**2))
        
    p_iso[p_iso< 1.e-307] = 1.e-307   
    res = np.log(p_iso) + np.log(weight)
    res = -np.mean(res)
    
    #print res, p_iso.max(), p_iso.min() #'   '.join('%0.3f' % v for v in theta)
    
    return res





    
