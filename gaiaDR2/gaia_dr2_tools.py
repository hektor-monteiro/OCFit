#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:34:43 2018

@author: hmonteiro
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import timeit
import multiprocessing as mp
from scipy.spatial.distance import cdist
from scipy import stats
import time
from astroquery.vizier import Vizier
import astropy.units as u
import os
import warnings
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.coordinates import Angle
from oc_tools_padova import *
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker,TextArea, DrawingArea
import corner
import emcee
from astropy.stats import mad_std
from scipy.optimize import least_squares,minimize
from math import ceil
from scipy import linalg

##########################################################################################
def find_TO(xcolor,ymag,box=0.2):
    magbins = np.linspace(ymag.min(),ymag.max(),int((ymag.max()-ymag.min())/box))
    colorbins = magbins*0.
    dcolor = magbins*0.
    for i in range(magbins.size):
        colorbins[i] = np.nanmedian(xcolor[np.abs(ymag-magbins[i]) < box])
        dcolor[i] = np.nanstd(xcolor[np.abs(ymag-magbins[i]) < box])
        
    return colorbins, magbins, dcolor
    
##########################################################################################
def lowess_ag(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest
##########################################################################################
# hill function to perform smooth transitions
def hill(x,p,h):
    return  x**p/(x**p + h**p)

##########################################################################################
def gather_results(outfile,results_dir,overwrite=True):
    
    dirs = os.listdir(results_dir)
    dirs = [s for s in dirs if os.path.isdir(results_dir+s)]
    
    if overwrite:
        logfile = open(results_dir+outfile, "w")
    else:
        logfile = open(results_dir+outfile, "a")
    logfile.write(time.strftime("%c")+'\n')
    logfile.write(' \n')
    logfile.write('             name;       rah;   ram;   ras;        deg;    dem;    decs;    RA_ICRS;     DE_ICRS;    radius;   crad;     dist;     e_dist;    age;     e_age;     FeH;    e_FeH;     Av;      e_Av;      AG;      e_AG;       Nc;      Plx;     sigPlx;   e_Plx;    pmRA;     sigpmRA;  e_pmRA;   pmDE;     sigpmDE;  e_pmDE;   Vr;      e_Vr;\n ')
    for folder in dirs:
        f = open(results_dir+folder+'/results_'+folder+'.txt', "r")
        aux=f.readlines()
        try:
            logfile.write(aux[3])
        except:
            continue
    
    logfile.close()
    
    print('Done consolidating log file...')
        
##########################################################################################
def likelihood_dist(dist,plx, eplx):
    return -np.sum(np.log(1/np.sqrt(2*np.pi)/eplx * 
                         np.exp( -0.5*(plx - (1/dist))**2/eplx**2. )))

##########################################################################################
def infer_dist(plx, erplx, guess=1.):
    res = minimize(likelihood_dist, guess, method='Nelder-Mead', tol=1e-6, 
                   args=(plx,erplx))
    return res.x[0]
    

##########################################################################################
class AnchoredText(AnchoredOffsetbox):
    def __init__(self, s, loc, pad=0.4, borderpad=0.5, prop=None, frameon=True):

        self.txt = TextArea(s,
                            minimumdescent=False)


        super(AnchoredText, self).__init__(loc, pad=pad, borderpad=borderpad,
                                           child=self.txt,
                                           prop=prop,
                                           frameon=frameon)
##########################################################################################
def gaussian(x, mu, sig):
    return 1./np.sqrt(2*np.pi)/sig * np.exp( -0.5*(x - mu)**2/(2 * sig**2.) )

##########################################################################################
def smoothclamp(x, mi, mx): 
    return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )
##########################################################################################
# calculate spatial density given RA and DEC of stars
    
def star_density(ra,dec,ra_ref,dec_ref):
    
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [1,1]
    w.wcs.cdelt = np.array([1./60,1./60])
    w.wcs.crval = [ra_ref,dec_ref]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    
    sky_coords = SkyCoord(ra, dec, unit="deg")
    cluster_center = SkyCoord(ra_ref, dec_ref, unit="deg")
    pix_coords = sky_coords.to_pixel(w)
    
    # calculate kernel density with sig_pix bandwidth    
    values = np.vstack([pix_coords[0], pix_coords[1]])
    kernel_dens = stats.gaussian_kde(values,bw_method='silverman')
    dens_map = kernel_dens(values)
    
    # get distance from density max center
    ang_sep_peak = sky_coords.separation(SkyCoord(ra[np.argmax(dens_map)],
                                                     dec[np.argmax(dens_map)],
                                                     unit="deg"))
    
    # get distance from cluster center
    ang_sep = sky_coords.separation(cluster_center)
    
    return pix_coords[0], pix_coords[1], dens_map, np.argmax(dens_map), ang_sep.to(u.arcmin)#, ang_sep_peak.to(u.arcmin)

##########################################################################################
# tri-variate normal distribution
def normal_3D(x,y,z,mu_x,mu_y,mu_z,sigx,sigy,sigz,rhoxy,rhoxz,rhoyz):

    md=[]
    for i in range(len(x)):
        C00 = sigx[i]**2
        C01 = sigx[i]*sigy[i]*rhoxy[i]
        C10 = sigx[i]*sigy[i]*rhoxy[i]
        C11 = sigy[i]**2
        C02 = sigx[i]*sigz[i]*rhoxz[i]
        C20 = sigx[i]*sigz[i]*rhoxz[i]
        C12 = sigy[i]*sigz[i]*rhoyz[i]
        C21 = sigy[i]*sigz[i]*rhoyz[i]
        C22 = sigz[i]**2
        
        cov_xyz = [[C00,C01,C02],
                   [C10,C11,C12],
                   [C20,C21,C22]]
        
        det_cov_xyz = np.linalg.det(cov_xyz)
        inv_cov_xyz = np.linalg.inv(cov_xyz)
    
        x_diff = (x[i] - mu_x)
        y_diff = (y[i] - mu_y)
        z_diff = (z[i] - mu_z)
    
        diff_xyz = np.transpose([x_diff, y_diff, z_diff])
    
        t0 = 1./np.sqrt((2.*np.pi)**3 * det_cov_xyz)
        t1 = np.dot(np.transpose(diff_xyz),inv_cov_xyz)
        t2 = np.dot(t1,diff_xyz)    
    
        md.append( t0 * np.exp(-0.5 * t2) )
    #print 'saiu...'
    return np.array(md)
    

##########################################################################################
# 
def likelihood_3D(theta,data):
    
    pmRA,pmDE,Plx,erRA,erDE,erPlx,pmRApmDEcor,PlxpmRAcor,PlxpmDEcor,weight = data
    
    Nc,pmRA_c,pmDE_c,plx_c,sigRA_c,sigDE_c,sigplx_c, \
       pmRA_f,pmDE_f,plx_f,sigRA_f,sigDE_f,sigplx_f = theta
    
    # for the cluster
    sigx = np.sqrt(sigRA_c**2+erRA**2)
    sigy = np.sqrt(sigDE_c**2+erDE**2)
    sigz = np.sqrt(sigplx_c**2+erPlx**2)
    
    rhoxy = pmRApmDEcor
    rhoxz = PlxpmRAcor
    rhoyz = PlxpmDEcor
    
    phi_c = Nc*normal_3D(pmRA,pmDE,Plx,pmRA_c,pmDE_c,plx_c,sigx,sigy,sigz,rhoxy,rhoxz,rhoyz)
    
    # for the field
    sigx = np.sqrt(sigRA_f**2+erRA**2)
    sigy = np.sqrt(sigDE_f**2+erDE**2)
    sigz = np.sqrt(sigplx_f**2+erPlx**2)
    
    rhoxy = pmRApmDEcor
    rhoxz = PlxpmRAcor
    rhoyz = PlxpmDEcor
    
    phi_f = (1.-Nc)*normal_3D(pmRA,pmDE,Plx,pmRA_f,pmDE_f,plx_f,sigx,sigy,sigz,rhoxy,rhoxz,rhoyz)
    
    phi = phi_c + phi_f
    phi[phi<1.e-16] = 1.e-30
    phi[~np.isfinite(phi)] = 1.e-30
    
    if ( (sigRA_c**2+sigDE_c**2) > (sigRA_f**2+sigDE_f**2) ):
        phi = phi*0.

    return -np.sum(np.log(phi))

##########################################################################################
# 
def membership_3D(theta,data,just_oc = False):
    
    pmRA,pmDE,Plx,erRA,erDE,erPlx,pmRApmDEcor,PlxpmRAcor,PlxpmDEcor,weight = data
    
    Nc,pmRA_c,pmDE_c,plx_c,sigRA_c,sigDE_c,sigplx_c, \
       pmRA_f,pmDE_f,plx_f,sigRA_f,sigDE_f,sigplx_f = theta
    
    # for the cluster
    sigx = np.sqrt(sigRA_c**2+erRA**2)
    sigy = np.sqrt(sigDE_c**2+erDE**2)
    sigz = np.sqrt(sigplx_c**2+erPlx**2)
    
    rhoxy = pmRApmDEcor
    rhoxz = PlxpmRAcor
    rhoyz = PlxpmDEcor
    
    phi_c = Nc*normal_3D(pmRA,pmDE,Plx,pmRA_c,pmDE_c,plx_c,sigx,sigy,sigz,rhoxy,rhoxz,rhoyz)
    
    # for the field
    sigx = np.sqrt(sigRA_f**2+erRA**2)
    sigy = np.sqrt(sigDE_f**2+erDE**2)
    sigz = np.sqrt(sigplx_f**2+erPlx**2)
    
    rhoxy = pmRApmDEcor
    rhoxz = PlxpmRAcor
    rhoyz = PlxpmDEcor
    
    phi_f = (1.-Nc)*normal_3D(pmRA,pmDE,Plx,pmRA_f,pmDE_f,plx_f,sigx,sigy,sigz,rhoxy,rhoxz,rhoyz)
    
    phi = phi_c + phi_f
    phi[phi<1.e-16] = 1.e-30
    phi[~np.isfinite(phi)] = 1.e-30
    
    if ( (sigRA_c**2+sigDE_c**2) > (sigRA_f**2+sigDE_f**2) ):
        phi = phi*0.
        
    if just_oc:
        prob = phi_c/phi_c.max()
    else:
        prob = phi_c/(phi_c+phi_f)
        
    prob[~np.isfinite(prob)] = 0.
    
    return prob
##########################################################################################
# function to add columns to rec array

def add_col(array,col,col_type=('float', '<f8')):
    
    y=np.zeros(array.shape, dtype=array.dtype.descr+[col_type])    
    for name in array.dtype.names: y[name] = array[name]    
    y[col_type[0]]=col    
    return y

##########################################################################################
def PM_cluster_model(pars,data):
    
    pmRA,pmDE,erRA,erDE,pmRApmDEcor = data
    erRA,erDE = 3*erRA,3*erDE
    Nc,sigRA_c,sigDE_c,pmRA_c,pmDE_c,rho_c,sigRA_f,sigDE_f,pmRA_f,pmDE_f,rho_f= pars

    # CLUSTER
    t1 = 1./(2.*np.pi*np.sqrt(sigRA_c**2+erRA**2)*np.sqrt(sigDE_c**2+erDE**2)*np.sqrt(1.-pmRApmDEcor**2))
    t2 = 0.5/(1.-pmRApmDEcor**2)
    t3 = (pmRA-pmRA_c)/np.sqrt(sigRA_c**2+erRA**2)
    t4 = (pmDE-pmDE_c)/np.sqrt(sigDE_c**2+erDE**2)
    t5 = 2.*pmRApmDEcor * t3 * t4 
    phi_c = Nc * t1 * np.exp(-t2 * ( t3**2 + t4**2 - t5 ) )

    return phi_c 

##########################################################################################
def PM_field_model(pars,data):
    
    pmRA,pmDE,erRA,erDE,pmRApmDEcor = data
    erRA,erDE = 3*erRA,3*erDE
    Nc,sigRA_c,sigDE_c,pmRA_c,pmDE_c,rho_c,sigRA_f,sigDE_f,pmRA_f,pmDE_f,rho_f= pars

    # GAUSSIAN FIELD 
    t1 = 1./(2.*np.pi*np.sqrt(sigRA_f**2+erRA**2)*np.sqrt(sigDE_f**2+erDE**2)*np.sqrt(1.-(pmRApmDEcor+rho_f)**2))
    t2 = 0.5/(1.-(pmRApmDEcor+rho_f)**2)
    t3 = (pmRA-pmRA_f)/np.sqrt(sigRA_f**2+erRA**2)
    t4 = (pmDE-pmDE_f)/np.sqrt(sigDE_f**2+erDE**2)
    t5 = 2.*(pmRApmDEcor+rho_f) * t3 * t4 
    phi_f = (1.-Nc) * t1 * np.exp(-t2 * ( t3**2 + t4**2 - t5 ) )

    return phi_f

##########################################################################################
# Theta = data and model
def likelihood_PM(theta,data,weight):
    
#    pmRA,pmDE,erRA,erDE = data
    Nc,sigRA_c,sigDE_c,pmRA_c,pmDE_c,rho_c,sigRA_f,sigDE_f,pmRA_f,pmDE_f,rho_f = theta
    
    phi_c = PM_cluster_model(theta,data) 
    phi_f = PM_field_model(theta,data) 
    
    phi = phi_c + phi_f
        
    if ( (sigRA_c**2+sigDE_c**2) > (sigRA_f**2+sigDE_f**2) ):
        phi = phi*0. 

    return -np.mean(np.log(phi) + np.log(weight))

##########################################################################################
# Theta = data and model
def likelihood_Plx(theta,data, weight):
    
#    pmRA,pmDE,erRA,erDE = data
    Nc,plx_c,sig_c,plx_f,sig_f = theta
    
    phi_c = Nc * gaussian(data,plx_c,sig_c) 
    phi_f = (1. - Nc) * gaussian(data,plx_f,sig_f) 
    
    phi = phi_c + phi_f

    if (sig_c > sig_f):
        phi = phi*0.

    phi[phi < 0.] = 1.e-300

    return -np.mean(np.log(phi) + np.log(weight))

##########################################################################################
def fit_plx(obs,weight):  
    
    # filter data
    sig_clip = 6.
    cond1 = np.abs(obs-np.nanmean(obs)) < sig_clip*np.nanstd(obs)    
    obs = obs[cond1]
    weight = weight[cond1]

    print ('Starting parallax fitting...')
    
    # Define the parameter space
    prange = np.array([[0.1,0.9],
                       [0,obs.max()],
                       [0.001,0.05],
                       [obs.min(),obs.max()],
                       [0.1,obs.std()]])
    
    midpoint = (prange[:,1]-prange[:,0])/2.+prange[:,0]    
    ndim = prange.shape[0]
    
    # define CE tweak parameters
    nruns = 5
    itmax = 100    
    sample = 800
    
    band = 0.15
    alpha = 0.1
    tol = 1.e-40

    guess=False 
    
    res = np.ndarray([nruns,ndim])
    
    for n in range(nruns):
        res[n,:] = run_CE(likelihood_Plx,obs,weight,sample,prange,itmax,band,alpha,tol,
           mp.cpu_count()-1,guess)

    print ('')
    print ('-------------------------------------------------------------')
    print (' Final result')
    print ('-------------------------------------------------------------')
    print ('   '.join('%0.3f' % v for v in np.median(res,axis=0)))
    print ('   '.join('%0.3f' % v for v in res.std(axis=0)))
    print ('')
    print ('parallax fitting done...')

    
    return np.median(res,axis=0),np.std(res,axis=0)

##########################################################################################
def run_CE(objective,data,weight,sample,prange,itmax,bandwidth,alpha,tol,nthreads=1,guess=False,pm=False):
    
    # Define arrays to be used
    ndim = prange.shape[0]
    lik = np.zeros(sample)
    center = np.zeros([ndim,itmax])
    sigma = np.zeros([ndim,itmax])
    pars_best = 0
    avg_var = 1e3
    
    # generate initial solution population
    pars=[]
    for k in range(ndim):
        aux = np.random.uniform(prange[k,0],prange[k,1], sample)
        pars.append(aux)    
    pars=np.array(pars)
    
    if (guess != False):
        aux = []
        for k in range(ndim):
            aux.append(np.random.uniform(0.9*guess[k],1.1*guess[k], int(0.1*sample)))
        pars[:,0:int(0.1*sample)] = np.array(aux)    
        

    iter = 0
    
    while (iter < itmax):
        
##########################################################################################
        pool = mp.Pool(processes=mp.cpu_count()-1)
        res = [pool.apply_async(objective, args=(theta,data,weight,)) for theta in pars.T]
        lik = np.array([p.get() for p in res])
        pool.close()
        pool.join()
###########################################################################################

        # sort solution in descending likelihood
        
        ind = np.argsort(lik)

        # best solution in iteration
        pars_best = np.copy(pars[:,ind[0]])
        lik_best = lik[ind[0]]
        
        # indices of band best solutions
        ind_best = ind[0:int(bandwidth*sample)]
    
        # discard indices that are out of parameter space
        ind_best = ind_best[np.isfinite(lik[ind_best])]
        
        # calculate new proposal distribution
        if (iter == 0):
            center[:,iter] = np.nanmean(pars[:,:],axis=1)
            sigma[:,iter] = np.nanstd(pars[:,:],axis=1)
        else:
            sigma_ori = sigma
            center[:,iter] = alpha*np.nanmean(pars[:,ind_best],axis=1) + (1.-alpha)*center[:,iter-1]
            sigma[:,iter] = alpha*np.nanstd(pars[:,ind_best],axis=1) + (1.-alpha)*sigma[:,iter-1]
            if pm:
                sigma[1:6,iter] = 0.1*alpha*np.nanstd(pars[1:6,ind_best],axis=1) + (1.-0.1*alpha)*sigma_ori[1:6,iter-1]
                        
        # check center variance
        if (iter > 2):
            avg_var = np.mean(np.abs(center[:,iter]-center[:,iter-1])/center[:,iter])
                    
        # generate new proposed solutions
        center[:,iter] = (center[:,iter]+pars_best)/2.
        
        pars = np.random.normal(center[:,iter],sigma[:,iter],(sample,ndim)).T
        
        # keep best solution
        pars[:,0]=pars_best

        for n in range(sample):
            ind_low = np.where(pars[:,n]-prange[:,0] < 0.)
            pars[ind_low,n] = prange[ind_low,0]
            ind_hi = np.where(pars[:,n]-prange[:,1] > 0.)
            pars[ind_hi,n] = prange[ind_hi,1]
            
            
        iter += 1
        
    print( '     '.join('%0.3f' % v for v in pars_best), "{0:0.3e}".format(lik_best) )
    
    return pars_best

##########################################################################################
# Fit isochrones GAIA SYNTHETIC

#(dir+'/'+obs_file,  guess, magcut,  obs_plx=Plxmean,
#                                          obs_plx_er=Plxsig, 
#                                          prior=prior, 
#                                          bootstrap=False)    

def fit_iso_GAIA(obs_file,verbosefile,guess=False,magcut=17.0, member_cut=0.5, obs_plx=False, 
                  obs_plx_er=0.05,prior=np.array([[1.],[1.e6]]), bootstrap=False,fixFeH=False):
    
    print ('Starting isochrone fitting...')
#    verbosefile.write('Starting isochrone fitting...\n')
    
    obs = np.genfromtxt(obs_file,names=True)
        
    #remove nans
    cond1 = np.isfinite(obs['Gmag'])
    cond2 = np.isfinite(obs['BPmag'])
    cond3 = np.isfinite(obs['RPmag'])
    cond4 = obs['BPmag'] < magcut
    cond5 = obs['Pmemb'] > member_cut
    
    cond6 = obs['RFG'] > 50.0
    cond7 = obs['RFBP'] > 20.0
    cond8 = obs['RFRP'] > 20.0
    cond9 = obs['E_BR_RP_'] < 1.3+0.06*(obs['BPRP'])**2
    cond10 = obs['E_BR_RP_'] > 1.0+0.015*(obs['BPRP'])**2
    cond11 = obs['Nper'] > 8
       
    ind  = np.where(cond1&cond2&cond3&cond4&cond5&cond6&cond7&cond8&cond9&cond10&cond11)
   
    obs = obs[ind]

    obs_oc = np.copy(obs[['Gmag','BPmag','RPmag']])
    obs_oc.dtype.names=['Gmag','G_BPmag','G_RPmag']
    obs_oc_er = np.copy(obs[['e_Gmag','e_BPmag','e_RPmag']])
    obs_oc_er.dtype.names=['Gmag','G_BPmag','G_RPmag']
    weight = obs['Pmemb'] * (obs_oc['G_BPmag'].min()/obs_oc['G_BPmag'])**1
    
    ###########################################################################
    # Apply photometry correction as suggested in GAIA site
    
    obs_oc['Gmag'][(obs_oc['Gmag']>6.)] = obs_oc['Gmag'][(obs_oc['Gmag']>6.)] - 0.0032*(obs_oc['Gmag'][(obs_oc['Gmag']>6.)] - 6.)
    obs_oc['G_BPmag'][(obs_oc['Gmag']>10.9)] = obs_oc['G_BPmag'][(obs_oc['Gmag']>10.9)] - 0.005
    obs_oc['G_BPmag'][(obs_oc['Gmag']<10.9)] = obs_oc['G_BPmag'][(obs_oc['Gmag']<10.9)] - 0.026
    obs_oc['G_RPmag'] = obs_oc['G_RPmag'] - 0.012
    
    ###########################################################################
    # load full isochrone grid data and arrays of unique Age and Z values
    grid_dir = './grids/'
    mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='GAIA')
#    filters = ['Gmag','G_BPmag','G_RPmag']
    filters = ['G_BPmag','G_RPmag']
    refmag = 'G_BPmag'
    
    labels=['age', 'dist', 'met', 'Av']
    
    prange = np.array([[6.65,10.3],
                       [0.05,20.],
                       [-0.9,0.7],
                       [0.01,8.0]])
            
    if (guess != False):
        #prange[0,:] = [np.max([6.65, guess[0]-1.5]), np.min([10.3, guess[0]+1.5])]
        if(obs_plx > 3*obs_plx_er):
            prange[1,:] = [np.max([0.,1./(obs_plx+3*obs_plx_er)]), np.min([25.,1./np.abs(obs_plx-3*obs_plx_er)])]
        else:
            prange[1,:] = [np.max([0.,1./(obs_plx+3*obs_plx_er)]), np.min([25.,1./np.abs(obs_plx-0.8*obs_plx)])]
            
    # define lower limit for Av based on prior sigma
    if(prior[1,0] != 1.e3):
        prange[0,:] = [np.max([6.65, guess[0]-prior[1,0]]), np.min([10.3, guess[0]+prior[1,0]])]
          
    if fixFeH:
        prange[2,:] = [fixFeH-1.0e-3,fixFeH+1.0e-3]
        
        
    print('1/pi: ',1./obs_plx)
    print('age prange:',prange[0,:])    
    print('distance prange:',prange[1,:])
    print('Av prange:',prange[3,:])
    print('FeH prange:',prange[2,:])
    

    verbosefile.write('1/pi: %6.3f \n'%(1./obs_plx))
    verbosefile.write('age prange: [%6.1f , %6.1f] \n'%(prange[0,0],prange[0,1]))
    verbosefile.write('distance prange: [%6.1f , %6.1f] \n'%(prange[1,0],prange[1,1]))
    verbosefile.write('FeH prange: [%6.1f , %6.1f] \n'%(prange[2,0],prange[2,1]))

    ndim = prange.shape[0]

    # define CE tweak parameters
    nruns = 10
    itmax = 100    
    sample = 500

    band = 0.1
    alpha = 0.1
    tol = 1.e-3

    res = np.zeros([nruns,ndim])
    
    # start main loop of the method
    print ('----------------------------------------------------------------------')
    print ('Age       Dist.       [FeH]      Av')
    print ('----------------------------------------------------------------------')
    
    verbosefile.write('----------------------------------------------------------------------\n')
    verbosefile.write('Age       Dist.       [FeH]      Av \n')
    verbosefile.write('----------------------------------------------------------------------\n')

    #seed = 2**25
    for n in range(nruns):

        seed = np.random.randint(2**20,2**32) 
     
        if bootstrap:
            ind_boot = np.random.choice(np.arange(obs_oc.size), 
                                        size=obs_oc.size, replace=True)
            data_boot = obs_oc[ind_boot]
            data_boot_er = obs_oc_er[ind_boot]
            weight_boot = weight[ind_boot]
        else:
            data_boot = obs_oc
            data_boot_er = obs_oc_er
            weight_boot = weight
            
        res[n,:] = run_isoc_CE(lnlikelihoodCE,data_boot,data_boot_er,filters,refmag,prange,
           sample,itmax,band,alpha,tol,weight_boot,prior,seed,mp.cpu_count()-1,guess)
        
        verbosefile.write('   '.join('%6.3f' % v for v in res[n,:])+'\n')
        
        guess = res[n,:].tolist()
       
    print ('')
    print ('-------------------------------------------------------------')
    print (' Final result')
    print ('-------------------------------------------------------------')
    print ('   '.join('%0.6f' % v for v in np.median(res,axis=0)))
    print ('   '.join('%0.3f' % v for v in np.std(res,axis=0)))
    print ('')    
    print ('Finished isochrone fitting...')

    verbosefile.write('\n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write(' Final result \n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write('   '.join('%0.6f' % v for v in np.median(res,axis=0))+'\n')
    verbosefile.write('   '.join('%0.3f' % v for v in np.std(res,axis=0))+'\n')
    verbosefile.write('\n')    
    verbosefile.write('Finished isochrone fitting...\n')

    return np.median(res,axis=0),res.std(axis=0)
#    return res[nruns-1,:],res.std(axis=0)
        
##############################################################################

# Fit isochrones
    
def fit_isochrone(obs_file, verbosefile,guess=False,magcut=17.0, obs_plx=False, 
                  obs_plx_er=0.05,prior=np.array([[1.],[1.e6]]), bootstrap=False):
    
    print ('Starting isochrone fitting...')
    verbosefile.write('Starting isochrone fitting...\n')
    
    obs = np.genfromtxt(obs_file,names=True)
        
    #remove nans
    cond1 = np.isfinite(obs['Gmag'])
    cond2 = np.isfinite(obs['BPmag'])
    cond3 = np.isfinite(obs['RPmag'])
    cond4 = obs['BPmag'] < magcut
    cond5 = obs['BPmag'] > 10.
    
    ind  = np.where(cond1&cond2&cond3&cond4)
    
    obs = obs[ind]

    obs_oc = np.copy(obs[['Gmag','BPmag','RPmag']])
    obs_oc.dtype.names=['Gmag','G_BPmag','G_RPmag']
    obs_oc_er = np.copy(obs[['e_Gmag','e_BPmag','e_RPmag']])
    obs_oc_er.dtype.names=['Gmag','G_BPmag','G_RPmag']
    weight = obs['P'] * (obs_oc['G_BPmag'].min()/obs_oc['G_BPmag'])**1
    

    # load full isochrone grid data and arrays of unique Age and Z values
    grid_dir = './grids/'
    mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='GAIA')
#    filters = ['Gmag','G_BPmag','G_RPmag']
    filters = ['G_BPmag','G_RPmag']
    refmag = 'G_BPmag'
    
    labels=['age', 'dist', 'met', 'Av']
    
    prange = np.array([[6.65,10.3],
                       [0.05,20.],
                       [-0.9,0.7],
                       [0.01,8.0]])
            
    if (guess != False):
        #prange[0,:] = [np.max([6.65, guess[0]-1.5]), np.min([10.3, guess[0]+1.5])]
        if(obs_plx > 3*obs_plx_er):
            prange[1,:] = [np.max([0.,1./(obs_plx+3*obs_plx_er)]), np.min([25.,1./np.abs(obs_plx-3*obs_plx_er)])]
        else:
            prange[1,:] = [np.max([0.,1./(obs_plx+3*obs_plx_er)]), np.min([25.,1./np.abs(obs_plx-0.8*obs_plx)])]
            
    # define lower limit for Av based on prior sigma
    if(prior[1,0] != 1.e3):
        prange[0,:] = [np.max([6.65, guess[0]-prior[1,0]]), np.min([10.3, guess[0]+prior[1,0]])]
            
    print('1/pi: ',1./obs_plx)
    print('age prange:',prange[0,:])    
    print('distance prange:',prange[1,:])
    print('Av prange:',prange[3,:])
    

    verbosefile.write('1/pi: %6.1f \n'%(1./obs_plx))
    verbosefile.write('age prange: [%6.1f , %6.1f] \n'%(prange[0,0],prange[0,1]))
    verbosefile.write('distance prange: [%6.1f , %6.1f] \n'%(prange[1,0],prange[1,1]))

    ndim = prange.shape[0]

    # define CE tweak parameters
    nruns = 10
    itmax = 150    
    sample = 500

    band = 0.1
    alpha = 0.1
    tol = 1.e-3

    res = np.zeros([nruns,ndim])
    
    # start main loop of the method
    print ('----------------------------------------------------------------------')
    print ('Age       Dist.       [FeH]      Av')
    print ('----------------------------------------------------------------------')
    
    verbosefile.write('----------------------------------------------------------------------\n')
    verbosefile.write('Age       Dist.       [FeH]      Av \n')
    verbosefile.write('----------------------------------------------------------------------\n')

    #seed = 2**25
    for n in range(nruns):

        seed = np.random.randint(2**20,2**32) 
     
        if bootstrap:
            ind_boot = np.random.choice(np.arange(obs_oc.size), 
                                        size=obs_oc.size, replace=True)
            data_boot = obs_oc[ind_boot]
            data_boot_er = obs_oc_er[ind_boot]
        else:
            data_boot = obs_oc
            data_boot_er = obs_oc_er
            
        res[n,:] = run_isoc_CE(lnlikelihoodCE,data_boot,data_boot_er,filters,refmag,prange,
           sample,itmax,band,alpha,tol,weight,prior,seed,mp.cpu_count()-1,guess)
        
        verbosefile.write('   '.join('%6.3f' % v for v in res[n,:])+'\n')
        
        guess = res[n,:].tolist()
       
    print ('')
    print ('-------------------------------------------------------------')
    print (' Final result')
    print ('-------------------------------------------------------------')
    print ('   '.join('%0.6f' % v for v in np.median(res,axis=0)))
    print ('   '.join('%0.3e' % v for v in np.std(res,axis=0)))
    print ('')    
    print ('Finished isochrone fitting...')

    verbosefile.write('\n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write(' Final result \n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write('   '.join('%0.6f' % v for v in np.median(res,axis=0))+'\n')
    verbosefile.write('   '.join('%0.3e' % v for v in np.std(res,axis=0))+'\n')
    verbosefile.write('\n')    
    verbosefile.write('Finished isochrone fitting...\n')

    return np.median(res,axis=0),res.std(axis=0)
#    return res[nruns-1,:],res.std(axis=0)
        
##############################################################################

def run_isoc_CE(objective,obs,obs_er,filters,refmag,prange,sample,itmax,band,alpha,tol,
                weight,prior,seed,nthreads=1,guess=False):

    # Define arrays to be used
    ndim = prange.shape[0]
    lik = np.zeros(sample)
    center = np.zeros([ndim,itmax])
    sigma = np.zeros([ndim,itmax])
    pars_best = 0
    avg_var = 1e3
    diag_ind = np.diag_indices(ndim)
    midpoint = (prange[:,1]-prange[:,0])/2.+prange[:,0]
    
    # generate initial solution population
    pars=[]
    for k in range(ndim):
        aux = np.random.uniform(prange[k,0],prange[k,1], sample)
        pars.append(aux)    
    pars=np.array(pars)
    
    if (guess != False):
        pars[:,0] = guess

    iter = 0
    
    # enforce prange limits
    for n in range(sample):
        ind_low = np.where(pars[:,n]-prange[:,0] < 0.)
        pars[ind_low,n] = prange[ind_low,0]
        ind_hi = np.where(pars[:,n]-prange[:,1] > 0.)
        pars[ind_hi,n] = prange[ind_hi,1]
                
    while (iter < itmax and avg_var > tol):
                
##########################################################################################
#     parallel test
        pool = mp.Pool(processes=nthreads)
        res = [pool.apply_async(objective, args=(theta,obs,obs_er,filters,
                                                    refmag,prange,weight,prior,seed,)) for theta in pars.T]
        lik = np.array([p.get() for p in res])
        pool.close()
        pool.join()
###########################################################################################
#        ########### Smooth likelihood   #######################################
#        bandwidth = np.nanstd(pars,axis=1)/2.
#        indb = np.where(bandwidth<0.1)
#        bandwidth[indb] = 0.1
#        bandwidth = [0.001,0.001,0.2,0.01]
#        lik_s = np.zeros(lik.size)
#        
#        for i in range(sample):
#            t1 = np.sum(np.exp(-0.5*(pars[0,i]-pars[0,:])**2/bandwidth[0]**2) *\
#            np.exp(-0.5*(pars[1,i]-pars[1,:])**2/bandwidth[1]**2) *\
#            np.exp(-0.5*(pars[2,i]-pars[2,:])**2/bandwidth[2]**2) *\
#            np.exp(-0.5*(pars[3,i]-pars[3,:])**2/bandwidth[3]**2) * lik)
#            
#            t2 = np.sum(np.exp(-0.5*(pars[0,i]-pars[0,:])**2/bandwidth[0]**2) *\
#            np.exp(-0.5*(pars[1,i]-pars[1,:])**2/bandwidth[1]**2) *\
#            np.exp(-0.5*(pars[2,i]-pars[2,:])**2/bandwidth[2]**2) *\
#            np.exp(-0.5*(pars[3,i]-pars[3,:])**2/bandwidth[3]**2) )            
# 
#            lik_s[i] = t1/t2
#            
#        lik = lik_s

             
        # sort solution in descending likelihood
        ind = np.argsort(lik)
        
        # best solution in iteration
        pars_best = np.copy(pars[:,ind[0]])
        lik_best = lik[ind[0]]
        # indices of band best solutions
        ind_best = ind[0:int(band*sample)]
    
        # discard indices that are out of parameter space
        ind_best = ind_best[np.isfinite(lik[ind_best])]
        
        ########### Smooth likelihood   #######################################
        
        beta = alpha #* hill(iter,4.,0.5*itmax) + 0.01
        peso = np.resize(-lik,(ndim,sample))

        # calculate new proposal distribution
        if (iter == 0):
            center[:,iter] = np.nanmean(pars[:,ind_best],axis=1)
            covmat = np.cov(pars[:,ind_best])
        else:
            center[:,iter] = beta*np.nansum(pars[:,ind_best]*peso[:,ind_best],axis=1) \
            / np.nansum(peso[:,ind_best],axis=1) +  (1.-beta)*center[:,iter-1]
            
            covmat = beta*np.cov(pars[:,ind_best]) + (1.-beta)*covmat

        # check center variance
        if (iter > 10):
            #avg_var = np.max(np.abs(center[:,iter]-center[:,iter-1])/center[:,iter])
            avg_var = np.max(np.std(center[:,iter-10:iter],axis=1)/center[:,iter])
        if(covmat[2,2] < 0.1):
            covmat[2,2] = 0.2
            
#        print(pars[2,:])
#        print('center:',center[:,iter])
#        print(iter, covmat[diag_ind])
#        print('best:',lik_best,pars_best)
#        print('')
                    
        pars = np.random.multivariate_normal(center[:,iter],covmat,sample).T
        
        # enforce prange limits
        for n in range(sample):
            ind_low = np.where(pars[:,n]-prange[:,0] < 0.)
            pars[ind_low,n] = midpoint[ind_low]
            ind_hi = np.where(pars[:,n]-prange[:,1] > 0.)
            pars[ind_hi,n] = midpoint[ind_hi]


        # keep best solution
        pars[:,0]=pars_best
        
        iter += 1

#        print 'Best solution'
    print ('     '.join('%0.3f' % v for v in pars_best), "{0:0.2f}".format(lik_best), iter, "{0:0.5f}".format(avg_var))    
    return pars_best
    
#    print ('     '.join('%0.3f' % v for v in center[:,iter-1]), "{0:0.2f}".format(lik_best), iter, "{0:0.5f}".format(avg_var))    
#    return center[:,iter-1]
##########################################################################################
# Fit isochrones
    
def fit_isochrone_mcmc(obs_file, nwalkers, burn_in, nsteps, thin, guess=False,magcut=17.0):
    seed = np.random.randint(2**25,2**30)
    print ('-------------------------------------------------------------')
    print ('Starting MCMC fitting...')
    print ('-------------------------------------------------------------')
    
    obs = np.genfromtxt(obs_file,names=True)
    
    #remove nans
    cond1 = np.isfinite(obs['Gmag'])
    cond2 = np.isfinite(obs['BPmag'])
    cond3 = np.isfinite(obs['RPmag'])
    cond4 = obs['Gmag'] < magcut
    
    ind  = np.where(cond1&cond2&cond3&cond4)
    
    obs = obs[ind]

    obs_oc = np.copy(obs[['Gmag','BPmag','RPmag']])
    obs_oc.dtype.names=['Gmag','G_BPmag','G_RPmag']
    obs_oc_er = np.copy(obs[['e_Gmag','e_BPmag','e_RPmag']])
    obs_oc_er.dtype.names=['Gmag','G_BPmag','G_RPmag']
    weight = obs['P'] * obs_oc['Gmag'].min()/obs_oc['Gmag']
    

    # load full isochrone grid data and arrays of unique Age and Z values
    grid_dir = './grids/'
    mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='MIST-GAIA')
    filters = ['Gmag','G_BPmag','G_RPmag']
    refmag = 'Gmag'
    
    labels=['age', 'dist', 'met', 'Ebv', 'Rv','bin', 'alpha']
    
    prange = np.array([[6.5,10.3],
                       [0.1,10.],
                       [2e-06,0.048],
                       [0.1,2.0],
                       [2,4.],
                       [0.,0.8],
                       [1.5,3.5]])
            
            
    ndim = prange.shape[0]

    midpoint = (prange[:,1]-prange[:,0])/2.+prange[:,0]
    ndim = prange.shape[0]

    # define uniformly distributed walker starting positions
    pos=[]
    lik=[]
    for i in range(nwalkers):
        pars = []
        for k in range(ndim):
            pars.append(np.random.uniform(prange[k,0],prange[k,1]))
        pos.append(np.array(pars))
        lik.append(lnlikelihood(pars,obs_oc,obs_oc_er,filters,refmag,prange,weight))

    # If there is initial guess generate walkers around it
#    scale=(prange[:,1]-prange[:,0])/10.
#    if guess:
#        pos = [guess + scale*np.random.randn(ndim) for i in range(nwalkers)]
        
    start_time = timeit.default_timer()

    # setup sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood,a=1.1,
                                    args=(obs_oc,obs_oc_er,filters,refmag,prange,weight), 
                                    threads=mp.cpu_count()-1,live_dangerously=True)
    # run sampler in the burn in phase
    sampler.run_mcmc(pos, burn_in)
    
    # process samples
    samples = sampler.chain[:,:, :].reshape((-1, ndim))
    
    # get best values and confidence intervals
    best_vals = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [15, 50, 84],
                                                    axis=0))))
    
    # get best solution from maximul likelihood sampled
    
    best_sol = sampler.flatchain[sampler.flatlnprobability.argmax()]
    
    # reset sampler
    sampler.reset()
    
    # setup sampler after burn-in
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood,a=1.1,
                                    args=(obs_oc,obs_oc_er,filters,refmag,prange,weight,seed), 
                                    threads=mp.cpu_count()-1,live_dangerously=True)
    
    print ('done burn in phase...')
    print ('')
    print ('Best solution of burn in phase: ', best_sol)
    print ('Average solution of burn in phase: ',best_vals[:,0])
    
    # redefine initial positions based on burn in results
#    scale = 0.01*best_vals[:,0]
#    pos = [best_vals[:,0] + scale*np.random.randn(ndim) for i in range(nwalkers)]
    
    # run sampler for final sample
    sampler.run_mcmc(pos, nsteps, thin=thin)
    
    # get final best solution    
    best_sol = sampler.flatchain[sampler.flatlnprobability.argmax()]
    
    print ('Finished sampling')
    
    samples = sampler.chain[:,nsteps/2/thin:, :].reshape((-1, ndim))
    best_vals = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [5, 50, 95],
                                                    axis=0))))
    
    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))
    
    print ('Elapsed time: ', (timeit.default_timer() - start_time)/60., ' minutes')


    ###############################################################
    # print results
    
    fig = corner.corner(samples, labels=labels, levels=(0.68,0.95), smooth=True)
    
    for i in range(ndim):
        print (labels[i],best_sol[i],'-',best_vals[i,1],'+',best_vals[i,2])
    
    print ('')
    print ('From sample averages:')
    for i in range(ndim):
        print (labels[i],best_vals[i,0],'-',best_vals[i,1],'+',best_vals[i,2])
        
        
    # plot chains
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.chain
    for i in range(ndim):
        ax = axes[i]
        for k in range(nwalkers):
            ax.plot(np.array(samples[k,:, i]), "k", alpha=0.1)
        ax.set_ylabel(labels[i])
    
    # plot averages
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        for k in range(nwalkers):
            avgs = np.cumsum(np.array(samples[k,:, i]))/(np.arange(len(samples[k,:, i]))+1)        
            ax.plot(avgs, "k", alpha=0.1)
        ax.set_ylabel(labels[i])
    

    print ('-------------------------------------------------------------')
    print (' Final result')
    print ('-------------------------------------------------------------')
    print ('   '.join('%0.3f' % v for v in best_vals[:,1]))
    print ('   '.join('%0.3f' % v for v in (best_vals[:,1]+best_vals[:,2])/3))

    return best_sol, (best_vals[:,1]+best_vals[:,2])/3


##########################################################################################
# Fit isochrones
    
def fit_CCD(obs_file,guess=False,magcut=17.0):
    
    from scipy.optimize import differential_evolution
    
    print ('Starting CCD fitting...')
    
    obs = np.genfromtxt(obs_file,names=True)
    
    #remove nans
    cond1 = np.isfinite(obs['Gmag'])
    cond2 = np.isfinite(obs['BPmag'])
    cond3 = np.isfinite(obs['RPmag'])
    cond4 = obs['Gmag'] < magcut
    
    
    ind  = np.where(cond1&cond2&cond3&cond4)
    
    obs = obs[ind]

    obs_oc = np.copy(obs[['Gmag','BPmag','RPmag']])
    obs_oc.dtype.names=['Gmag','G_BPmag','G_RPmag']
    obs_oc_er = np.copy(obs[['e_Gmag','e_BPmag','e_RPmag']])
    obs_oc_er.dtype.names=['Gmag','G_BPmag','G_RPmag']
    weight = obs['P'] #* obs_oc['Gmag'].min()/obs_oc['Gmag']
    

    # load full isochrone grid data and arrays of unique Age and Z values
    grid_dir = './grids/'
    mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='GAIA')
    filters = ['Gmag','G_BPmag','G_RPmag']
    refmag = 'G_BPmag'
        
    prange = np.array([(6.6,10.1),
                       (.1,10.),
                       (-0.9,0.7),
                       (0.001,6.0)])

    midpoint = (prange[:,1]-prange[:,0])/2.+prange[:,0]
    ndim = prange.shape[0]

    # define CE tweak parameters
    nruns = 5
    itmax = 100    
    sample = 500

    band = 0.15
    alpha = 0.1
    tol = 1.e-3

    res = np.zeros([nruns,ndim])
    
    # start main loop of the method
    print ('----------------------------------')
    print ('Age       Dist.       Met.      Av')
    print ('-----------------------------------')
    
    
    for n in range(nruns):
        # set seed for the run
        seed = np.random.randint(2**20,2**30)
        
        result = differential_evolution(lnlikelihoodCE, prange, 
                                        args=(obs_oc,obs_oc_er,filters,refmag,prange,weight,seed))

        res[n,:] = result.x
       
    print ('')
    print ('-------------------------------------------------------------')
    print (' Final result')
    print ('-------------------------------------------------------------')
    print ('   '.join('%0.3f' % v for v in np.median(res,axis=0)))
    print ('   '.join('%0.3f' % v for v in res.std(axis=0)))
    print ('')
    
    print ('Finished isochrone fitting...')

    return np.median(res,axis=0),res.std(axis=0)
        
##############################################################################


















