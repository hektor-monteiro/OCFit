#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:34:43 2018

@author: Wilton Dias and Hektor Monteiro 
"""

import numpy as np
import multiprocessing as mp
from scipy import stats
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.stats import mad_std
from math import ceil
from scipy import linalg
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker,TextArea, DrawingArea
from scipy.optimize import least_squares,minimize
import os
from scipy.interpolate import interp1d,LinearNDInterpolator,griddata
from scipy.integrate import trapz,cumtrapz


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
def likelihood_dist(dist,plx, eplx):
    return -np.sum(np.log(1./np.sqrt(2*np.pi)/eplx * 
                         np.exp( -0.5*(plx - (1/dist))**2/(2 * eplx**2.) )))

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


###############################################
# Load binary file with full isochrone grid
# and returns array of data and arrays of unique age and Z values
#
def load_mod_grid(dir, isoc_set='UBVRI'):
    global mod_grid
    global age_grid
    global z_grid

    if(isoc_set == 'UBVRI'):
        mod_grid = np.load(dir+'full_isoc_UBVRI_CMD33.npy')
        
    if(isoc_set == 'MIST-UBVRI'):
        mod_grid = np.load(dir+'full_isoc_MIST-UBVRI.npy')
                    
    age_grid = np.unique(mod_grid['Age'])
    z_grid = np.unique(mod_grid['Zini'])
    
    return mod_grid, age_grid, z_grid

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
#    ang_sep = sky_coords.separation(SkyCoord(ra[np.argmax(dens_map)], 
#                                                dec[np.argmax(dens_map)], 
#                                                unit="deg"))
    
    # get distance from cluster center
    ang_sep = sky_coords.separation(cluster_center)
    
    return pix_coords[0], pix_coords[1], dens_map, np.argmax(dens_map), ang_sep.to(u.arcmin)

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
    sig_clip = 3.
    cond1 = np.abs(obs-np.nanmean(obs)) < sig_clip*np.nanstd(obs)    
    obs = obs[cond1]
    weight = weight[cond1]

    print ('Starting parallax fitting...')
    
#     Define the parameter space
    prange = np.array([[0.1,0.9],
                       [0,obs.max()],
                       [0.001,0.05],
                       [obs.min(),obs.max()],
                       [0.1,obs.std()]])
    
    # define CE tweak parameters
    nruns = 5
    itmax = 100    
    sample = 800
    
    band = 0.15
    alpha = 0.1
    tol = 1.e-3

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
# Fit isochrones
        
def fit_isochroneUBVRI(obs_file, verbosefile, probcut, guess=False,magcut=20.0, obs_plx=False, 
                  obs_plx_er=0.05,prior=np.array([[1.],[1.e6]]),bootstrap=True):
    
    print ('Starting isochrone fitting...')
    
    obs = np.genfromtxt(obs_file,names=True)

       
    #remove nans
    cond1 = np.isfinite(obs['U'])
    cond2 = np.isfinite(obs['B'])
    cond3 = np.isfinite(obs['V'])
    cond4 = np.isfinite(obs['R'])
    cond5 = np.isfinite(obs['I'])
    cond6 = obs['V'] < magcut
    cond7 = obs['P'] > probcut
    
    
    ind  = np.where(cond1&cond2&cond3&cond4&cond5&cond6&cond7)
    
    obs = obs[ind]

    obs_oc = np.copy(obs[['U','B','V','R','I']])
    obs_oc.dtype.names=['Umag','Bmag','Vmag','Rmag','Imag']
    obs_oc_er = np.copy(obs[['SU','SB','SV','SR','SI']])
    obs_oc_er.dtype.names=['Umag','Bmag','Vmag','Rmag','Imag']
    weight = obs['P']
    
    # load full isochrone grid data and arrays of unique Age and Z values
    grid_dir = os.getenv("HOME")+'/OCFit/grids/'
    mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='UBVRI')
    filters = ['Umag','Bmag','Vmag','Rmag','Imag']
    refmag = 'Vmag'
    
    labels=['age', 'dist', 'met', 'Av']
    
    prange = np.array([[6.65,10.3],
                       [0.05,20.],
                       [-0.9,0.7],
                       [0.01,5.0]])

    ndim = prange.shape[0]

    # define CE tweak parameters
    nruns = 3
    itmax = 100    
    sample = 300

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

    for n in range(nruns):

        seed= None #np.random.randint(2**20,2**21) # for each run set distinct seed for IMF sampling
        
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
       
    print ('')
    print ('-------------------------------------------------------------')
    print (' Final result')
    print ('-------------------------------------------------------------')
    print ('   '.join('%0.6f' % v for v in np.median(res,axis=0)))
    print ('   '.join('%0.6f' % v for v in mad_std(res,axis=0)))
    print ('')    
    print ('Finished isochrone fitting...')

    verbosefile.write('\n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write(' Final result \n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write('   '.join('%0.6f' % v for v in np.median(res,axis=0))+'\n')
    verbosefile.write('   '.join('%0.6f' % v for v in mad_std(res,axis=0))+'\n')
    verbosefile.write('\n')    

    return np.median(res,axis=0),res.std(axis=0)
        
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
#     run liklihood calculation in parallel
        pool = mp.Pool(processes=nthreads)
        res = [pool.apply_async(objective, args=(theta,obs,obs_er,filters,
                                                    refmag,prange,weight,prior,seed,)) for theta in pars.T]
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
        ind_best = ind[0:int(band*sample)]
    
        # discard indices that are out of parameter space
        ind_best = ind_best[np.isfinite(lik[ind_best])]
        
        # leaving here for now. This allows for change in convergence speed
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

        # check center variance in the last 10 iterations
        if (iter > 10):
            avg_var = np.max(np.std(center[:,iter-10:iter],axis=1)/center[:,iter])
        
        if(covmat[2,2] < 0.1):
            covmat[2,2] = 0.2

        pars = np.random.multivariate_normal(center[:,iter],covmat,sample).T
        
        # enforce prange limits
        for n in range(sample):
            ind_low = np.where(pars[:,n]-prange[:,0] < 0.)
            pars[ind_low,n] = midpoint[ind_low]
            ind_hi = np.where(pars[:,n]-prange[:,1] > 0.)
            pars[ind_hi,n] = midpoint[ind_hi]


        # keep best solution
        pars[:,0]=pars_best

#        print('center:',center[:,iter])
#        print(iter, avg_var, covmat[diag_ind])
#        print('best:',lik_best,pars_best)
#        print('')
        
        iter += 1

#        print 'Best solution'
    print ('     '.join('%0.3f' % v for v in pars_best), "{0:0.2f}".format(lik_best), iter, "{0:0.5f}".format(avg_var))
    
    return pars_best
    
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
    
    bin_comp_stars = sample_from_isoc(isoc_bin,bands,refMag,nbin,imf=imf,alpha=alpha,
                                      beta=beta,seed=seed)
    

    for filter in bands:
        m1 = isoc[filter][prob_bin < bin_frac]
        m2 = bin_comp_stars[filter]
        mcomb = m1 - 2.5 * np.log10(1.0 + 10**(-0.4*(m2-m1)))
        isoc[filter][prob_bin < bin_frac] = mcomb
        
    return isoc

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
# Make an observed synthetic cluster given an isochrone,

def make_obs_iso(bands, iso, dist, Av):
    #redden and move isochrone
    obs_iso = np.copy(iso)
    
    for filter in bands:
        
        # get CCm coeficients
        wav,a,b = ccm_coefs(filter)
    
        # apply ccm model and make observed iso
        obs_iso[filter] = iso[filter] + 5.*np.log10(dist*1.e3) - 5.+ ( (a + b/3.1)*Av )
        
    return obs_iso


###############################################
# generate a synthetic cluster
    
def model_cluster(age,dist,FeH,Av,bin_frac,nstars,bands,refMag,Mcut=False,error=True,
                  seed=None,Abscut=False,imf='chabrier',alpha=2.3,beta=-3):
        
    # get isochrone
    met = (10.**FeH)*0.0152
    grid_iso = get_iso_from_grid(age,met,bands,refMag,Abscut=Abscut)

    # make an observed isochrone
    obs_iso = make_obs_iso(bands, grid_iso, dist, Av)

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
def lnlikelihoodCE(theta,obs_iso,obs_iso_er,bands,refMag,prange,weight,prior=[[1.],[1.e3]],seed=None):
    
    # generate synth cluster from input parameters
    age, dist, FeH, Av = theta
    
    bin_frac = 0.5

    # number of stars to generate
    nstars = obs_iso.size
    Mlim = obs_iso[refMag].max()
    
    # get synth isochrone

    mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,800,bands,
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



