#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:21:11 2017

@author: hmonteiro
"""
import os
# to disable numpy multithreading
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import mkl
mkl.set_num_threads(1)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
import sys
import timeit
import multiprocessing as mp
from scipy.spatial.distance import cdist
from scipy import stats
import time
from astroquery.vizier import Vizier
import astropy.units as u
import warnings
from astropy.modeling import models, fitting
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from gaia_dr2_tools import *
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.integrate import simps

warnings.filterwarnings("ignore")

#os.system('rm -r ~/.astropy/cache/astroquery/Vzier/*')
#os.environ["OMP_NUM_THREADS"] = "1"

plt.close('all')


dirout = os.getenv("HOME")+'/OCFit/results/'

############################################################################
# Read data from Cantat CE isochrone fits
    
clusters = np.genfromtxt(os.getenv("HOME")+'/OCFit/source/catalog-V3.0-2019.txt', 
                         names=True,dtype=None)
print ('done reading Catalog data.')

# read EBV data from https://stilism.obspm.fr/
ebv_gaia_2mass = np.genfromtxt(os.getenv("HOME")+'/OCFit/source/clusters_for_ebv_with_stilism_reddening.csv',
                               delimiter=',',names=True,dtype=None)

############################################################################
# choose specific clusters to fit

todo = ['Berkeley_19']
#todo = ['Berkeley_19','Berkeley_17','NGC_6705','NGC_2420','Kronberger_149','King_7', 'NGC_6791']

ind_todo = []
guess_Av = []
guess_Av_sig = []
for oc in todo:
#    print(oc,np.where(clusters['name'] == oc.encode()))
    ind_todo.append( (np.where(clusters['name'] == oc.encode())[0][0] ) )
    
    ind_ebv = np.where(ebv_gaia_2mass['source_id'] == oc.encode())
    print(oc,ebv_gaia_2mass['source_id'][ind_ebv])
    guess_Av.append(3.1*ebv_gaia_2mass['ebv'][ind_ebv])
    guess_Av_sig.append(3.1*np.abs(ebv_gaia_2mass['e_ebv_max'][ind_ebv]+ebv_gaia_2mass['e_ebv_min'][ind_ebv])/2.)
    
clusters = clusters[ind_todo]
guess_Av = np.array(guess_Av).ravel()
guess_Av_sig = np.array(guess_Av_sig).ravel()

############################################################################

log_res = []
log_res_er = []

for i in range(clusters['name'].size):
    
    plt.close('all')
    
    start = time.time()
    
    name, radius = clusters['name'][i].decode(), clusters['radius'][i]
    radius = radius + 2. + 0.2*(radius-10)*smoothclamp((radius-10),0,1)
        
    # create directory for results of the cluster
    try:
        os.stat(dirout+name)
    except:
        os.mkdir(dirout+name)       

    # file to save all output
    verbosefile = open(dirout+name+'/'+'verbose-output.dat', 'w') 

    logfilename = 'results_'+name+'.txt'
    logfile = open(dirout+name+'/'+logfilename, "w")
    logfile.write(time.strftime("%c")+'\n')
    logfile.write(' \n')
    logfile.write('             name       rah   ram   ras        deg    dem    decs    RA_ICRS     DE_ICRS    radius   crad     dist     e_dist    age     e_age     FeH    e_FeH     Av      e_Av      AG      e_AG       Nc      Plx     sigPlx   e_Plx    pmRA     sigpmRA  e_pmRA   pmDE     sigpmDE  e_pmDE   Vr      e_Vr \n')

    if (clusters['radius'][i] < 1500.):
                    
        print ('')
        print ('##########################################################################################')
        print ('------------------------------------------------------------------')
        print ('Doing ', name, ' with radius ',radius, ' arcmin', 'i=',i)

        verbosefile.write('\n')
        verbosefile.write('##########################################################################################\n')
        verbosefile.write('------------------------------------------------------------------\n')
        verbosefile.write('Doing %s with radius %f6.1 arcmin with i= %i \n'%(name,radius,i))

    else:
        print ('')
        print ('Cluster ', name, ' with radius ',radius, ' is too big. Skiping...')
        
        verbosefile.write('\n')
        verbosefile.write('Cluster %s with %6.1f radius is too big, skipping...\n'%(name,radius))
        
        logfile = open(dirout+name+'/'+logfilename, "a")
        logfile.write('cluster '+'%s'%name+' with radius '+'%f'%radius+' is too big. Skiping...\n')
        logfile.write('\n')
        
        print ('')
        verbosefile.write('\n')
        
        continue

    outfile = name
        
    #Vizier.COLUMNS = ['*','pmRApmDEcor']
    v = Vizier(columns=['*', 'pmRApmDEcor','PlxpmRAcor','PlxpmDEcor',
                        'RFG','RFBP','RFRP','E(BR/RP)','Nper'],
               row_limit=999999999,timeout=120)
    #Vizier.VIZIER_SERVER = 'vizier.cfa.harvard.edu'
    #Vizier.VIZIER_SERVER = 'vizier.u-strasbg.fr'
    
    coord = '%f %f'%(clusters['RA_ICRS'][i],clusters['DE_ICRS'][i])
    if not os.path.isfile(dirout+name+'/'+outfile+'_data.npy'):
        res=v.query_region(SkyCoord(coord, unit=u.deg), radius=radius*u.arcmin,catalog='I/345/gaia2')
        data = np.array(res[0])
        np.save(dirout+name+'/'+outfile+'_data', data)
    else:
        # lugar para ler o arquivo de entrada texto
        print ('reading file...')
        data = np.load(dirout+name+'/'+outfile+'_data.npy')
        print ('done.')
    
####################################################################################
    # calculate membership probability
    
    sig_clip = 6.

    cond1 = np.isfinite(data['pmRA'])
    cond2 = np.isfinite(data['pmDE'])
    cond3 = np.isfinite(data['Plx'])
    cond4 = np.abs(data['pmRA']-np.nanmean(data['pmRA'])) < sig_clip*np.nanstd(data['pmRA'])
    cond5 = np.abs(data['pmDE']-np.nanmean(data['pmDE'])) < sig_clip*np.nanstd(data['pmDE'])
    
    cond6 = data['RFG'] > 50.0
    cond7 = data['RFBP'] > 20.0
    cond8 = data['RFRP'] > 20.0

    cond9 = data['E_BR_RP_'] < 1.3+0.06*(data['BP-RP'])**2
    cond10 = data['E_BR_RP_'] > 1.0+0.015*(data['BP-RP'])**2

    cond11 = data['Nper'] > 8

       
    ind  = np.where(cond1&cond2&cond3&cond4&cond5&cond6&cond7&cond8&cond9&cond10&cond11)
    
    pmRA = (data['pmRA'])[ind]
    erpmRA = (data['e_pmRA'])[ind]
    
    pmDE = (data['pmDE'])[ind]
    erpmDE = (data['e_pmDE'])[ind]
    
    Plx = (data['Plx'])[ind]
    erPlx = (data['e_Plx'])[ind]

    Gmag = (data['Gmag'])[ind]
    BPmag = (data['BPmag'])[ind]
    RPmag = (data['RPmag'])[ind]    
    erGmag = (data['e_Gmag'])[ind]
    erBPmag = (data['e_BPmag'])[ind]
    erRPmag = (data['e_RPmag'])[ind]    

    Vr = (data['RV'])[ind]
    erVr = (data['e_RV'])[ind]
    AG = (data['AG'])[ind]
    
    ra = (data['RA_ICRS'])[ind]*u.degree
    dec = (data['DE_ICRS'])[ind]*u.degree
    
    gaiaID = (data['Source'])[ind]
 
    pmRApmDEcor = (data['pmRApmDEcor'])[ind]
    PlxpmRAcor = (data['PlxpmRAcor' ])[ind]
    PlxpmDEcor = (data['PlxpmDEcor'])[ind]

    err = np.sqrt(data['e_pmRA']**2+data['e_pmDE']**2)
    err = err[ind]
    
    ###################################################################
    # PM and Parallax density distribution    
    values = np.vstack([pmRA, pmDE, Plx])
    PM_dens = stats.gaussian_kde(values, bw_method='silverman')
    PM_dens.set_bandwidth(bw_method=PM_dens.factor / 5.)
    
    weight = PM_dens(values) 
    
    ###################################################################
    # Calculate membership probabilities

    fit_data = (pmRA,pmDE,Plx,erpmRA,erpmDE,erPlx,pmRApmDEcor,PlxpmRAcor,PlxpmDEcor,weight)

    pars = np.array([clusters['Nc'][i]/float(Plx.size),clusters['pmRA'][i],clusters['pmDE'][i],
                     clusters['Plx'][i],clusters['sigpmRA'][i],clusters['sigpmDE'][i],
                     clusters['sigPlx'][i], pmRA.mean(),pmDE.mean(),Plx.mean(),pmRA.std(),pmDE.std(),Plx.std()])

    members =  membership_3D(pars,fit_data)

    ind_m = members > .51

    ind_m = members > .51
    if (Plx[ind_m].size > 200):
        ind_m = members > .9
    ####################################################################################
    ## Calculate kernel density

    values = np.vstack([pmRA, pmDE])

    # Perform a kernel density estimate on the data using sklearn.
    PM_dens = stats.gaussian_kde(values, bw_method='silverman')
    PM_dens.set_bandwidth(bw_method=PM_dens.factor / 5.)
    PM_dens = PM_dens(values)    

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    # show 1sigma ellipse
    e_c = Ellipse(xy=[clusters['pmRA'][i],clusters['pmDE'][i]],
                  width=3*clusters['sigpmRA'][i], height=3*clusters['sigpmDE'][i],
                angle=0.,fc=None, ec='r',fill=False)
    
    e_f = Ellipse(xy=[pmRA.mean(),pmDE.mean()],width=3*pmRA.std(), height=3*pmDE.std(),
                angle=np.arccos(np.corrcoef(pmRA,pmDE)[0,1])*180./np.pi-90.,fc=None, ec='k',fill=False)
    #np.arccos(x[10])*180./np.pi-90.
    
    ax.add_artist(e_c)
    ax.add_artist(e_f)
    
    ax.tricontourf(pmRA,pmDE,PM_dens, 60)
    ax.set_xlabel('pmRA')
    ax.set_ylabel('pmDE')
    
    plt.savefig(dirout+name+'/'+outfile+'_PM2Dhist-full.png', dpi=300)

    ####################################################################################
    # save all good data to file
    np.savez(dirout+name+'/'+name+'_var_save.npz', ra.value,dec.value,pmRA,pmDE,erpmRA,erpmDE,Plx,erPlx,Gmag,BPmag,RPmag,
             erGmag,erBPmag,erRPmag,Vr,erVr,AG,pmRApmDEcor,PlxpmRAcor,PlxpmDEcor)

    ####################################################################################
    # Plot RAW CMD
    fig, ax = plt.subplots()   
    plt.scatter(BPmag-RPmag,Gmag,s=1,color='k')
    plt.ylim(Gmag.max(),Gmag.min())
    plt.xlabel('(BPmag - RPmag)')
    plt.ylabel('Gmag')
    plt.title(name)    
    plt.savefig(dirout+name+'/'+outfile+'_CMD_raw.png', dpi=300)

    ####################################################################################
    # Plot decontaminated CMD
    fig, ax = plt.subplots()   
    plt.scatter(BPmag-RPmag,Gmag,s=1,color='gray',alpha=0.4)
    plt.scatter((BPmag-RPmag)[ind_m], Gmag[ind_m],s=10*members[ind_m],c=members[ind_m],cmap='jet')
    plt.ylim(Gmag.max(),Gmag.min())
    plt.xlabel('(BPmag - RPmag)')
    plt.ylabel('Gmag')
    plt.title(name)    
    plt.savefig(dirout+name+'/'+outfile+'_CMD_members.png', dpi=300)

    ####################################################################################
    # Plot VPD
    fig, ax = plt.subplots()   
    plt.scatter(pmRA,pmDE,s=1,color='gray',alpha=0.4)
    plt.scatter(pmRA[ind_m], pmDE[ind_m],s=1*members[ind_m],c=members[ind_m],cmap='jet')
    #plt.ylim(Gmag.max(),Gmag.min())
    plt.xlabel('pm RA')
    plt.ylabel('pm DEC')
    plt.title(name)    
    plt.savefig(dirout+name+'/'+outfile+'_VPD.png', dpi=300)

    ####################################################################################
    # Plot mass distribution
    fig, ax = plt.subplots()   
    plt.scatter(ra,dec,s=1,color='gray',alpha=0.4)
    plt.scatter(ra[ind_m], dec[ind_m],s=10*members[ind_m],c=(BPmag-RPmag)[ind_m],cmap='jet')
    #plt.ylim(Gmag.max(),Gmag.min())
    plt.xlabel('pm RA')
    plt.ylabel('pm DEC')
    plt.title(name)    
    plt.savefig(dirout+name+'/'+outfile+'_Mass-dist.png', dpi=300)

    ####################################################################################
    # Plot AG distribution
    
    if (AG[np.isfinite(AG)].size > 10):
        fig, ax = plt.subplots()   
        plt.tricontourf(ra[np.isfinite(AG)],dec[np.isfinite(AG)],AG[np.isfinite(AG)], 
                           30,cmap='jet',interp='bilinear')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.title(name)    
        plt.colorbar()
        plt.savefig(dirout+name+'/'+outfile+'_AG_map.png', dpi=300)

    ####################################################################################
    # Plot member extinction histogram
    AG_mean = np.nanmean(AG[ind_m])
    AG_std = np.nanstd(AG[ind_m])
    if (np.isfinite(AG_mean) and np.isfinite(AG_std)):
        fig, ax = plt.subplots()
        count, bins, patches = plt.hist(AG[ind_m][np.isfinite(AG[ind_m])],bins='auto')
        at = AnchoredText('%.3f +/- %.3f'%(AG_mean,AG_std),loc=2, frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at) 
        plt.title(name)    
        plt.savefig(dirout+name+'/'+outfile+'_AG-hist.png', dpi=300)

    
    ####################################################################################
    # Plot Radial velocity histogram
    Vr_mean = np.nanmean(Vr[ind_m])
    Vr_std = np.nanstd(Vr[ind_m])
    
    if (np.isfinite(Vr_mean) and np.isfinite(Vr_std)):
        
        indf = np.isfinite(Vr[ind_m])
        Vr_mean = np.sum((Vr[ind_m]/erVr[ind_m])[indf])/np.sum((1./erVr[ind_m])[indf])
        Vr_std = (Vr[ind_m])[indf].size/np.sum((1./erVr[ind_m])[indf])
        
        fig, ax = plt.subplots()
        count, bins, patches = plt.hist((Vr[ind_m])[indf],bins='auto')
        plt.text(Vr_mean+Vr_std,count.max(),'%.3f +/- %.3f'%(Vr_mean,Vr_std))
        plt.title(name)  
        
        at = AnchoredText('%.3f +/- %.3f'%(Vr_mean,Vr_std),loc=2, frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at) 
        
        plt.savefig(dirout+name+'/'+outfile+'_Vr-hist.png', dpi=300)

    
    ####################################################################################
    # obtain star density map
    
    ra_cen = clusters['RA_ICRS'][i]*u.degree
    dec_cen = clusters['DE_ICRS'][i]*u.degree
        
    xd,yd,dens_map,peak_ind,c_dist = star_density(ra,dec,ra_cen.value, dec_cen.value)
    
    fig, ax = plt.subplots()
    rad = Circle((0.,0.),radius=clusters['radius'][i],
                 ls='--',fill=False)
    ax.tricontourf(xd,yd,dens_map, 30,cmap='Greys')
    ax.add_artist(rad)
    ax.set_aspect('equal')
    ax.plot(0,0,'+k')
    ax.plot(xd[peak_ind],yd[peak_ind],'+w')
    ax.scatter(xd[ind_m],yd[ind_m],s=6.*members[ind_m]**4,c=members[ind_m],cmap='jet')
    ax.set_title(name)
    ax.set_xlabel('Arcmin')
    ax.set_ylabel('Arcmin')
    plt.savefig(dirout+name+'/'+outfile+'_star_dens_Wmembers.png', dpi=300)

    print ('cluster center from catalog: %f %f'%(ra_cen.value,
                                                    dec_cen.value))

    print ('coordinates of max density: %f %f'%(ra[peak_ind].value,
                                                    dec[peak_ind].value))
    
    verbosefile.write('cluster center from catalog: %f %f \n'%(ra_cen.value,
                                                    dec_cen.value))

    verbosefile.write('coordinates of max density: %f %f \n'%(ra[peak_ind].value,
                                                    dec[peak_ind].value))
    
    ####################################################################################
    # plot king profile 

    if (xd[ind_m].size > 3):
        fig, ax = plt.subplots() 

        # calculate kernel density with sig_pix bandwidth    
        values = np.vstack([xd[ind_m],yd[ind_m]])
        kernel_dens = stats.gaussian_kde(values,bw_method='silverman')
        mdens_map = kernel_dens(values)

        plt.scatter(c_dist,dens_map,s=1,color='k',alpha=0.1)
        plt.scatter(c_dist[ind_m],mdens_map,s=10*members[ind_m],
                    c=members[ind_m],cmap='jet')
        plt.ylim(0.,mdens_map.max())
        plt.xlabel('dist. from center (arcmin)')
        plt.ylabel('Norm. density')
        plt.savefig(dirout+name+'/'+outfile+'_king.png', dpi=300)

    ####################################################################################
    # write member file
    if os.path.isfile(dirout+name+'/'+outfile+'_members.dat'):
        os.system('rm '+dirout+name+'/'+outfile+'_members.dat')
        
    filters = ['Gmag','G_BPmag','G_RPmag']
    refmag = 'Gmag'

    f=open(dirout+name+'/'+outfile+'_members.dat','ab')

    head = ['id', 'raj2000', 'dej2000', 'Gmag', 'e_Gmag', 'BPmag', 'e_BPmag', 
            'RPmag', 'e_RPmag', 'Plx','e_Plx','P']
    
    f.write(('     '.join(head)+'\n').encode())

    for k in range(Gmag[ind_m].size):
        line = [k,(ra[ind_m].value)[k], (dec.value[ind_m])[k],(Gmag[ind_m])[k],(erGmag[ind_m])[k],
            (BPmag[ind_m])[k],(erBPmag[ind_m])[k],(RPmag[ind_m])[k],(erRPmag[ind_m])[k],
            (Plx[ind_m])[k],(erPlx[ind_m])[k],(members[ind_m])[k]]
        
        np.savetxt(f,[line], fmt=' '.join(['%i'] + ['%f']*2 + ['%2.4f']*9))
        
    f.close()

    ####################################################################################
    # write full star file
    if os.path.isfile(dirout+name+'/'+outfile+'_stars.dat'):
        os.system('rm '+dirout+name+'/'+outfile+'_stars.dat')
        
    filters = ['Gmag','G_BPmag','G_RPmag']
    refmag = 'Gmag'

    f=open(dirout+name+'/'+outfile+'_stars.dat','ab')

    head = ['id', 'raj2000', 'dej2000', 'Gmag', 'e_Gmag', 'BPmag', 'e_BPmag', 
            'RPmag', 'e_RPmag', 'pmRA','e_pmRA','pmDE','e_pmDE','Plx','e_Plx','Vr','e_Vr','P']
    
    f.write(('     '.join(head)+'\n').encode())

    for k in range(Gmag.size):
        line = [(gaiaID)[k],(ra.value)[k], (dec.value)[k],(Gmag)[k],(erGmag)[k],
            (BPmag)[k],(erBPmag)[k],(RPmag)[k],(erRPmag)[k],(pmRA)[k],(erpmRA)[k],(pmDE)[k],(erpmDE)[k],
            (Plx)[k],(erPlx)[k],(Vr)[k],(erVr)[k],(members)[k]]
        
        np.savetxt(f,[line], fmt=' '.join(['%i'] + ['%f']*2 + ['%2.4f']*15))
        
    f.close()
    
    ####################################################################################
    #     fit isochrone to the data
    guess_dist = infer_dist(Plx[ind_m]+0.029, erPlx[ind_m],guess=1./clusters['Plx'][i])
    print('Infered distance from parallax: %8.3f \n'%(guess_dist))
    verbosefile.write('Infered distance from parallax: %8.3f \n'%(guess_dist))

#    dist_posterior_x=[]
#    dist_posterior_y=[]
#    for d in np.linspace(0.1,50.,100): 
#        dist_posterior_x.append(d)
#        dist_posterior_y.append(-likelihood_dist(d,Plx[ind_m], erPlx[ind_m]))
 
    
#    guess = [clusters['age'][i],guess_dist,0.,
#                 clusters['ebv'][i]*clusters['Rv'][i]]
    
    if (clusters['Plx'][i] > 1*clusters['e_Plx'][i]):
        dist_guess_sig = ( 1./(clusters['Plx'][i]-1*clusters['e_Plx'][i]) - 
                          1./(clusters['Plx'][i]+1*clusters['e_Plx'][i]) )/2.
    else:
        dist_guess_sig = np.min([0.5*guess_dist,1.])
        
    if np.isfinite(guess_Av[i]):
        Av_guess = guess_Av[i]
        Av_guess_sig = guess_Av_sig[i]
    else:
        Av_guess = 1.
        Av_guess_sig = 1.e3
        
    print('Ag from GAIA members: %8.3f +/- %8.3f \n'%(AG_mean, AG_std))
    verbosefile.write('Ag from GAIA members: %8.3f +/- %8.3f \n'%(AG_mean, AG_std))

    print('Aprox. Ag from E(B-V) map using Rv=3.1: %8.3f +/- %8.3f  \n'%(0.9*guess_Av[i], 0.9*guess_Av_sig[i]))
    verbosefile.write('Aprox. Ag from E(B-V) map using Rv=3.1: %8.3f +/- %8.3f  \n'%(0.9*guess_Av[i], 0.9*guess_Av_sig[i]))

    guess = [clusters['age'][i],guess_dist,0.,Av_guess]    
#    guess_sig = np.array([1.e3, dist_guess_sig, 1.e3, 1.e3])
    guess_sig = np.array([1.e3, 1.e3, 1.e3, 1.e3])
      
    prior = np.stack([guess,guess_sig])
    
    print ('Guess:')
    print(guess)
    print('Prior sig: ')
    print(guess_sig)
    
    verbosefile.write('Guess: \n')
    verbosefile.write(str(guess)+'\n')
    
    verbosefile.write('Prior sigma: \n')
    verbosefile.write(str(guess_sig)+'\n')
    
    magcut = BPmag[ind_m].max()

    print ('Mag. Cut:')
    print(magcut)

    verbosefile.write('Mag. Cut: \n')
    verbosefile.write(str(magcut)+'\n')

    npoint = np.where(Gmag[ind_m] < magcut)
    print ('number of member stars:', npoint[0].size)
    verbosefile.write('number of member stars: %i \n'%npoint[0].size)

    res_isoc, res_isoc_er = np.array([]),np.array([])
    
    if (np.ravel(npoint).size > 5 and Gmag[ind_m].size < 10000):
    
        res_isoc, res_isoc_er = fit_isochrone(dirout+name+'/'+outfile+'_members.dat', 
                                              verbosefile,guess, magcut, 
                                              obs_plx=clusters['Plx'][i],
                                              obs_plx_er=clusters['sigPlx'][i], 
                                              prior=prior, 
                                              bootstrap=True)
        
        fig, ax = plt.subplots()
        plt.scatter(BPmag-RPmag,Gmag,s=1,color='gray',alpha=0.4)
        plt.scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m], cmap='jet',s=2.e4*erGmag[ind_m],c=members[ind_m])
#        plt.scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m], cmap='jet',s=20*members[ind_m],c=members[ind_m])
        plt.ylim(Gmag.max(),Gmag.min()-1.)
        plt.xlim(np.nanmin(BPmag-RPmag),np.nanmax(BPmag-RPmag))
        grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag, nointerp=True)
        #grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag)
        fit_iso = make_obs_iso(filters, grid_iso, res_isoc[1], res_isoc[3], gaia_ext = True)                
        plt.plot(fit_iso['G_BPmag']-fit_iso['G_RPmag'],fit_iso['Gmag'],'g', label='best solution',alpha=0.9)
        plt.xlabel('(BPmag - RPmag)')
        plt.ylabel('Gmag')
        plt.title(name)    
        
#        mod_cluster = model_cluster(res_isoc[0],res_isoc[1],res_isoc[2],res_isoc[3],0.5,2000,filters,
#                                    'G_BPmag',error=False,Mcut=magcut,seed=2**25,
#                                    imf='chabrier',alpha=2.1, beta=-3.)
#        plt.scatter(mod_cluster['G_BPmag']-mod_cluster['G_RPmag'],mod_cluster['Gmag'], s=5,alpha=0.9)
        
        plt.savefig(dirout+name+'/'+outfile+'_isoc-fit.png', dpi=300)

    else:
        print ('Cluster has too many members for isochrone fit...')
        verbosefile.write('Cluster has too many or too few members for isochrone fit...\n')
        
    ####################################################################################
    # Write a log file 
    logfile = open(dirout+name+'/'+logfilename, "a")
    logfile.write('%+20s;'%name)
    logfile.write('    %+2s;    %+2s;    %+2s;    %+2s;    %+2s;    %+2s;'%(clusters['RAh'][i],
                                                                            clusters['RAm'][i],
                                                                            clusters['RAs'][i],
                                                                            clusters['DEg'][i],
                                                                            int(clusters['DEm'][i]),
                                                                            clusters['DEs'][i]))
    logfile.write(' %8.3f;'%clusters['RA_ICRS'][i])                               
    logfile.write(' %8.3f;'%clusters['DE_ICRS'][i]) 

    logfile.write('%8.1f;'%radius)                               
    logfile.write('%8.1f;'%c_dist[ind_m].mean().value) 

    # isochrone results
    if (Gmag[ind_m].size > 5 and Gmag[ind_m].size < 10000):
        logfile.write('%8i;'%(res_isoc[1]*1.e3))                              
        logfile.write('%8i;'%(res_isoc_er[1]*1.e3))                              
    
        logfile.write('%8.3f;'%(res_isoc[0]))                              
        logfile.write('%8.3f;'%(res_isoc_er[0]))                              
    
        logfile.write('%8.3f;'%(res_isoc[2]))                              
        logfile.write('%8.3f;'%(res_isoc_er[2]))                              
    
        logfile.write('%8.3f;'%(res_isoc[3]))                              
        logfile.write('%8.3f;'%(res_isoc_er[3]))                              
    
    else:
        for j in range(8):
            logfile.write('%8.3f;'%(np.nan))
                             
    # extinction from GAIA
    logfile.write('%8.3f;'%AG_mean)
    logfile.write('%8.3f;'%AG_std)
    
    #number of cluster members
    logfile.write('%8i;'%clusters['Nc'][i])
    #logfile.write('%8i;'%(np.sqrt(plx_res_er[0]**2 + np.std(res,axis=0)[0]**2)*Plx.size))

    # parallax used
    logfile.write('%8.3f;'%clusters['Plx'][i])
    logfile.write('%8.3f;'%clusters['sigPlx'][i])
    logfile.write('%8.3f;'%(clusters['sigPlx'][i]/np.sqrt(clusters['Nc'][i])))
    
    # astrometric solution
    logfile.write('%8.3f;'%clusters['pmRA'][i])
    logfile.write('%8.3f;'%clusters['sigpmRA'][i])
    logfile.write('%8.3f;'%(clusters['sigpmRA'][i]/np.sqrt(clusters['Nc'][i])))

    logfile.write('%8.3f;'%clusters['pmDE'][i])
    logfile.write('%8.3f;'%clusters['sigpmDE'][i])
    logfile.write('%8.3f;'%(clusters['sigpmDE'][i]/np.sqrt(clusters['Nc'][i])))

    # Radial velocity from GAIA
    logfile.write('%8.3f;'%Vr_mean)
    logfile.write('%8.3f;'%Vr_std)

    logfile.write('\n')    
    logfile.close()
            
    end = time.time()
    print ('Elapsed time:', (end - start)/60)   
    
    ####################################################################################
    # Plot VPD, CMD, etc
    fig, ax = plt.subplots(3, 2,figsize=(10,15))
    
    #ax[0,0].scatter(pmRA,pmDE,s=1,color='gray',alpha=0.4)
    ax[0,0].scatter(pmRA,pmDE,s=1,color='gray',alpha=0.4)
    ax[0,0].scatter(pmRA[ind_m], pmDE[ind_m],s=1*members[ind_m],c=members[ind_m],cmap='jet')

    ax[0,0].set_xlabel('pmRA (mas/yr)')
    ax[0,0].set_ylabel('pmDEC (mas/yr)')
    ax[0,0].set_title(name)
        
    #-------------------------------------------------------------------------------------------
    ind = np.argsort(weight)
    ax[0,1].scatter(pmRA[ind],pmDE[ind],s=10*weight[ind],c=2*weight[ind]**1, cmap='jet',alpha=0.6 )

    e_c = Ellipse(xy=[clusters['pmRA'][i],clusters['pmDE'][i]],
                  width=3*clusters['sigpmRA'][i], height=3*clusters['sigpmDE'][i],
                angle=0.,fc=None, ec='r',fill=False)
    ax[0,1].scatter(clusters['pmRA'][i],clusters['pmDE'][i],marker='+',c='w')
    
    e_f = Ellipse(xy=[pmRA.mean(),pmDE.mean()],width=3*pmRA.std(), height=3*pmDE.std(),
                angle=np.arccos(np.corrcoef(pmRA,pmDE)[0,1])*180./np.pi-90.,fc=None, ec='k',fill=False)
    #ax[0,1].add_artist(e_c)
    ax[0,1].add_artist(e_f)
    ax[0,1].set_xlabel('pmRA (mas/yr)')
    ax[0,1].set_ylabel('pmDE (mas/yr)')
    

    #-------------------------------------------------------------------------------------------

    if (np.ravel(npoint).size > 5 and Gmag[ind_m].size < 10000):
        
        ax[1,0].scatter(BPmag-RPmag,Gmag,s=1,color='gray',alpha=0.4)
        ax[1,0].scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m], cmap='jet',s=5*members[ind_m],c=members[ind_m])
        ax[1,0].set_ylim(Gmag.max(),Gmag.min()-1)
        ax[1,0].set_xlim(np.nanmin(BPmag-RPmag),np.nanmax(BPmag-RPmag))
        ax[1,0].plot(fit_iso['G_BPmag']-fit_iso['G_RPmag'],fit_iso['Gmag'],'g', label='best solution',alpha=0.9)
        ax[1,0].set_xlabel('BPmag - RPmag')
        ax[1,0].set_ylabel('Gmag')

    #-------------------------------------------------------------------------------------------
    ax[1,1].scatter(BPmag-RPmag,Gmag,s=1,color='k')
    ax[1,1].set_ylim(Gmag.max(),Gmag.min()-1)
    ax[1,1].set_xlim(np.nanmin(BPmag-RPmag),np.nanmax(BPmag-RPmag))
    ax[1,1].set_xlabel('BPmag - RPmag')
    ax[1,1].set_ylabel('Gmag')

    #-------------------------------------------------------------------------------------------
    if (Plx[ind_m].size > 3.):
        
        kernel = 'gaussian'
        X=(Plx[ind_m])[:, np.newaxis]
        X_plot = np.linspace(Plx[ind_m].min(),Plx[ind_m].max(),1000)[:, np.newaxis]
        
        band = np.min([0.025,np.abs(1.06*clusters['sigPlx'][i]*(Plx[ind_m].size)**(-1./5))])
        kde = KernelDensity(kernel=kernel, bandwidth=band).fit(X)
        log_dens = kde.score_samples(X_plot)

        ax[2,0].stackplot(X_plot[:, 0], np.exp(log_dens), alpha=0.5)
        ax[2,0].plot(X_plot[:, 0], np.exp(log_dens), alpha=0.2,label='Plx data',color='b')
        ax[2,0].plot(X[:, 0], -0.05 - 0.05 * np.random.random(X.shape[0]), '.k', markersize=1)
        #ax.text(Plx[ind_m].min(),plx_std,np.exp(log_dens).max(),'%.3f +/- %.3f'%(plx_res[1],plx_res[2]))
        at = AnchoredText('%.3f +/- %.3f'%(clusters['Plx'][i],clusters['sigPlx'][i]),loc=2, frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[2,0].add_artist(at) 

        ax[2,0].set_xlabel('Parallax (mas)')
        ax[2,0].set_ylabel('normalized density ')

    #-------------------------------------------------------------------------------------------
        
    ax[2,1].hist(members[np.isfinite(members)])
    ax[2,1].set_xlabel('Membership')
    ax[2,1].set_ylabel('counts')


    plt.tight_layout()
    
    plt.savefig(dirout+name+'/'+outfile+'_comp-figs.png', dpi=300)

    verbosefile.close()

print ('All done...')
    








