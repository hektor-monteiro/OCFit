#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:21:11 2017

@author: hmonteiro
"""
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
import os
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


############################################################################
# Get data from Vizier for each DALM cluster

dirout = '/home/hmonteiro/OCFit/results/'

tipos=[('name', 'S17'), ('coord', 'S19'), ('class', 'S6'), ('diam', '<f8'), ('dist', '<f8'), 
       ('f0', '?'), ('ebv', '<f8'), ('age', '<f8'), ('pmRA', '<f8'), ('e_pmRA', '<f8'), 
       ('pmDE', '<f8'), ('e_pmDE', '<f8'), ('Nc', '<i8'), ('ref', 'S4'), ('Rv', '<f8'), 
       ('eRV', '<f8'), ('N', '<i8'), ('ref2', 'S8'), ('ME', 'S9'), ('eME', 'S8'), ('Nme', 'S5'), 
       ('TrTyp', 'S11')]

clusters = np.genfromtxt('DALM-clusters.csv', delimiter=';',names=True,dtype=tipos)
#clusters = np.genfromtxt('newOC.csv', delimiter=';',names=True,dtype=tipos)

############################################################################

todo = ['NGC 2425']

ind_todo = []
for oc in todo:
    print(oc,np.where(clusters['name'] == oc.encode()))
    ind_todo.append( (np.where(clusters['name'] == oc.encode())[0][0] ) )
    
clusters = clusters[ind_todo]

############################################################################

log_res = []
log_res_er = []


for i in range(clusters['name'].size):
    
    plt.close('all')
    
    start = time.time()
    
    name, radius, coord, dist = clusters['name'][i].decode("utf-8").replace(" ", "_"), (clusters['diam'][i]+2.)/2., \
                                clusters['coord'][i].decode("utf-8"), clusters['dist'][i]
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

    if (clusters['diam'][i]/2. < 1500.):
                    
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

    outfile = name+'_data'
    
    Vizier.ROW_LIMIT = 999999999
    Vizier.TIMEOUT = 120
    #Vizier.COLUMNS = ['*','pmRApmDEcor']
    v = Vizier(columns=['*', 'pmRApmDEcor','PlxpmRAcor','PlxpmDEcor',
                        'RFG','RFBP','RFRP','E(BR/RP)','Nper'],
               row_limit=999999999)
    #Vizier.VIZIER_SERVER = 'vizier.cfa.harvard.edu'
    #Vizier.VIZIER_SERVER = 'vizier.u-strasbg.fr'
    if not os.path.isfile(dirout+name+'/'+outfile+'.npy'):
        res=v.query_region(SkyCoord(coord, unit=(u.hourangle, u.deg)), radius=radius*u.arcmin,catalog='I/345/gaia2')
        data = np.array(res[0])
        np.save(dirout+name+'/'+outfile, data)
    else:
        # lugar para ler o arquivo de entrada texto
        print ('reading file...')
        data = np.load(dirout+name+'/'+outfile+'.npy')
        print ('done.')
    
    #sys.exit()
    #res = Vizier.query_region("dias 6", radius=10.*u.arcmin,catalog='I/345/gaia2')
    
    ############################################################################
    
    sig_clip = 3.
    
    # Clean sample
    er_lim = 10.5
    cond1 = np.abs(data['pmRA']-np.nanmean(data['pmRA'])) < sig_clip*np.nanstd(data['pmRA'])
    cond2 = np.abs(data['pmDE']-np.nanmean(data['pmDE'])) < sig_clip*np.nanstd(data['pmDE'])

    cond3 = data['RFG'] > 50.0
    cond4 = data['RFBP'] > 20.0
    cond5 = data['RFRP'] > 20.0
    
    cond6 = data['E_BR_RP_'] < 1.3+0.06*(data['BP-RP'])**2
    cond7 = data['E_BR_RP_'] > 1.0+0.015*(data['BP-RP'])**2
    
    cond8 = np.abs(data['e_pmDE']/data['pmDE']) < er_lim
    cond9 = np.isfinite(data['pmRA'])
    cond10 = np.isfinite(data['pmDE'])
    cond11 = np.isfinite(data['Plx'])
    cond12 = np.abs(data['e_pmRA']/data['pmRA']) < er_lim
    cond13 = data['Gmag'] < 16.
    
    cond14 = data['Nper'] > 8
        
    ind  = np.where(cond1&cond2&cond3&cond4&cond5&cond6&cond7&cond8&
                    cond9&cond10&cond11&cond12&cond13&cond14)
    
#    ind  = np.where(cond1&cond2&cond8&
#                    cond9&cond10&cond11&cond12&cond13&cond14)
    
    # check to see if it has been done
    if (ind[0].size < 5):
        print ('Cluster ', name, ' with too few stars. Skiping...')
        verbosefile.write('Cluster %s with too few stars, skipping...\n'%name)
        
        logfile = open(dirout+name+'/'+logfilename, "a")
        logfile.write('Cluster '+'%s'%name+' with too few stars. Skiping...\n')
        logfile.write('\n')

        continue

    
    print ('number of stars: ', ind[0].size)
    verbosefile.write('number of stars: %i \n'%ind[0].size)
    

    if(ind[0].size > 100000):
        print ('Cluster with too many stars... skipping')
        verbosefile.write('Cluster with too many stars... skipping')
        continue

    pmRA = (data['pmRA'])[ind]
    erRA = (data['e_pmRA'])[ind]
    
    pmDE = (data['pmDE'])[ind]
    erDE = (data['e_pmDE'])[ind]
    
    Plx = (data['Plx'])[ind]

    Gmag = (data['Gmag'])[ind]
    BPmag = (data['BPmag'])[ind]
    RPmag = (data['RPmag'])[ind]
    
    Rv = (data['RV'])[ind]
    
    pmRApmDEcor = (data['pmRApmDEcor'])[ind]
    PlxpmRAcor = (data['PlxpmRAcor' ])[ind]
    PlxpmDEcor = (data['PlxpmDEcor'])[ind]

    err = np.sqrt(data['e_pmRA']**2+data['e_pmDE']**2)
    err = err[ind]
    
    weight = np.min(err)/(err)
        
    ###################################################################
    # PM and Parallax density distribution    
    values = np.vstack([pmRA, pmDE, Plx])
    PM_dens = stats.gaussian_kde(values, bw_method='silverman')
    PM_dens.set_bandwidth(bw_method=PM_dens.factor / 5.)
    
    weight = PM_dens(values) 

    ###################################################################
    prange = np.array([[0.001,0.9],
                       [.05,.25],
                       [.05,.25],
#                       [pmRA.min(),pmRA.max()],
#                       [pmDE.min(),pmDE.max()],
                       np.sort([0.9*pmRA[np.argmax(weight)],1.1*pmRA[np.argmax(weight)]]),
                       np.sort([0.9*pmDE[np.argmax(weight)],1.1*pmDE[np.argmax(weight)]]),
#                       [-10,-5.],
#                       [5,10.],
                       [-0.9,0.9],
                       [.1,1.5*pmRA.std()],
                       [.1,1.5*pmDE.std()],
                       [-15,15.],
                       [-15,15.],
                       [-0.9,0.9]])
#       
    midpoint = (prange[:,1]-prange[:,0])/2.+prange[:,0]
    
    ndim = prange.shape[0]
    
    # define CE tweak parameters
    nruns = 5
    itmax = 100    
    sample = 1000
    
    band = 0.15
    alpha = 0.2
    tol = 1.e-40
    
    if (np.isfinite(clusters['pmRA'][i])&np.isfinite(clusters['pmDE'][i])):
        guess = [0.1,0.1,0.1,clusters['pmRA'][i],clusters['pmDE'][i],0.,pmRA.std(),
                 pmDE.std(),pmRA.mean(),pmDE.mean(),0.]
    else:
        guess=False
    
    guess=False
    res = np.ndarray([nruns,ndim])
    
    fit_data = (pmRA,pmDE,erRA,erDE,pmRApmDEcor)
    
    for n in range(nruns):
        res[n,:] = run_CE(likelihood_PM,fit_data,weight,sample,prange,itmax,band,alpha,tol,
           mp.cpu_count()-1,guess=guess,pm=True)
        
        verbosefile.write(str(res[n,:])+'\n')
        
        guess = res[n,:].tolist()
            
    print ('')
    print ('-------------------------------------------------------------')
    print (' Final result')
    print ('-------------------------------------------------------------')
    print ('   '.join('%0.3f' % v for v in np.median(res,axis=0)))
    print ('   '.join('%0.3f' % v for v in res.std(axis=0)))
    print ('')
    
    verbosefile.write('\n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write(' Final result \n')
    verbosefile.write('-------------------------------------------------------------\n')
    verbosefile.write('   '.join('%0.3f' % v for v in np.median(res,axis=0))+'\n')
    verbosefile.write('   '.join('%0.3f' % v for v in res.std(axis=0))+'\n')
    verbosefile.write('\n')

    ####################################################################################
    # Plot final result
    
    x = np.median(res,axis=0)
    x_er = np.std(res,axis=0)
    
    log_res.append([name,radius,x])
    log_res_er.append([name,radius,res.std(axis=0)])

    
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    # show 1sigma ellipse
    e_c = Ellipse(xy=[x[3],x[4]],width=x[1], height=x[2],
                angle=0.,fc=None, ec='r',fill=False)
    
    e_f = Ellipse(xy=[x[8],x[9]],width=x[6], height=x[7],
                angle=np.arccos(x[10])*180./np.pi-90.,fc=None, ec='k',fill=False)
    
    ind = np.argsort(weight)
    ax.scatter(pmRA[ind],pmDE[ind],s=10*weight[ind],c=2*weight[ind]**1, cmap='jet',alpha=0.6 )
    #    ax.scatter(pmRA,pmDE,s=1)
    ax.scatter(x[3],x[4],s=20,color='r')
    ax.scatter(x[8],x[9],s=20,color='k')
    
    ax.add_artist(e_c)
    ax.add_artist(e_f)
    
    ax.set_xlabel('pmRA')
    ax.set_ylabel('pmDE')
    
    plt.savefig(dirout+name+'/'+outfile+'_PMfinal.png', dpi=300)
    
    ####################################################################################
    ## Calculate kernel density

    values = np.vstack([pmRA, pmDE])

    # Perform a kernel density estimate on the data using sklearn.
    PM_dens = stats.gaussian_kde(values, bw_method='silverman')
    PM_dens.set_bandwidth(bw_method=PM_dens.factor / 5.)
    PM_dens_cut = PM_dens(values)    

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    # show 1sigma ellipse
    e_c = Ellipse(xy=[x[3],x[4]],width=3*x[1], height=3*x[2],
                angle=0.,fc=None, ec='r',fill=False)
    
    e_f = Ellipse(xy=[x[8],x[9]],width=3*x[6], height=3*x[7],
                angle=np.arccos(x[10])*180./np.pi-90.,fc=None, ec='k',fill=False)
    #np.arccos(x[10])*180./np.pi-90.
    
    ax.add_artist(e_c)
    ax.add_artist(e_f)
    
    ax.tricontourf(pmRA,pmDE,PM_dens_cut, 60)
    ax.set_xlabel('pmRA')
    ax.set_ylabel('pmDE')
    pmRAdens,pmDEdens = pmRA,pmDE
    
    plt.savefig(dirout+name+'/'+outfile+'_PM2Dhist-cut.png', dpi=300)
    
    ####################################################################################
    # calculate membership probability

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
    
#    ind  = np.where(cond1&cond2&cond3&cond4&cond5&cond11)
    
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

    Rv = (data['RV'])[ind]
    erRv = (data['e_RV'])[ind]
    AG = (data['AG'])[ind]
    
    ra = (data['RA_ICRS'])[ind]*u.degree
    dec = (data['DE_ICRS'])[ind]*u.degree
    
    gaiaID = (data['Source'])[ind]
 
    pmRApmDEcor = (data['pmRApmDEcor'])[ind]
    PlxpmRAcor = (data['PlxpmRAcor' ])[ind]
    PlxpmDEcor = (data['PlxpmDEcor'])[ind]

    err = np.sqrt(data['e_pmRA']**2+data['e_pmDE']**2)
    err = err[ind]
    
    weight = np.min(err)/(err)
    
    ###################################################################
    # PM and Parallax density distribution    
    values = np.vstack([pmRA, pmDE, Plx])
    PM_dens = stats.gaussian_kde(values, bw_method='silverman')
    PM_dens.set_bandwidth(bw_method=PM_dens.factor / 5.)
    
    weight = PM_dens(values) 

    ###################################################################
    
    fit_data = (pmRA,pmDE,erpmRA,erpmDE,pmRApmDEcor) 
    members =  PM_cluster_model(x,fit_data) / (PM_cluster_model(x,fit_data)+
                                PM_field_model(x,fit_data))
    
    ind_m = members > .9
    if (Plx[ind_m].size <= 3):
        ind_m = members > .5
    
    
    ####################################################################################
    # Plot member parallax histogram
    
    plx_mean = np.nanmean(Plx[ind_m])
    plx_std = np.nanstd(Plx[ind_m])
    
    if (np.isfinite(plx_mean) and np.isfinite(plx_std) and Plx[ind_m].size > 3.):
        
        # fit parallax distribution
        plx_res, plx_res_er = fit_plx(Plx[ind_m],weight[ind_m])

        kernel = 'gaussian'
        X=(Plx[ind_m])[:, np.newaxis]
        X_plot = np.linspace(Plx[ind_m].min(),Plx[ind_m].max(),1000)[:, np.newaxis]
        
        phi_c = plx_res[0] * gaussian(X_plot,plx_res[1],plx_res[2]) 
        phi_f = (1. - plx_res[0]) * gaussian(X_plot,plx_res[3],plx_res[4]) 

        #X_plot = (np.sort(Plx[ind_m]))[:, np.newaxis]
        band = np.min([0.025,1.06*plx_std*(Plx[ind_m].size)**(-1./5)])
        kde = KernelDensity(kernel=kernel, bandwidth=band).fit(X)
        log_dens = kde.score_samples(X_plot)

        fig, ax = plt.subplots()
        ax.set_title(name)
        ax.stackplot(X_plot[:, 0], np.exp(log_dens), alpha=0.5)
        ax.plot(X_plot[:, 0], np.exp(log_dens), alpha=0.2,label='Plx data',color='b')
        ax.plot(X[:, 0], -0.05 - 0.05 * np.random.random(X.shape[0]), '.k', markersize=1)
        ax.plot(X_plot,phi_c,label='phi_c')
        ax.plot(X_plot,phi_f,label='phi_f')
        ax.legend(loc='upper right')
        #ax.text(Plx[ind_m].min(),plx_std,np.exp(log_dens).max(),'%.3f +/- %.3f'%(plx_res[1],plx_res[2]))
        at = AnchoredText('%.3f +/- %.3f'%(plx_res[1],plx_res[2]),loc=2, frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at) 
       
        plt.savefig(dirout+name+'/'+outfile+'_Plx-hist.png', dpi=300)
            
    fit_data = (pmRA,pmDE,Plx,erpmRA,erpmDE,erPlx,pmRApmDEcor,PlxpmRAcor,PlxpmDEcor,weight)
    pars = np.array([plx_res[0]*x[0],x[3],x[4],plx_res[1],x[1],x[2],plx_res[2]*3.,x[8],x[9],np.nanmean(Plx),
            x[6],x[7],np.nanstd(Plx)])
            
    print ('Nc after parallax fit: ', plx_res[0]*x[0])
    verbosefile.write('Nc after parallax fit: %i \n'%(plx_res[0]*x[0]))
    
    members =  membership_3D(pars,fit_data)
    
    # seleciona corte no membership
 
    ind_m = members > .5 

    ####################################################################################
    # save all good data to file
    np.savez(dirout+name+'/'+name+'_var_save.npz', ra.value,dec.value,pmRA,pmDE,erpmRA,erpmDE,Plx,erPlx,Gmag,BPmag,RPmag,
             erGmag,erBPmag,erRPmag,Rv,erRv,AG,pmRApmDEcor,PlxpmRAcor,PlxpmDEcor)

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
        plt.text(AG_mean+AG_std,count.max(),'%.3f +/- %.3f'%(AG_mean,AG_std))
        plt.title(name)    
        plt.savefig(dirout+name+'/'+outfile+'_AG-hist.png', dpi=300)

    
    ####################################################################################
    # Plot Radial velocity histogram
    Rv_mean = np.nanmean(Rv[ind_m])
    Rv_std = np.nanstd(Rv[ind_m])
    
    if (np.isfinite(Rv_mean) and np.isfinite(Rv_std)):
        
        indf = np.isfinite(Rv[ind_m])
        Rv_mean = np.sum((Rv[ind_m]/erRv[ind_m])[indf])/np.sum((1./erRv[ind_m])[indf])
        Rv_std = (Rv[ind_m])[indf].size/np.sum((1./erRv[ind_m])[indf])
        
        fig, ax = plt.subplots()
        count, bins, patches = plt.hist((Rv[ind_m])[indf],bins='auto')
        plt.text(Rv_mean+Rv_std,count.max(),'%.3f +/- %.3f'%(Rv_mean,Rv_std))
        plt.title(name)  
        
        at = AnchoredText('%.3f +/- %.3f'%(Rv_mean,Rv_std),loc=2, frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at) 
        
        plt.savefig(dirout+name+'/'+outfile+'_Rv-hist.png', dpi=300)

    
    ####################################################################################
    # obtain star density map
    
    ra_cen = Angle(coord[0:8], unit=u.hour).to(u.degree)
    dec_cen = Angle(coord[10:25], unit=u.degree)
        
    xd,yd,dens_map,peak_ind,c_dist = star_density(ra,dec,ra_cen.value, dec_cen.value)
    
    fig, ax = plt.subplots()
    rad = Circle((0.,0.),radius=(clusters['diam'][i])/2.,
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
            'RPmag', 'e_RPmag', 'pmRA','e_pmRA','pmDE','e_pmDE','Plx','e_Plx','Rv','e_Rv','P']
    
    f.write(('     '.join(head)+'\n').encode())

    for k in range(Gmag.size):
        line = [(gaiaID)[k],(ra.value)[k], (dec.value)[k],(Gmag)[k],(erGmag)[k],
            (BPmag)[k],(erBPmag)[k],(RPmag)[k],(erRPmag)[k],(pmRA)[k],(erpmRA)[k],(pmDE)[k],(erpmDE)[k],
            (Plx)[k],(erPlx)[k],(Rv)[k],(erRv)[k],(members)[k]]
        
        np.savetxt(f,[line], fmt=' '.join(['%i'] + ['%f']*2 + ['%2.4f']*15))
        
    f.close()
    
#%%
    ####################################################################################
    #     fit isochrone to the data
    magcut = BPmag[ind_m].max()
    npoint = np.where(Gmag[ind_m] < magcut)
    guess_dist = infer_dist(Plx[ind_m], erPlx[ind_m],guess=1./plx_res[1])
    
    print('Infered distance from parallax: %8.3f \n'%(guess_dist))
    verbosefile.write('Infered distance from parallax: %8.3f \n'%(guess_dist))


    if (plx_res[1] > 3*plx_res[2]/np.sqrt(npoint[0].size)):
        dist_guess_sig = ( 1./(plx_res[1]-3*plx_res[2]/np.sqrt(npoint[0].size)) - 
                          1./(plx_res[1]+3*plx_res[2]/np.sqrt(npoint[0].size)) )/2.
    else:
        dist_guess_sig = np.min([0.5*guess_dist,1.])

    if (np.isfinite(float(clusters['age'][i])) and np.isfinite(clusters['dist'][i]) and 
        np.isfinite(float(clusters['ebv'][i]))):

        guess = [float(clusters['age'][i]),guess_dist,0.0,float(clusters['ebv'][i])*3.1]
        guess_sig = np.array([1.e3, dist_guess_sig, 1.e3, 1.e3])      
        prior = np.stack([guess,guess_sig])
    else:
        guess=False
        guess_sig = np.array([1.e3, 1.e3, 1.e3, 1.e3])
        prior=np.array([[1.],[1.e3]])

    print ('Guess:')
    print(guess)
    print('Prior sig: ')
    print(guess_sig)
    
    verbosefile.write('Guess: \n')
    verbosefile.write(str(guess)+'\n')
    
    verbosefile.write('Prior sigma: \n')
    verbosefile.write(str(guess_sig)+'\n')

    print ('Mag. Cut:')
    print(magcut)

    verbosefile.write('Mag. Cut: \n')
    verbosefile.write(str(magcut)+'\n')
    
    print ('number of member stars:', npoint[0].size)
    verbosefile.write('number of member stars: %i \n'%npoint[0].size)

    res_isoc, res_isoc_er = np.array([]),np.array([])

    if (np.ravel(npoint).size > 5 and Gmag[ind_m].size < 10000):
    
        res_isoc, res_isoc_er = fit_isochrone(dirout+name+'/'+outfile+'_members.dat', 
                                              verbosefile,guess, magcut, 
                                              obs_plx=plx_res[1], prior=prior, 
                                              bootstrap=True)
        
        fig, ax = plt.subplots()
        plt.scatter(BPmag-RPmag,Gmag,s=1,color='gray',alpha=0.4)
        plt.scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m], cmap='jet',s=2.e4*erGmag[ind_m],c=members[ind_m])
#        plt.scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m], cmap='jet',s=20*members[ind_m],c=members[ind_m])
        plt.ylim(Gmag.max(),Gmag.min()-1.)
        plt.xlim(np.nanmin(BPmag-RPmag),np.nanmax(BPmag-RPmag))
        grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag, nointerp=False)
        fit_iso = make_obs_iso(filters, grid_iso, res_isoc[1], res_isoc[3], gaia_ext = True)
        ind = np.argsort(fit_iso['Mini'])               
        plt.plot(fit_iso['G_BPmag'][ind]-fit_iso['G_RPmag'][ind],fit_iso['Gmag'][ind], 'r',label='com polinomio',alpha=0.9)
        plt.xlabel('(BPmag - RPmag)')
        plt.ylabel('Gmag')
        plt.title(name)    
        plt.savefig(dirout+name+'/'+outfile+'_isoc-fit.png', dpi=300)

    else:
        print ('Cluster has too many members for isochrone fit...')
        verbosefile.write('Cluster has too many or too few members for isochrone fit...\n')


    ####################################################################################
    # Write a log file 
    logfile = open(dirout+name+'/'+logfilename, "a")
    logfile.write('%+20s;'%name)
    logfile.write('    %+2s;    %+2s;    %+2s;    %+2s;    %+2s;    %+2s;'%tuple(coord.split()))

    logfile.write(' %8.3f;'%ra_cen.value)                               
    logfile.write(' %8.3f;'%dec_cen.value) 

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
    logfile.write('%8i;'%(plx_res[0]*x[0]*Plx.size))
    #logfile.write('%8i;'%(np.sqrt(plx_res_er[0]**2 + np.std(res,axis=0)[0]**2)*Plx.size))

    # parallax from gaussian fit to members
    logfile.write('%8.3f;'%plx_res[1])
    logfile.write('%8.3f;'%plx_res[2])
    logfile.write('%8.3f;'%(plx_res[2]/np.sqrt(plx_res[0]*x[0]*Plx.size)))
    
    # astrometric solution
    logfile.write('%8.3f;'%(np.median(res,axis=0))[3])
    logfile.write('%8.3f;'%(np.median(res,axis=0))[1])
    logfile.write('%8.3f;'%(np.median(res,axis=0)[1]/np.sqrt(plx_res[0]*x[0]*Plx.size)))

    logfile.write('%8.3f;'%(np.median(res,axis=0))[4])
    logfile.write('%8.3f;'%(np.median(res,axis=0))[2])
    logfile.write('%8.3f;'%(np.median(res,axis=0)[2]/np.sqrt(plx_res[0]*x[0]*Plx.size)))

    # Radial velocity from GAIA
    logfile.write('%8.3f;'%Rv_mean)
    logfile.write('%8.3f;'%Rv_std)

    logfile.write('\n')    
    logfile.close()
            
    end = time.time()
    print ('Elapsed time:', (end - start)/60)   
    
    ####################################################################################
    # Plot VPD, CMD, etc
    fig, ax = plt.subplots(3, 2,figsize=(10,15))
    
    ax[0,0].scatter(pmRA,pmDE,s=1,color='gray',alpha=0.4)
    ax[0,0].scatter(pmRA[ind_m], pmDE[ind_m],s=1*members[ind_m],c=members[ind_m],cmap='jet')
    #plt.ylim(Gmag.max(),Gmag.min())
    ax[0,0].set_xlabel('pm RA')
    ax[0,0].set_ylabel('pm DEC')
    ax[0,0].set_title(name)
        
    #-------------------------------------------------------------------------------------------
    e_c = Ellipse(xy=[x[3],x[4]],width=3*x[1], height=3*x[2],
                angle=0.,fc=None, ec='r',fill=False)
    
    e_f = Ellipse(xy=[x[8],x[9]],width=3*x[6], height=3*x[7],
                angle=np.arccos(x[10])*180./np.pi-90.,fc=None, ec='k',fill=False)

    ax[0,1].add_artist(e_c)
    ax[0,1].add_artist(e_f)
        
    ax[0,1].tricontourf(pmRAdens,pmDEdens,PM_dens_cut, 60)
    ax[0,1].set_xlabel('pmRA')
    ax[0,1].set_ylabel('pmDE')
    

    #-------------------------------------------------------------------------------------------

    if (np.ravel(npoint).size > 5 and Gmag[ind_m].size < 10000):
        
        ax[1,0].scatter(BPmag-RPmag,Gmag,s=1,color='gray',alpha=0.4)
        ax[1,0].scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m], cmap='jet',s=5*members[ind_m],c=members[ind_m])
        ax[1,0].set_ylim(Gmag.max(),Gmag.min()-1)
        ax[1,0].set_xlim(np.nanmin(BPmag-RPmag),np.nanmax(BPmag-RPmag))
    
        ax[1,0].plot(fit_iso['G_BPmag']-fit_iso['G_RPmag'],fit_iso['Gmag'],'g', label='best solution',alpha=0.9)
        ax[1,0].set_xlabel('BPmag - RPmag')
        ax[1,0].set_ylabel('Gmag')
        ax[1,0].legend()

    #-------------------------------------------------------------------------------------------
    ax[1,1].scatter(BPmag-RPmag,Gmag,s=1,color='k')
    ax[1,1].set_ylim(Gmag.max(),Gmag.min()-1)
    ax[1,1].set_xlim(np.nanmin(BPmag-RPmag),np.nanmax(BPmag-RPmag))
    ax[1,1].set_xlabel('BPmag - RPmag')
    ax[1,1].set_ylabel('Gmag')

    #-------------------------------------------------------------------------------------------
    if (np.isfinite(plx_mean) and np.isfinite(plx_std) and Plx[ind_m].size > 3.):
        
        kernel = 'gaussian'
        X=(Plx[ind_m])[:, np.newaxis]
        X_plot = np.linspace(Plx[ind_m].min(),Plx[ind_m].max(),1000)[:, np.newaxis]
        
        band = np.min([0.025,np.abs(1.06*plx_std*(Plx[ind_m].size)**(-1./5))])
        kde = KernelDensity(kernel=kernel, bandwidth=band).fit(X)
        log_dens = kde.score_samples(X_plot)

        ax[2,0].stackplot(X_plot[:, 0], np.exp(log_dens), alpha=0.5)
        ax[2,0].plot(X_plot[:, 0], np.exp(log_dens), alpha=0.2,label='Plx data',color='b')
        ax[2,0].plot(X[:, 0], -0.05 - 0.05 * np.random.random(X.shape[0]), '.k', markersize=1)
        #ax.text(Plx[ind_m].min(),plx_std,np.exp(log_dens).max(),'%.3f +/- %.3f'%(plx_res[1],plx_res[2]))
        at = AnchoredText('%.3f +/- %.3f'%(plx_res[1],plx_res[2]),loc=2, frameon=True)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[2,0].add_artist(at) 
    #-------------------------------------------------------------------------------------------
        
    ax[2,1].hist(members[np.isfinite(members)])
    ax[2,1].set_xlabel('Membership')


    plt.tight_layout()
    
    plt.savefig(dirout+name+'/'+outfile+'_comp-figs.png', dpi=300)

    verbosefile.close()

print ('All done...')
    








