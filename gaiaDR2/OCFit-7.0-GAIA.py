#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
esse programa ajusta isocronas aos dados gaia dr2
busca todos os arquivos .txt (ja formatados) de um diretorio
ajusta a iso em cada um deles
escreve os resultados num log e tambem no diretorio de cada arquivo

esta configurado para os dados Cantat-2018

@author: wilton
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

import random
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
import astropy.coordinates as coord

from oc_tools_padova import *
from gaia_dr2_tools import *

from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.integrate import simps
import glob
import os
import statistics
from statsmodels import robust

from dustmaps.bayestar import BayestarQuery
from dustmaps.planck import PlanckQuery
from dustmaps.config import config
config['data_dir'] = '/home/hmonteiro/ownCloud/work/clusters/gaia-dr2/extinction/dustmaps' 

warnings.filterwarnings("ignore")

#os.system('rm -r ~/.astropy/cache/astroquery/Vzier/*')
#os.environ["OMP_NUM_THREADS"] = "1"

plt.close('all')


dir = os.getenv("HOME")+'/ownCloud/work/clusters/gaia-dr2/catalog-7.0/no0_memb3D_fof-comp/'
dirout = dir+'results/' 
logfilename = 'log-results-CANTAT-2018-no0_memb3D_fof-comp.txt'


############################################################################
# Get Apogee data from Vizier

v = Vizier(columns=['**'],row_limit=-1)
if not os.path.isfile(dirout+'/APOGEE-Metalicity-table.npy'):
    res = v.get_catalogs('J/A+A/623/A80/table2')
    apogee = np.array(res[0])
    np.save(dirout+'/APOGEE-Metalicity-table.npy', apogee)
else:
    # lugar para ler o arquivo de entrada texto
    print ('reading file from disk...')
    apogee = np.load(dirout+'/APOGEE-Metalicity-table.npy')
    print ('done.')

############################################################################
# Get Netopil data from Vizier
v = Vizier(columns=['**'],row_limit=-1)
if not os.path.isfile(dirout+'/Netopil16-Metalicity-table.npy'):
    res = v.get_catalogs('J/A+A/585/A150/tablea1')
    netopil = np.array(res[0])
    np.save(dirout+'/Netopil16-Metalicity-table.npy', netopil)
else:
    # lugar para ler o arquivo de entrada texto
    print ('reading file from disk...')
    netopil = np.load(dirout+'/Netopil16-Metalicity-table.npy')
    print ('done.')
    
netopil['Cluster'] = np.array([x.decode().replace(" ", "_") for x in netopil['Cluster']])

############################################################################



logfile = open(dirout+logfilename, "a+")
logfile.write(time.strftime("%c")+'\n')
logfile.write(' \n')
logfile.write('             name;       RA_ICRS;     DE_ICRS;       dist;     e_dist;    age;     e_age;     FeH;    e_FeH;     Av;      e_Av;      AG;      e_AG;       Nc;         RV;      e_RV;   NRV;   REFmembers;   REFRv;   REFparametersfund;     \n')
logfile.close


REFmembers = 1 # = cantat2018
REFRV = 1 # referencia para RV = 1 = nos
REFfundamentalparameters = 1 # referencia para parametros fund (d, age, FeH, Av) # 1 = nos


files = [f for f in sorted(glob.glob(dir + "data/*.dat", recursive=True),key=os.path.getsize)]

# Read Ebv for prior
ebv_gaia_2mass = np.genfromtxt(dir+'catalog-V7.0-2019-stilism-ebv.dat',
                               delimiter=',',names=True,dtype=None)


contfiles = 0
logfile = open(dirout+'/'+logfilename, "a")

for i in range(len(files)):
    
    filename_w_ext = os.path.basename(files[i])
    name, file_extension = os.path.splitext(filename_w_ext)
    print('Rodando o cluster:',name)
    
    # create directory for results of the cluster
    try:
        os.stat(dirout+name)
    except:
        os.mkdir(dirout+name)       
    
    
    # file to save all output
    verbosefile = open(dirout+name+'/'+'verbose-output.dat', 'w') 
    
    magcut = 40.
    guess = False
   
    obs = np.genfromtxt(files[i],names=True)
    
    #remove nans para fazer os plots
    cond1 = np.isfinite(obs['Gmag'])
    cond2 = np.isfinite(obs['BPmag'])
    cond3 = np.isfinite(obs['RPmag'])
    
    cond4 = obs['RFG'] > 50.0
    cond5 = obs['RFBP'] > 20.0
    cond6 = obs['RFRP'] > 20.0
    cond7 = obs['E_BR_RP_'] < 1.3+0.06*(obs['BPRP'])**2
    cond8 = obs['E_BR_RP_'] > 1.0+0.015*(obs['BPRP'])**2
    cond9 = obs['Nper'] > 8
       
    ind  = np.where(cond1&cond2&cond3&cond4&cond5&cond6&cond7&cond8&cond9)
    
    obs = obs[ind]
    
    
    Gmag = obs['Gmag']
    BPmag = obs['BPmag']
    RPmag = obs['RPmag']
    sGmag = obs['e_Gmag']
    sBPmag = obs['e_BPmag']
    sRPmag = obs['e_RPmag']
    members = obs['Pmemb']
    weight = obs['Pmemb']
    
    ###########################################################################
    # Apply photometry correction as suggested in GAIA site
    
    Gmag[(Gmag>6.)] = Gmag[(Gmag>6.)] - 0.0032*(Gmag[(Gmag>6.)]-6.)
    BPmag[Gmag>10.9] = BPmag[Gmag>10.9] - 0.005
    BPmag[Gmag<10.9] = BPmag[Gmag<10.9] - 0.026
    RPmag = RPmag - 0.012

    ###########################################################################
    member_cut = 0.5
    if (members.max() <= member_cut):
        print(name,' cluster has no stars with P>0.5')
        continue
    ind_m = members > member_cut
    
    Nmembers = len(ind_m)
   
    RAmean = statistics.median(obs['RA_ICRS'])
    DEmean = statistics.median(obs['DE_ICRS'])
    
    Plx = obs['Plx']
    erPlx = obs['e_Plx']
  
    Plxmean = Plx.mean()
    Plxsig =  Plx.std()

    
    AG = obs['AG']
    AG_mean = np.nanmean(AG[ind_m])
    AG_std = np.nanstd(AG[ind_m])

    Rv = obs['RV']
    erRv = obs['e_RV']
    Rv_mean = np.nanmean(Rv[ind_m])
    Rv_std = np.nanstd(Rv[ind_m])
    NstarsRv = 0
    
    if (np.isfinite(Rv_mean) and np.isfinite(Rv_std)):
        
        indf = np.isfinite(Rv[ind_m])
        Rv_mean = np.sum((Rv[ind_m]/erRv[ind_m])[indf])/np.sum((1./erRv[ind_m])[indf])
        Rv_std = (Rv[ind_m])[indf].size/np.sum((1./erRv[ind_m])[indf])
        NstarsRv = np.count_nonzero(indf)

    filters = ['Gmag','G_BPmag','G_RPmag']
    refmag = 'Gmag'
    
    ####################################################################################
    # Plot decontaminated CMD
    fig, ax = plt.subplots()   
    plt.scatter(BPmag-RPmag,Gmag,s=1,color='gray',alpha=0.4)
    plt.scatter((BPmag-RPmag)[ind_m], Gmag[ind_m],s=1.0e4*sGmag[ind_m],c=members[ind_m],cmap='jet')
    plt.ylim(Gmag.max()+0.5,Gmag.min()-0.5)
    plt.xlabel(r'$G_{BP} - G_{RP}$')
    plt.ylabel(r'$G$')
    
    plt.text((BPmag-RPmag)[ind_m].min(),Gmag[ind_m].min(),name)  
    plt.savefig(dirout+name+'/'+name+'_CMD_members.png', dpi=300)

    ####################################################################################
    #     fit isochrone to the data
    
    guess_dist = infer_dist(Plx[ind_m]+0.029, erPlx[ind_m],guess=1./Plx[ind_m].mean())
    print('Infered distance from parallax: %8.3f \n'%(guess_dist))
    #    verbosefile.write('Infered distance from parallax: %8.3f \n'%(guess_dist))
    dist_posterior_x=[]
    dist_posterior_y=[]
    for d in np.linspace(0.01,3*guess_dist,1000): 
        dist_posterior_x.append(d)
        dist_posterior_y.append(-likelihood_dist(d,Plx[ind_m]+0.029, erPlx[ind_m]))
    dist_posterior_x = np.array(dist_posterior_x)
    dist_posterior_y = np.array(dist_posterior_y)
    dist_posterior_y[dist_posterior_y<0.]=0
    cum = np.cumsum(dist_posterior_y)/np.sum(dist_posterior_y)
    #    conf_int = np.where((cum > 0.16)&(cum<0.84))[0]
    conf_int = np.where((cum > 0.16)&(cum<0.5))[0]
    try:
    #        dist_guess_sig = (dist_posterior_x[conf_int[-1]] - dist_posterior_x[conf_int[0]])/2.
        dist_guess_sig = (dist_posterior_x[conf_int[-1]] - dist_posterior_x[conf_int[0]])
    
    except:
        print('using rough distance interval estimate...')
        if (obs['Plx'][i] > 1*obs['e_Plx'][i]):
            dist_guess_sig = ( 1./(obs['Plx'][i]-1*obs['e_Plx'][i]) - 
                              1./(obs['Plx'][i]+1*obs['e_Plx'][i]) )/2.
        else:
            dist_guess_sig = np.min([0.5*guess_dist,1.])

    magcut = BPmag[ind_m].max()
    
    print ('Mag. Cut:')
    print(magcut)

    verbosefile.write('Mag. Cut: \n')
    verbosefile.write(str(magcut)+'\n')
         
    print ('Membership Cut:')
    print(member_cut)

    verbosefile.write('Membership Cut: \n')
    verbosefile.write(str(member_cut)+'\n')
         
       
    # para guess Av: 
    try:
        
        ind_ebv = np.where(ebv_gaia_2mass['source_id'] == name.encode())
        guess_Av = (3.1*ebv_gaia_2mass['ebv'][ind_ebv])[0]
        guess_Av_sig = (3.1*np.abs(ebv_gaia_2mass['e_ebv_max'][ind_ebv]+ebv_gaia_2mass['e_ebv_min'][ind_ebv])/2.)[0]
        guess_Av_fac = (ebv_gaia_2mass['dist'][ind_ebv]/ebv_gaia_2mass['distance'][ind_ebv])
    
        print('Aprox. Av from E(B-V) map using Rv=3.1: %8.3f +/- %8.3f  \n'%(guess_Av, guess_Av_sig))
        verbosefile.write('Aprox. Av from E(B-V) map using Rv=3.1: %8.3f +/- %8.3f  \n'%(guess_Av, guess_Av_sig))
    except:
        print('Cluster not in 3D extinction catalog...')
        verbosefile.write('Cluster not in 3D extinction catalog...')
        guess_Av = 1.0
        guess_Av_sig = 1.0e3
        guess_Av_fac = 1.
                
    if (guess_Av_fac <= 1.  or guess_Av <= 1.5):
        Av_guess = guess_Av
        Av_guess_sig = guess_Av_sig
    else:
        coords = coord.SkyCoord(RAmean*u.deg,DEmean*u.deg, distance=guess_dist*u.kpc)
        planck = PlanckQuery()
        ebvp = planck(coords)
        Av_guess = (guess_Av+ebvp*3.1)/2
        Av_guess_sig = 0.5*Av_guess
        
        print('Planck E(B-V) = {:.3f}  mag  Av = {:.3f}'.format(ebvp,3.1*ebvp))
        verbosefile.write('Planck E(B-V) = {:.3f}  mag  Av = {:.3f} \n'.format(ebvp,3.1*ebvp))

    # check to see if there is a metalicity determinantion to use in prior
    
    # priority is for Netopil HighRes
    if (name.encode() in netopil['Cluster']):
        inda = np.where(netopil['Cluster'] == name.encode())[0][0]
        if (np.isfinite(netopil['__Fe_H_HQS'][inda])):
            guess_FeH =netopil['__Fe_H_HQS'][inda]
            if(np.isfinite(netopil['e__Fe_H_HQS'][inda]) and netopil['e__Fe_H_HQS'][inda] > 0.):
                guess_FeH_sig = netopil['e__Fe_H_HQS'][inda]
            else:
                guess_FeH_sig = 0.1
            print('Using Netopil HQS FeH value in prior...')
            verbosefile.write('Using Netopil HQS FeH value in prior...\n')
        elif(np.isfinite(netopil['__Fe_H_LQS'][inda])):
            guess_FeH =netopil['__Fe_H_LQS'][inda]
            if(np.isfinite(netopil['e__Fe_H_LQS'][inda]) and netopil['e__Fe_H_LQS'][inda] > 0.):
                guess_FeH_sig = netopil['e__Fe_H_LQS'][inda]
            else:
                guess_FeH_sig = 0.1
        else:
            guess_FeH = 0.0
            guess_FeH_sig = 0.1
            print('Using Netopil LQS FeH value in prior...')
            verbosefile.write('Using Netopil LQS FeH value in prior...\n')
            
    elif (name.encode() in apogee['Cluster']):
        inda = np.where(apogee['Cluster'] == name.encode())[0][0]
        guess_FeH =apogee['__Fe_H_'][inda]  
        guess_FeH_sig = apogee['e__Fe_H_'][inda]
        if (~np.isfinite(guess_FeH)):
            guess_FeH = 0.
            guess_FeH_sig = 1.e3
        print('Using Apogee FeH value in prior...')
        verbosefile.write('Using Apogee FeH value in prior...\n')
        
    else:
        # get galactic coordinates for apogee sample
        coords = coord.SkyCoord(RAmean*u.deg,DEmean*u.deg, distance=guess_dist*u.kpc)
        c2 = coords.transform_to(coord.Galactocentric)    
        gal_coords = [c2.x.to(u.kpc),c2.y.to(u.kpc),c2.z.to(u.kpc)]    
        GCradius = np.sqrt(gal_coords[0]**2+gal_coords[1]**2).value
        
        # calculate FeH from gradient
        if(GCradius < 13.9): 
            guess_FeH = -0.068 * (GCradius-8.0)
            guess_FeH_sig = 0.1
        else:
            guess_FeH = -0.009 * (GCradius-13.9) - 0.4
            guess_FeH_sig = 0.1
      
        print('Galactocentric radius: ',GCradius)
        verbosefile.write('Galactocentric radius: %8.2f \n'%GCradius)
        print('FeH prior from Galactic Gradient: %8.2f +- %8.2f'%(guess_FeH,guess_FeH_sig))
        verbosefile.write('FeH prior from Galactic Gradient: %8.2f +- %8.2f \n'%(guess_FeH,guess_FeH_sig))

    guess = [8.0,guess_dist,guess_FeH,Av_guess]
    guess_sig = np.array([1.e3, dist_guess_sig, guess_FeH_sig, Av_guess_sig])      
    
#    guess = [8.0,guess_dist,guess_FeH,Av_guess]
#    guess_sig = np.array([1.e3, dist_guess_sig, 1.e3, Av_guess_sig])      
#    guess_sig = np.array([1.e3, 1.e3, 1.e3, 1.e3])      
      
#    guess = [8.0,guess_dist,guess_FeH,Av_guess]
#    guess_sig = np.array([1.e3, dist_guess_sig, 1.e3, 1.e3])      

    prior = np.stack([guess,guess_sig])
    
    print ('Guess:')
    print(guess)
    print('Prior sig: ')
    print(guess_sig)
    
    verbosefile.write('Guess: \n')
    verbosefile.write(str(guess)+'\n')
    
    verbosefile.write('Prior sigma: \n')
    verbosefile.write(str(guess_sig)+'\n')
    
#    print ('Mag. Cut:')
#    print(magcut)
#    
#    verbosefile.write('Mag. Cut: \n')
#    verbosefile.write(str(magcut)+'\n')
    
    npoint = np.where(BPmag[ind_m] < magcut)
    print ('number of member stars:', npoint[0].size)
    verbosefile.write('number of member stars: %i \n'%npoint[0].size)
   
    res_isoc, res_isoc_er = np.array([]),np.array([])

    if (np.ravel(npoint).size > 5 and Gmag[ind_m].size < 10000):
    
        res_isoc, res_isoc_er = fit_iso_GAIA(files[i],verbosefile,
                                             guess, magcut, member_cut,
                                             obs_plx=Plxmean,
                                             obs_plx_er=Plxsig, 
                                             prior=prior, 
                                             bootstrap=True,
                                             #fixFeH=guess_FeH+1.0e-6)
                                             fixFeH=False)
    
   
    # to salve in a log file:
        logfile.write('{:<20s};'.format(name))                          
        
        logfile.write('%10.4f;'%RAmean)
        logfile.write('%10.4f;'%DEmean)                      
        
        logfile.write('%8i;'%(res_isoc[1]*1.e3))                              
        logfile.write('%8i;'%(res_isoc_er[1]*1.e3))                              
    
        logfile.write('%8.3f;'%(res_isoc[0]))                              
        logfile.write('%8.3f;'%(res_isoc_er[0]))                              
    
        logfile.write('%8.3f;'%(res_isoc[2]))                              
        logfile.write('%8.3f;'%(res_isoc_er[2]))                              
    
        logfile.write('%8.3f;'%(res_isoc[3]))                              
        logfile.write('%8.3f;'%(res_isoc_er[3]))                              
    
        logfile.write('%8.3f;'%AG_mean)                              
        logfile.write('%8.3f;'%AG_std)                              
 
        logfile.write('%8i;'%Nmembers)                              
                                     
        logfile.write('%8.3f;'%Rv_mean)                              
        logfile.write('%8.3f;'%Rv_std)                              

        logfile.write('%8i;'%NstarsRv)                              
                              
        logfile.write('%8i;'%REFmembers)
        logfile.write('%8i;'%REFRV)
        logfile.write('%8i;'%REFfundamentalparameters) 
        logfile.write(' \n')
        logfile.close
                              

    
    
    # CMD   
        fig, ax = plt.subplots()
        plt.scatter(BPmag-RPmag,Gmag,s=1,color='gray',alpha=0.4)
        plt.scatter((BPmag-RPmag)[ind_m], Gmag[ind_m],s=10*members[ind_m],c=members[ind_m],cmap='jet')
        plt.ylim(Gmag.max()+0.5,Gmag.min()-1.)
        plt.xlim(np.nanmin(BPmag-RPmag)-0.3,np.nanmax(BPmag-RPmag)+0.3)
        
        grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag, Abscut=False, nointerp=False)
        fit_iso = make_obs_iso(filters, grid_iso, res_isoc[1], res_isoc[3], gaia_ext = True)                
        plt.plot(fit_iso['G_BPmag']-fit_iso['G_RPmag'],fit_iso['Gmag'],'g',   label='best solution',alpha=0.9)
        plt.xlabel(r'$G_{BP} - G_{RP}$')
        plt.ylabel(r'$G$')
        plt.text((BPmag-RPmag)[ind_m].min(),Gmag[ind_m].min(),name)
        plt.savefig(dirout+name+'/'+name+'_CMD_isocfit.png', dpi=300)
    

    # cor - cor
        fig, ax = plt.subplots()
        plt.scatter(BPmag-RPmag,Gmag-RPmag,s=1,color='gray',alpha=0.4)
        plt.scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m]-RPmag[ind_m], cmap='jet',s=1.0e4*sGmag[ind_m],c=members[ind_m])
        plt.ylim(np.nanmin(Gmag-RPmag)-0.3,np.nanmax(Gmag-RPmag)+0.3)
        plt.xlim(np.nanmin(BPmag-RPmag)-0.3,np.nanmax(BPmag-RPmag)+0.3)
        grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag, Abscut=False, nointerp=False)
        fit_iso = make_obs_iso(filters, grid_iso, res_isoc[1], res_isoc[3], gaia_ext = True)                
        plt.plot(fit_iso['G_BPmag']-fit_iso['G_RPmag'],fit_iso['Gmag']-fit_iso['G_RPmag'],'g',   label='best solution',alpha=0.9)
        plt.xlabel(r'$G_{BP} - G_{RP}$')
        plt.ylabel(r'$G - G_{RP}$')
        plt.text((BPmag-RPmag)[ind_m].min(),Gmag[ind_m].min(),name)
        plt.savefig(dirout+name+'/'+name+'_CCD_isocfit.png', dpi=300)

        verbosefile.close()
    

  
print ('All done...')
    
    
    
    
    
    
    
    
    
