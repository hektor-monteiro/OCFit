#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:40:23 2019
last update 20 oct 2019
@author: Hektor & Wilton



magcut = stars with Vmag greater than this value are not used in the fit
probcut = stars with membership probality than this value are not used in the fit

To convert Vega to ABmags, add
        uJAVA 1.116
        F378 0.489
        F395 -0.019
        F410 -0.168
        F430 -0.135
        gSDSS -0.102
        F515 -0.061
        rSDSS 0.150
        F660 0.312
        iSDSS 0.389
        F861 0.530
        zSDSS 0.562

"""
import numpy as np
from matplotlib import pyplot as plt
from SPLUS_tools_V2 import *
import os
import time
import sys

plt.close('all')

########################################################################################

# directory where the codes are 
dir = os.getenv("HOME")+'/OCFit/SPLUS/'
dirout = dir+'Results/'
# create directory for results
try:
    os.stat(dirout)
except:
    os.mkdir(dirout)       

data_dir = dir+'data/'

# file containing the observational data
file = 'Blanco1Splus+cantat-2.csv'

obs_file = data_dir+file

#name of the cluster being fit
name =  'Blanco_1'



# magcut = stars with mags greater than this value will not be used
magcut = 50.
probcut = 0.5

# filters = ['gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag']
# refmag = 'gSDSSmag'

# filters = ['rSDSSmag','iSDSSmag','zSDSSmag']
# refmag = 'rSDSSmag'

# filters = ['gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag','F515mag', 'F660mag', 'F861mag']
# refmag = 'gSDSSmag'

filters = ['F515mag', 'F660mag', 'F861mag']
refmag = 'F515mag'


guess = None
########################################################################################




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


#verbosefile.write('Starting isochrone fitting...\n')


obs = np.genfromtxt(obs_file,names=True,dtype=None,delimiter=',')

# Convert from Vegamags to AB mags

obs['g_auto'] = obs['g_auto'] + 0.102
obs['r_auto'] = obs['r_auto'] - 0.150
obs['i_auto'] = obs['i_auto'] - 0.389
obs['z_auto'] = obs['z_auto'] - 0.562
obs['F515_auto'] = obs['F515_auto'] + 0.061
obs['F660_auto'] = obs['F660_auto'] - 0.312
obs['F861_auto'] = obs['F861_auto'] - 0.530

# Setup obs data to be fit   
#obs_oc = np.copy(obs[['g_aper','r_aper','i_aper','z_aper']])
#obs_oc_er = np.copy(obs[['eg_aper','er_aper','ei_aper','ez_aper']])

obs_oc = np.copy(obs[['g_auto','r_auto','i_auto','z_auto','F515_auto','F660_auto','F861_auto']])
obs_oc_er = np.copy(obs[['eg_auto','er_auto','ei_auto','ez_auto','eF515_auto','eF660_auto','eF861_auto']])

obs_oc.dtype.names=['gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag','F515mag','F660mag','F861mag']
obs_oc_er.dtype.names=['gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag','F515mag','F660mag','F861mag']


## repalce -99 values for n.nan
for filt in filters:
    obs_oc[filt][np.abs(obs_oc[filt] - (-99.)) < 1.] = np.nan
    obs_oc[filt][np.abs(obs_oc[filt] - (99.)) < 1.] = np.nan
    obs_oc_er[filt][np.abs(obs_oc_er[filt] - 0.) < 1.0e-10] = np.nan
    

Plx = obs['Plx']
erPlx = obs['e_Plx']
members = obs['Pmemb']
weight = obs['Pmemb']
ind_m = members > probcut


# Plot cmd
    
fig, ax = plt.subplots()
color = obs_oc['gSDSSmag']-obs_oc['zSDSSmag']
Ymag = obs_oc['gSDSSmag']

plt.scatter(color,Ymag,s=10*members,c=members,cmap='jet')
plt.ylim(np.nanmax(Ymag)+0.5,np.nanmin(Ymag)-0.5)
plt.xlim(np.nanmin(color)-0.3,np.nanmax(color)+0.3)
plt.xlabel('g-z')
plt.ylabel('g')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_cmd.png', dpi=300)
    
# Plot color-color
    
fig, ax = plt.subplots()
color1 = obs_oc['gSDSSmag']-obs_oc['rSDSSmag']
color2 = obs_oc['rSDSSmag']-obs_oc['iSDSSmag']

plt.scatter(color1,color2,s=10*members,c=members,cmap='jet')
plt.ylim(np.nanmin(color2)-0.3,np.nanmax(color2)+0.3)
plt.xlim(np.nanmin(color1)-0.3,np.nanmax(color1)+0.3)
plt.xlabel('g-r')
plt.ylabel('i-z')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_ccd.png', dpi=300)
    

##############################################################################
#distance prior from parallaxe data
##############################################################################
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
    if (clusters['Plx'][i] > 1*clusters['e_Plx'][i]):
        dist_guess_sig = ( 1./(clusters['Plx'][i]-1*clusters['e_Plx'][i]) - 
                          1./(clusters['Plx'][i]+1*clusters['e_Plx'][i]) )/2.
    else:
        dist_guess_sig = np.min([0.5*guess_dist,1.])
            
guessparameters = [8.8,guess_dist,0.0,1.]
guess_sig = np.array([1.e3, dist_guess_sig, 1.e3, 1.e3])  # prior values = [age,dist,Fe/H,Av]     


prior = np.stack([guessparameters,guess_sig])             # sigma of the prior values

print ('prior:')
print(guessparameters)
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

print ('number of member stars:', Plx[ind_m].size)
verbosefile.write('number of member stars: %i \n'%Plx[ind_m].size)


#################################################################


res_isoc, res_isoc_er = fit_isochrone(obs_oc,obs_oc_er, weight, filters, refmag, verbosefile, 
                                           probcut, guess=False,magcut=20.0, 
                                           obs_plx=False, obs_plx_er=Plx.std(),
                                           prior=np.array([[1.],[1.e6]]),
                                           bootstrap=False)

###############################################################################

# get isochrone of best fit
grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag, Abscut=False)
fit_iso = make_obs_iso(filters, grid_iso, res_isoc[1], res_isoc[3])                

# Plot cmd
    
fig, ax = plt.subplots()
color = obs_oc['gSDSSmag']-obs_oc['zSDSSmag']
Ymag = obs_oc['gSDSSmag']

plt.scatter(color,Ymag,s=10*members,c=members,cmap='jet')
plt.ylim(np.nanmax(Ymag)+0.5,np.nanmin(Ymag)-0.5)
plt.xlim(np.nanmin(color)-0.3,np.nanmax(color)+0.3)
plt.plot(fit_iso['gSDSSmag']-fit_iso['zSDSSmag'],fit_iso['gSDSSmag'])
plt.xlabel('g-z')
plt.ylabel('g')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_cmd_gxgz.png', dpi=300)
    
fig, ax = plt.subplots()
color = obs_oc['gSDSSmag']-obs_oc['rSDSSmag']
Ymag = obs_oc['gSDSSmag']

plt.scatter(color,Ymag,s=10*members,c=members,cmap='jet')
plt.ylim(np.nanmax(Ymag)+0.5,np.nanmin(Ymag)-0.5)
plt.xlim(np.nanmin(color)-0.3,np.nanmax(color)+0.3)
plt.plot(fit_iso['gSDSSmag']-fit_iso['rSDSSmag'],fit_iso['gSDSSmag'])
plt.xlabel('g-r')
plt.ylabel('g')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_cmd_gxgr.png', dpi=300)
    
fig, ax = plt.subplots()
color = obs_oc['gSDSSmag']-obs_oc['iSDSSmag']
Ymag = obs_oc['gSDSSmag']

plt.scatter(color,Ymag,s=10*members,c=members,cmap='jet')
plt.ylim(np.nanmax(Ymag)+0.5,np.nanmin(Ymag)-0.5)
plt.xlim(np.nanmin(color)-0.3,np.nanmax(color)+0.3)
plt.plot(fit_iso['gSDSSmag']-fit_iso['iSDSSmag'],fit_iso['gSDSSmag'])
plt.xlabel('g-i')
plt.ylabel('g')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_cmd_gxgi.png', dpi=300)
    
fig, ax = plt.subplots()
color = obs_oc['F660mag']-obs_oc['F861mag']
Ymag = obs_oc['F515mag']

plt.scatter(color,Ymag,s=10*members,c=members,cmap='jet')
plt.ylim(np.nanmax(Ymag)+0.5,np.nanmin(Ymag)-0.5)
plt.xlim(np.nanmin(color)-0.3,np.nanmax(color)+0.3)
plt.plot(fit_iso['F660mag']-fit_iso['F861mag'],fit_iso['F515mag'])
plt.xlabel('F660mag-F861mag')
plt.ylabel('F515mag')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_cmd_narrow.png', dpi=300)


# Plot color-color
    
fig, ax = plt.subplots()
color1 = obs_oc['gSDSSmag']-obs_oc['rSDSSmag']
color2 = obs_oc['iSDSSmag']-obs_oc['zSDSSmag']

plt.scatter(color1,color2,s=10*members,c=members,cmap='jet')
plt.plot(fit_iso['gSDSSmag']-fit_iso['rSDSSmag'],fit_iso['iSDSSmag']-fit_iso['zSDSSmag'])
plt.ylim(np.nanmin(color2)-0.3,np.nanmax(color2)+0.3)
plt.xlim(np.nanmin(color1)-0.3,np.nanmax(color1)+0.3)
plt.xlabel('g-r')
plt.ylabel('i-z')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_ccd_grxiz.png', dpi=300)

fig, ax = plt.subplots()
color1 = obs_oc['gSDSSmag']-obs_oc['rSDSSmag']
color2 = obs_oc['rSDSSmag']-obs_oc['iSDSSmag']

plt.scatter(color1,color2,s=10*members,c=members,cmap='jet')
plt.plot(fit_iso['gSDSSmag']-fit_iso['rSDSSmag'],fit_iso['rSDSSmag']-fit_iso['iSDSSmag'])
plt.ylim(np.nanmin(color2)-0.3,np.nanmax(color2)+0.3)
plt.xlim(np.nanmin(color1)-0.3,np.nanmax(color1)+0.3)
plt.xlabel('g-r')
plt.ylabel('r-i')
plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_ccd_grxri.png', dpi=300)

verbosefile.close()
logfile.close()

print ('DONE!')
