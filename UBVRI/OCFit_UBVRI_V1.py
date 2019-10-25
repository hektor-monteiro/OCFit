#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:40:23 2019
last update 11 oct 2019
@author: Hektor & Wilton

to change the number of runs go to function UBVRI_tools line 708
To determine reliable errors set bootstrap = True at line 79
 the input file must have at least the following columns:
 U       SU     B      SB     V       SV     R       SR     I      SI  P

(use the file synth-cluster-UBVRI.txt as example)

The parameter of the synthetic file are:
age = 8.3
FeH = 0.2 #-0.289895
dist = 2.500
Av = 1.5
bin_frac = 0.5

The code uses the parallaxes of the members as prior to estimate the distance.
To chage the priors go to line  168. Note that very large sigma values imply that prior is not used.

The code print the CMD and color-color plots with memberships in the output directory before making isochrone fits.

"""
import numpy as np
from matplotlib import pyplot as plt
from UBVRI_tools_V1 import *
import os
import time

plt.close('all')

########################################################################################

# directory where the codes are 
dir = os.getenv("HOME")+'/OCFit/UBVRI/'
dirout = dir+'Results/'
# create directory for results
try:
    os.stat(dirout)
except:
    os.mkdir(dirout)       

data_dir = dir+'data/'

#name of the cluster being fit
name =  'IC4651'

# file containing the observational data
obs_file = data_dir+'Dias6-UBVRI+GAIA.dat'
obs_file = data_dir+'NGC2635-UBVRI+Pgaia.dat'
obs_file = data_dir+'IC4651_18 _IC_4651_data_stars.dat_Equal.dat'

# magcut = stars with mags greater than this value will not be used
magcut = 20.


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

guess = False

obs = np.genfromtxt(obs_file,names=True)

#remove nans
cond5 = np.isfinite(obs['U'])
cond6 = np.isfinite(obs['B'])
cond7 = np.isfinite(obs['V'])
cond8 = np.isfinite(obs['R'])
cond9 = np.isfinite(obs['I'])
cond10 = obs['P'] > 0.001  # prob cut

ind  = np.where(cond5&cond6&cond7&cond8&cond9&cond10)

obs = obs[ind]


ind_m = obs['P'] > 0.51
Plx = obs['Plx']
erPlx = obs['e_Plx']

###################################################################
#plot CMD of members
refmag = 'Vmag'

Umag = obs['U']
Bmag = obs['B']
Vmag = obs['V']
Rmag = obs['R']
Imag = obs['I']
sUmag = obs['SU']
sBmag = obs['SB']
sVmag = obs['SV']
sRmag = obs['SR']
sImag = obs['SI']
members = obs['P']
ind_m = members > .9


# B-V versus V    
fig, ax = plt.subplots()
plt.scatter(Bmag-Vmag,Vmag,s=1,color='gray',alpha=0.4)
plt.scatter(Bmag[ind_m]-Vmag[ind_m],Vmag[ind_m], cmap='jet',s=4.e2*sVmag[ind_m],c=members[ind_m])
plt.ylim(Vmag.max()+0.5,Vmag.min()-1.)
plt.xlim(np.nanmin(Bmag-Vmag)-0.3,np.nanmax(Bmag-Vmag)+0.3)
plt.xlabel('B-V')
plt.ylabel('V')
#    plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_membership-BVxV.png', dpi=300)

# color - color
fig, ax = plt.subplots()
plt.scatter(Bmag-Vmag,Umag-Bmag,s=1,color='gray',alpha=0.4)
plt.scatter(Bmag[ind_m]-Vmag[ind_m],Umag[ind_m]-Bmag[ind_m], cmap='jet',s=4.e2*sVmag[ind_m],c=members[ind_m])
plt.ylim((Umag-Bmag).max()-0.5,(Umag-Bmag).min()-0.5)
plt.xlim(np.nanmin(Bmag-Vmag)-0.3,np.nanmax(Bmag-Vmag)+0.3)
plt.xlabel('B-V')
plt.ylabel('U-B')
plt.savefig(dirout+name+'/'+name+'_membership-BVxUB.png', dpi=300)
##################################################################################


###################################################################
#distance prior from parallaxe data
####################################################################################
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
            
guessparameters = [8.0,guess_dist,0.0,1.]
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

print ('number of member stars:', Vmag[ind_m].size)
verbosefile.write('number of member stars: %i \n'%Vmag[ind_m].size)
    
#################################################################


res_isoc, res_isoc_er = fit_isochroneUBVRI(obs_file, verbosefile, guess=False,magcut=20.0, obs_plx=False, 
              obs_plx_er=0.05,prior=np.array([[1.],[1.e6]]),bootstrap=True)


###############################################################################
filters = ['Umag','Bmag','Vmag','Rmag','Imag']
refmag = 'Vmag'

Umag = obs['U']
Bmag = obs['B']
Vmag = obs['V']
Rmag = obs['R']
Imag = obs['I']
sUmag = obs['SU']
sBmag = obs['SB']
sVmag = obs['SV']
sRmag = obs['SR']
sImag = obs['SI']
members = obs['P']
ind_m = members > .51

grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag, Abscut=False)
fit_iso = make_obs_iso(filters, grid_iso, res_isoc[1], res_isoc[3])                

# B-V versus V    
fig, ax = plt.subplots()
plt.scatter(Bmag-Vmag,Vmag,s=1,color='gray',alpha=0.4)
plt.scatter(Bmag[ind_m]-Vmag[ind_m],Vmag[ind_m], cmap='jet',s=4.e2*sVmag[ind_m],c=members[ind_m])
plt.ylim(Vmag.max()+0.5,Vmag.min()-1.)
plt.xlim(np.nanmin(Bmag-Vmag)-0.3,np.nanmax(Bmag-Vmag)+0.3)
plt.plot(fit_iso['Bmag']-fit_iso['Vmag'],fit_iso['Vmag'],'g',   label='best solution',alpha=0.9)
plt.xlabel('B-V')
plt.ylabel('V')
#    plt.title(name)    
plt.savefig(dirout+name+'/'+name+'_BVxV.png', dpi=300)



# cor - cor
fig, ax = plt.subplots()
plt.scatter(Bmag-Vmag,Umag-Bmag,s=1,color='gray',alpha=0.4)
plt.scatter(Bmag[ind_m]-Vmag[ind_m],Umag[ind_m]-Bmag[ind_m], cmap='jet',s=4.e2*sVmag[ind_m],c=members[ind_m])
plt.ylim((Umag-Bmag).max()-0.5,(Umag-Bmag).min()-0.5)
plt.xlim(np.nanmin(Bmag-Vmag)-0.3,np.nanmax(Bmag-Vmag)+0.3)
plt.plot(fit_iso['Bmag']-fit_iso['Vmag'],fit_iso['Umag']-fit_iso['Bmag'],'g', alpha=0.9)
plt.xlabel('B-V')
plt.ylabel('U-B')
plt.savefig(dirout+name+'/'+name+'_BVxUB.png', dpi=300)


# V-R versus V    
fig, ax = plt.subplots()
plt.scatter(Vmag-Rmag,Vmag,s=1,color='gray',alpha=0.4)
plt.scatter(Vmag[ind_m]-Rmag[ind_m],Vmag[ind_m], cmap='jet',s=4.e2*sVmag[ind_m],c=members[ind_m])
plt.ylim(Vmag.max()+0.3,Vmag.min()-1.)
plt.xlim(np.nanmin(Vmag-Rmag)-0.3,np.nanmax(Vmag-Rmag)+0.3)
plt.plot(fit_iso['Vmag']-fit_iso['Rmag'],fit_iso['Vmag'],'g', alpha=0.9)
plt.xlabel('V-R')
plt.ylabel('V')
plt.savefig(dirout+name+'/'+name+'_VRxV.png', dpi=300)


# V-I versus V    
fig, ax = plt.subplots()
plt.scatter(Vmag-Imag,Vmag,s=1,color='gray',alpha=0.4)
plt.scatter(Vmag[ind_m]-Imag[ind_m],Vmag[ind_m], cmap='jet',s=4.e2*sVmag[ind_m],c=members[ind_m])
plt.ylim(Vmag.max()+0.3,Vmag.min()-1.)
plt.xlim(np.nanmin(Vmag-Imag)-0.3,np.nanmax(Vmag-Imag)+0.3)
plt.plot(fit_iso['Vmag']-fit_iso['Imag'],fit_iso['Vmag'],'g', alpha=0.9)
plt.xlabel('V-I')
plt.ylabel('V')
plt.savefig(dirout+name+'/'+name+'_VIxV.png', dpi=300)


# U-B versus V    
fig, ax = plt.subplots()
plt.scatter(Umag-Bmag,Vmag,s=1,color='gray',alpha=0.4)
plt.scatter(Umag[ind_m]-Bmag[ind_m],Vmag[ind_m], cmap='jet',s=4.e2*sVmag[ind_m],c=members[ind_m])
plt.ylim(Vmag.max()+0.3,Vmag.min()-1.)
plt.xlim(np.nanmin(Umag-Bmag)-0.3,np.nanmax(Umag-Bmag)+0.3)
plt.plot(fit_iso['Umag']-fit_iso['Bmag'],fit_iso['Vmag'],'g',  label='best solution',alpha=0.9)
plt.xlabel('U-B')
plt.ylabel('V')
plt.savefig(dirout+name+'/'+name+'_UBxV.png', dpi=300)

verbosefile.close()
logfile.close()

print ('DONE!')