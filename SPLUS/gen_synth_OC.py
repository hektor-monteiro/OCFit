# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import matplotlib.pyplot as plt
from SPLUS_tools_V2 import *
from scipy import stats
import sys

#data_cont = np.genfromtxt('contamination-dias6-exper.txt',skip_header=82, names=True)

#plt.scatter(data['BV'],data['V'])
#plt.ylim(18,10)

GAIA = True
UBVRI = False

# load full isochrone grid data and arrays of unique Age and Z values
grid_dir = os.getenv("HOME")+'/OCFit/grids/'
mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='SPLUS')
    
filters = ['gSDSSmag','rSDSSmag','iSDSSmag','zSDSSmag']
refmag = 'gSDSSmag'


age = 8.
FeH = 0. #-0.289895
dist = 0.246
Av = 0.124
bin_frac = 0.5
nstars = 200
phot_err = 0.01
Mlim = 20.
seed=None #2**28
met = (10.**FeH)*0.0152

#grid_iso = get_iso_from_grid(age,met,filters,refmag)
#fit_iso = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = True)

mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,nstars,filters,
                            refmag,error=False,Mcut=Mlim,seed=seed,
                            imf='chabrier',alpha=2.1, beta=-3.)



# sort by V mag
indV = np.argsort(mod_cluster[refmag])

yMag = mod_cluster[refmag]
cor = mod_cluster[filters[0]]-mod_cluster[filters[1]]

cor_obs = obs_oc[filters[0]]-obs_oc[filters[1]]
yMag_obs = obs_oc[refmag]

#plt.figure()
#plt.scatter(data_cont['BV'],data_cont['V'],c='gray')

# plot isochrone used
grid_iso = get_iso_from_grid(age,(10.**FeH)*0.0152,filters,refmag, nointerp=False)
fit_iso = make_obs_iso(filters, grid_iso, dist, Av)           

#plt.plot(fit_iso['Bmag']-fit_iso['Vmag'],fit_iso['Vmag'], 'r',label='com polinomio',alpha=0.9)
plt.figure()
plt.scatter(cor,yMag,s=5,cmap='jet')
plt.scatter(cor_obs,yMag_obs,s=10*members,c=members,cmap='jet')
plt.ylim(20,5)
plt.plot(fit_iso[filters[0]]-fit_iso[filters[1]],fit_iso[refmag], 'r',label='com polinomio',alpha=0.9)


cor1 = mod_cluster[filters[0]]-mod_cluster[filters[1]]
cor1_obs = obs_oc[filters[0]]-obs_oc[filters[1]]

cor2 = mod_cluster[filters[1]]-mod_cluster[filters[1]]
cor2_obs = obs_oc[filters[1]]-obs_oc[filters[2]]

plt.figure()
plt.scatter(mod_cluster[filters[0]]-mod_cluster[filters[1]],mod_cluster[filters[1]]-mod_cluster[filters[2]],s=5,cmap='jet')
plt.plot(fit_iso[filters[0]]-fit_iso[filters[1]],fit_iso[filters[1]]-fit_iso[filters[2]], 'r',label='com polinomio',alpha=0.9)
plt.scatter(cor1_obs,cor2_obs,s=10*members,c=members,cmap='jet')

sys.exit()






#fig=plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#plt.scatter(field_ra,field_dec,s=(data_cont['V']-18.)**2,c='gray',cmap='jet')
#plt.scatter(cluster_ra,cluster_dec,s=2*(max(V)-V)**2,c=(B-V),cmap='jet')
#circ = plt.Circle((277,-13.), 0.05, color='gray', fill=False,ls='dashed')
#ax.add_patch(circ)
#ax.set_aspect('equal', 'datalim')

#############################################################################
## write data out to a file
#

f=open('synth-cluster-UBVRI.txt','ab')

# write out cluster stars

if UBVRI:
    head = ['id', 'raj2000', 'dej2000', 'u', 'su', 'b', 'sb', 'v', 'sv', 'r', 'sr', 'i', 'si', 
            'ub', 'sub', 'bv', 'sbv', 'vr', 'svr', 'ri', 'sri', 'j', 'h', 'ks', 'sj', 'sh', 
            'sks', 'magUCAC', 'mura', 'mude', 'smura', 'smude', 'sra1', 'sde1', 'sra2', 'sde2', 'p']
    f.write(('     '.join(head)+'\n').encode())

    for i in range(nstars):
        line = np.concatenate([[i,cluster_ra[i], cluster_dec[i],mod_cluster['Umag'][i],mod_cluster['Umag'][i]*0.05,
            mod_cluster['Bmag'][i],mod_cluster['Bmag'][i]*0.05,mod_cluster['Vmag'][i],mod_cluster['Vmag'][i]*0.05,
            mod_cluster['Rmag'][i],mod_cluster['Rmag'][i]*0.05,mod_cluster['Imag'][i],mod_cluster['Imag'][i]*0.05],
            [0.0]*23, [np.random.uniform(0.5,1.)]])
        np.savetxt(f,[line], fmt=' '.join(['%i'] + ['%f']*2 + ['%2.3f']*34))
    f.close()
    
if GAIA:
    head = ['id', 'raj2000', 'dej2000', 'Gmag', 'sGmag', 'G_BPmag', 'sG_BPmag', 
            'G_RPmag', 'sG_RPmag', 'B_Tmag', 'sB_Tmag', 'V_Tmag', 'sV_Tmag', 
            'Jmag', 'sJmag', 'Hmag', 'sHmag', 'Ksmag', 'sKsmag', 
            'mura', 'mude', 'smura', 'smude', 'p']
    
    f.write('     '.join(head)+'\n')

    for i in range(nstars):
        line = np.concatenate([[i,cluster_ra[i], cluster_dec[i],mod_cluster['Gmag'][i],mod_cluster_er['Gmag'][i],
            mod_cluster['G_BPmag'][i],mod_cluster_er['G_BPmag'][i],mod_cluster['G_RPmag'][i],mod_cluster_er['G_RPmag'][i],
            mod_cluster['B_Tmag'][i],mod_cluster_er['B_Tmag'][i],mod_cluster['V_Tmag'][i],mod_cluster_er['V_Tmag'][i]],
            [0.0]*10, [np.nan]])
        
        np.savetxt(f,[line], fmt=' '.join(['%i'] + ['%f']*2 + ['%2.3f']*21))
  
## write out field stars
#    
#v = data_cont['V']
#b = data_cont['BV']+v
#r = v-data_cont['VR']
#i = v-data_cont['VI']
#u = data_cont['UB']+b
#
#for k in range(v.size):
#    line = np.concatenate([[k+nstars,field_ra[k], field_dec[k],u[k],u[k]*0.02,b[k],b[k]*0.02,v[k],v[k]*0.02,
#            r[k],r[k]*0.02,i[k],i[k]*0.02],
#            [0.0]*23, [np.nan]])
#    
#    np.savetxt(f,[line], fmt=' '.join(['%i'] + ['%f']*2 + ['%2.3f']*34))
#  

f.close()
#############################################################################


# make kde plot
ra = np.concatenate([cluster_ra, field_ra])
dec = np.concatenate([cluster_dec, field_dec])

values = np.vstack([ra,dec])
kernel = stats.gaussian_kde(values)

x = ra
y = dec
z = kernel(values)
f, ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].tripcolor(x,y,z)
ax[1].tricontourf(x,y,z, 60) # choose 20 contour levels, just to show how good its interpolation is
#ax[1].plot(x,y, '.')
#ax[0].plot(x,y, '.')

plt.show()

