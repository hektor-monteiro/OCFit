# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import matplotlib.pyplot as plt
from oc_tools_padova import *
from scipy import stats

#data_cont = np.genfromtxt('contamination-dias6-exper.txt',skip_header=82, names=True)

#plt.scatter(data['BV'],data['V'])
#plt.ylim(18,10)

GAIA = True
UBVRI = False

if UBVRI:
    # load mod_grid
    # load full isochrone grid data and arrays of unique Age and Z values
    grid_dir = './grids/'
    filters = ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag', 'Jmag', 'Hmag', 'Kmag']
    refmag = 'Vmag'
    mod_grid, age_grid, z_grid = load_mod_grid(grid_dir)
    
if GAIA:
    grid_dir = './grids/'
    mod_grid, age_grid, z_grid = load_mod_grid(grid_dir, isoc_set='GAIA')
    filters = ['Gmag','G_BPmag','G_RPmag']
    refMag = 'Gmag'
    

age = 9.3
FeH = 0. #-0.289895
dist = 2.5
Av = 1.3#0.595023
bin_frac = 0.2
nstars = 2000
phot_err = 0.
Mlim = 19.
seed=2**25
met = (10.**FeH)*0.0152

#grid_iso = get_iso_from_grid(age,met,filters,refMag)
#fit_iso = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = True)

mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,nstars,filters,
                            refMag,error=False,Mcut=Mlim,seed=seed,
                            imf='chabrier',alpha=2.3, beta=-3.)


mod_cluster_er = get_phot_errors(mod_cluster,filters)

# generate coordinate information based on king profile
#field_ra, field_dec = gen_field_coordinates(232.45,-64.86,15., data_cont['V'].size)
cluster_ra, cluster_dec = gen_cluster_coordinates(232.45,-64.86,nstars, 5., 2000.,mod_cluster['Mini'])

# sort by V mag
indV = np.argsort(mod_cluster['Gmag'])

V = mod_cluster['Gmag']

cor = mod_cluster['G_BPmag']-mod_cluster['G_RPmag']

plt.figure()
#plt.scatter(data_cont['BV'],data_cont['V'],c='gray')
plt.scatter(cor,V,s=15,cmap='jet',c=mod_cluster['Mini'])
plt.colorbar()
plt.ylim(V.max(),V.min())

plt.figure()
plt.scatter(cor,V,s=15)
plt.ylim(V.max(),V.min())
plt.xlim(cor.min(),cor.max())

# plot isochrone used
grid_iso = get_iso_from_grid(age,(10.**FeH)*0.0152,filters,refMag, nointerp=False)
fit_iso = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = True)           
fit_iso_ccm = make_obs_iso(filters, grid_iso, dist, Av, gaia_ext = False)           

obs_isoG = grid_iso['Gmag'] + 5.*np.log10(dist*1.e3) - 5.+ 0.85926*Av  
obs_isoBP = grid_iso['G_BPmag'] + 5.*np.log10(dist*1.e3) - 5.+ 1.06794*Av  
obs_isoRP = grid_iso['G_RPmag'] + 5.*np.log10(dist*1.e3) - 5.+ 0.65199*Av  
ind = np.argsort(fit_iso['Mini'])
plt.plot(fit_iso['G_BPmag'][ind]-fit_iso['G_RPmag'][ind],fit_iso['Gmag'][ind], 'k',label='com polinomio',alpha=0.5)
#plt.plot(fit_iso_ccm['G_BPmag']-fit_iso_ccm['G_RPmag'],fit_iso_ccm['Gmag'], 'b',label='com CCM',alpha=0.9)
#plt.plot(obs_isoBP-obs_isoRP,obs_isoG,'k', label='sem polinomio')

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

f=open('synth-cluster-Vlim-20-GAIA-parsec1.2sc.txt','ab')

# write out cluster stars

if UBVRI:
    head = ['id', 'raj2000', 'dej2000', 'u', 'su', 'b', 'sb', 'v', 'sv', 'r', 'sr', 'i', 'si', 
            'ub', 'sub', 'bv', 'sbv', 'vr', 'svr', 'ri', 'sri', 'j', 'h', 'ks', 'sj', 'sh', 
            'sks', 'magUCAC', 'mura', 'mude', 'smura', 'smude', 'sra1', 'sde1', 'sra2', 'sde2', 'p']
    f.write('     '.join(head)+'\n')

    for i in range(nstars):
        line = np.concatenate([[i,cluster_ra[i], cluster_dec[i],mod_cluster['Umag'][i],mod_cluster_er['Umag'][i],
            mod_cluster['Bmag'][i],mod_cluster_er['Bmag'][i],mod_cluster['Vmag'][i],mod_cluster_er['Vmag'][i],
            mod_cluster['Rmag'][i],mod_cluster_er['Rmag'][i],mod_cluster['Imag'][i],mod_cluster_er['Imag'][i]],
            [0.0]*23, [np.nan]])
        np.savetxt(f,[line], fmt=' '.join(['%i'] + ['%f']*2 + ['%2.3f']*34))
    
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

