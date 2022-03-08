import numpy as np
import pandas as pd
import glob
import requests
from bs4 import BeautifulSoup
import urllib.request 
import re
import os
import sys 
import json
from astropy.io import fits
from astropy.table import Table
import astropy.wcs as wcs
from astropy.visualization import make_lupton_rgb
import pyds9 as ds9
from scipy.ndimage import gaussian_filter
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from prepare_library import create_perobs_data, cal_ave, add_MW, confusion_clean, CSC_clean_keepcols
from test_library import class_prepare, class_train_and_classify, class_save_res, col_rename, confident_flag, find_confident, plot_classifier_matrix_withSTD, prepare_cols
from pathlib import Path
import time
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm

def dict_update(a, b):
    a.update(b)
    return a.copy()

def get_evt2_file(obsid, path='.'):
    '''
    We assume that there exists a single evt2 file in primary directory in CXC database
    '''
    
    # folders organized by last digit of obsid
    last = str(obsid)[-1]
    primary_url = f'https://cxc.cfa.harvard.edu/cdaftp/byobsid/{last}/{str(obsid)}/primary'
    
    _ = glob.glob(f'{path}/*{int(obsid):05d}*')
    if _: return f'{primary_url}/{os.path.basename(_[0])}', _[0]
    
    html_text = requests.get(primary_url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    
    evt2_list = [_.get('href') for _ in soup.find_all('a') if re.search(r'evt2', _.get('href'))]
    if len(evt2_list) != 1:
        print(f'Error: there are {len(evt2_list)} evt2 files: {evt2_list}')
        
    evt2_filename = evt2_list[0]
                
    urllib.request.urlretrieve(f'{primary_url}/{evt2_filename}', f'{path}/{evt2_filename}')
    
    return f'{primary_url}/{evt2_filename}', f'{path}/{evt2_filename}'

def xy_filter_evt2(evt2_data, ccd_split=False):
    
    X = evt2_data
    
    cols = ['ccd_id', 'x', 'y', 'energy']
    
    mask = (500 < X['energy']) & (X['energy'] < 8000)
    
    X = Table(X)[mask][cols].to_pandas()

    #X = Table(X)[cols].to_pandas()
    
    X = X.dropna(subset=['x', 'y'])
    
    if ccd_split:
        ccds = np.unique(X['ccd_id'].tolist())
    
        xy = {ccd: data[['x', 'y']].values.astype(None).T for ccd, data in X.groupby('ccd_id')}
    else:
        xy = X[['x', 'y']].values.astype(None).T
    
    return xy

def process_fits(fn):
    
    with fits.open(fn) as _:
        head = _[1].header
        evt2_data = _[1].data
    
    return evt2_data, head

# ds9 image.fits -regions load input_imageCoordinates.reg -regions system wcs -regions skyformat sexagesimal -regions save outout_skyCoordinates.reg -exit

def wcs_to_physical(fn, xy, rad, fmt):
    '''
    fmt:  fk5 | galactic | image
    
    rad in minutes
    '''
    
    d = ds9.DS9()

    d.set(f'fits {fn}')
                           
    if len(np.shape(xy))==1:
        xy = [xy] 
    
    reg = f'regions command "{fmt}; '      

    for x, y in xy: 
        
        reg += f"circle {x} {y} {rad}';"

#         ra, dec = [':'.join(row[_].strip().split()) for _ in ['ra', 'dec']]

    reg += '"' 
    
    # print(reg)
    
    d.set(reg)
    
#     print(d.get('regions system'))
#     print(d.get('regions skyformat'))
    
    d.set(f'regions system physical')
#     d.set('regions skyformat sexagesimal')
    
    reg = d.get('regions')
    
    xy = re.findall(r'\((.*?)\)', reg)
    
    xy = np.array([_.split(',') for _ in xy]).astype(float)
    
    if len(xy) == 1:
        return xy[0]
    else:
        return xy   
    
    
def get_reg_phys(fn_evt2, dat_csv):
    
    max_points = 20 # d.set(reg) has limits on length of input
    
    d = ds9.DS9()

    d.set(f'fits {fn_evt2}')
    
    d.set(f'regions system physical')

    reg_begin = 'regions command "' 
    
    reg_points = []

    for i, row in dat_csv.iterrows():

        ra, dec = [':'.join(row[_].strip().split()) for _ in ['ra', 'dec']]

        reg_points.append(f'point {ra} {dec}; ')

    reg_end = '"' 
    
    n_splits = len(reg_points) // max_points + bool(np.mod(len(reg_points), max_points))
    
    splits = np.array_split(reg_points, n_splits)
    
    for s in splits:
        
        reg = reg_begin + ''.join(s) + reg_end

        d.set(reg)
                
    # print(d.get('regions'))

    reg_phys = re.findall(r'\((.*?)\)', d.get('regions'))
    reg_phys = np.array([_.split(',') for _ in reg_phys]).astype(float)
        
    # d.set('exit')  
    
    return reg_phys

def find_obs(df_per, ra, dec):
    df_per = df_per[df_per['instrument']=='ACIS'].reset_index()
    df_per['gti_end'] = pd.to_datetime(df_per['gti_end'], format="%Y-%m-%dT%H:%M:%S")
    df_per['gti_obs'] = pd.to_datetime(df_per['gti_obs'], format="%Y-%m-%dT%H:%M:%S")
    #print(df_per[['gti_obs','gti_end']])
    df_per['duration'] = df_per.apply(lambda row: (row.gti_end - row.gti_obs).total_seconds(),axis=1)
    #print(df_per[['gti_obs','gti_end','duration']])
    #print(df_per['duration'].unique())
    df_per['ra_pnt'] = df_per.apply(lambda row:Angle(row.ra_pnt, 'hourangle').degree, axis=1)
    df_per['dec_pnt'] = df_per.apply(lambda row:Angle(row.dec_pnt, 'deg').degree, axis=1)
    df_per['sep_pnt'] = df_per.apply(lambda row: SkyCoord(row.ra_pnt*u.degree,row.dec_pnt*u.degree).separation(SkyCoord(ra*u.degree,dec*u.degree)).arcminute,axis=1)

    df_obs = df_per[['name','obsid','duration','sep_pnt']].drop_duplicates().sort_values(by='duration')
    df_obs = df_obs[df_obs.sep_pnt<5].reset_index()
    #obsids_all = df_obs['obsid']
    #print(df_obs)
    obsid_all = list(df_obs['obsid'].unique())
    #print(obsid_all)
    obsids = []
    while sorted(df_obs.loc[df_obs.obsid.isin(obsids), 'name'].unique()) != sorted(df_obs['name'].unique()): 
        obsids.append(obsid_all[-1])
        obsid_all.pop()
        print('ha')
    print(obsids,obsid_all)
    return obsids


def prepare_field(df, data_dir, query_dir, field_name, name_col='name',search_mode='cone_search'):
    
    #'''
    df_pers = create_perobs_data(df, query_dir, data_dir, name_type='CSCview', name_col=name_col, ra_col='ra',dec_col='dec',coord_format='deg')
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    df_pers.to_csv(f'{data_dir}/{field_name}_per.csv', index=False)

    df_pers = pd.read_csv(f'{data_dir}/{field_name}_per.csv', low_memory=False)

    df_pers['name'] = df_pers['name'].str.lstrip()
    df_pers['per_remove_code'] = 0

    df_ave, df_obs = cal_ave(df_pers, data_dir, dtype='field',Chandratype='CSC',verb=0)

    df_ave.to_csv(f'{data_dir}/{field_name}_ave.csv', index=False)

    df_ave = pd.read_csv(f'{data_dir}/{field_name}_ave.csv')

    # cross-match with MW catalogs
    start = time.time()
    confusion = False if search_mode == 'cone_search' else True
    add_MW(df_ave, data_dir, field_name, Chandratype='CSC',confusion =confusion)
    end = time.time() 
    print(end - start)
    #'''
    df_MW = pd.read_csv(f'{data_dir}/{field_name}_MW.csv')
    df_MW_cf = confusion_clean(df_MW,X_PU='err_ellipse_r0',Chandratype='CSC')
    df_MW_cf.to_csv(f'{data_dir}/{field_name}_MW_clean.csv',index=False)

    df_MW_cf = pd.read_csv(f'{data_dir}/{field_name}_MW_clean.csv')
    #df_ave = TD_clean_vizier(df_MW_cf, remove_codes = [1, 32, 64]) # previousl no remove_codes =2?!

    df_MW_clean = CSC_clean_keepcols(df_MW_cf, withvphas=False)
    #df_MW_clean = vphasp_to_gaia_mags(df_MW_clean)

    df_remove = df_MW_clean[df_MW_clean['remove_code']==0].reset_index(drop=True)

    #sub_cols = ['name','ra','dec','err_ellipse_r0','err_ellipse_r1','significance','flux_aper90_ave_b','flux_aper90_ave_s','flux_aper90_ave_m','flux_aper90_ave_h', \
    #     'kp_prob_b_max','var_inter_prob','Gmag','BPmag','RPmag','Jmag','Hmag','Kmag','W1mag_comb','W2mag_comb','W3mag_allwise','rgeo','rpgeo']
    #ks_intra_prob_b	kp_intra_prob_b	var_inter_prob_b	Gmag	BPmag	RPmag	pm_gaia	pmRA_gaia	pmDE_gaia	Jmag	Hmag	Kmag	W1mag_catwise	W2mag_catwise	pmRA_catwise	pmDE_catwise	W1mag_allwise	W2mag_allwise	W3mag_allwise	W4mag_allwise	rgeo	rpgeo	main_id	main_type	W1mag_unwise	W2mag_unwise	W1mag_comb	W2mag_comb	CSC_flags]
    df_remove.to_csv(f'{data_dir}/{field_name}_MW_remove.csv', index=False)
    #df_remove[sub_cols].to_csv(f'{data_dir}/{field_name}_MW_subcols.csv', index=False)
    
    return df_remove

def combine_class_result(field_name, data_dir, dir_out, class_labels):

    df_all = pd.read_csv(f'{dir_out}/classes.csv')
    df_mean = df_all.groupby('name').mean().iloc[:,:len(class_labels)]

    df_std = df_all.groupby('name').std().iloc[:,:len(class_labels)]

    df_class = df_mean.idxmax(axis=1)
    df_prob = df_mean.max(axis=1)
    df_prob_e = pd.DataFrame(data=[df_std.values[i][np.argmax(np.array(df_mean), axis=1)[i]]  for i in range(len(df_std))], columns=['Class_prob_e'])
    df_mean = df_mean.add_prefix('P_')
    df_std  = df_std.add_prefix('e_P_')

    df = pd.concat([pd.concat([df_mean, df_std, df_class, df_prob], axis=1).rename(columns={0:'Class',1:'Class_prob'}).rename_axis('name').reset_index(), df_prob_e], axis=1)


    df_MW = pd.read_csv(f'{data_dir}/{field_name}_MW_remove.csv')
    df_MW = prepare_cols(df_MW, cp_thres=0, vphas=False,gaiadata=False)
    df_per   = pd.read_csv(f'{data_dir}/{field_name}_per.csv')
    df_per['name'] = df_per['name'].str.lstrip()
    df_per = df_per[['name','ra','dec']].drop_duplicates(subset=['name'])


    df_comb = pd.merge(df, df_MW[['name','significance','Fcsc_m']], on="name")

    df_comb = pd.merge(df_comb, df_per[['name','ra','dec']], how='inner',on="name")


    df = confident_flag(df_comb, method = 'sigma-mean', class_cols=class_labels)
    df.to_csv(f'{dir_out}/{field_name}_class.csv',index=False)

    field_mw_class = pd.merge(df.drop(columns=['significance','Fcsc_m','ra','dec']), df_MW, on='name')

    return field_mw_class

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:30:37 2021

@author: Steven
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import astropy as ap
import astropy.units as u
from astropy.io import fits
from scipy.stats import gaussian_kde
import holoviews as hv
import hvplot
import hvplot.pandas
import bokeh
from test_library import prepare_cols


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

frequencies = np.array([6.492e+13, 8.901e+13, 1.389e+14, 1.804e+14, 2.427e+14, 3.859e+14, 4.822e+14, 5.867e+14, 2.220e+17, 3.772e+17, 9.190e+17])

# zero points for converting from mags to fluxes from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse
zps = np.array([171.79, 309.54, 666.80, 1024.00, 1594.00, 2461.22, 2861.30, 3478.80])

# calculate width of frequency range of band from effective width of wavelength range and reference wavelength
def width(ref, weff):
    return (ap.constants.c/((ref-weff/2)*u.angstrom) - ap.constants.c/((ref+weff/2)*u.angstrom)).to(u.Hz).value

refs = np.array([46179.05, 33682.21, 21603.09, 16457.50, 12358.09, 7829.66, 6251.51, 5124.20])
weffs = np.array([10422.66, 6626.42, 2618.87, 2509.40, 1624.32, 2842.11, 4203.60, 2333.06])

widths= width(refs, weffs)

def prepare_sed(df_mw, name_col=False):
    
    df_cols = ['W2mag', 'W1mag', 'Kmag', 'Hmag', 'Jmag', 'RPmag', 'Gmag', 'BPmag', 'Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Class']
    df_cols.append(name_col) if name_col else None
    df_mw=df_mw.loc[:,df_cols]

    spec_cols = ['Frequency', 'Mag', 'Flux', 'Class']
    spec_cols.append(name_col) if name_col else None
    df_spec=pd.DataFrame(np.zeros([df_mw.shape[0]*11,len(spec_cols)]),
                      index=pd.MultiIndex.from_product([df_mw.index, ['W2mag', 'W1mag', 'Kmag', 'Hmag', 'Jmag', 'RPmag', 'Gmag', 'BPmag', 'soft', 'medium', 'hard']]),
                      columns=spec_cols)

    df_spec.index.names=['Source', 'Band']
    idx = pd.IndexSlice

    df_spec['Source Density, Flux']=0
    df_spec['Source Density, Flux Norm']=0
    df_spec['Flux Norm']=0
    
    for i, (bandname, band) in enumerate(df_spec.groupby(level=1, sort=False)):
        
        # set frequencies and classes in new dataframe
        df_spec.loc[band.index, 'Frequency']=frequencies[i]
        df_spec.loc[band.index, 'Class']=df_mw.loc[:,'Class'].to_numpy()
        if name_col:
            df_spec.loc[band.index, name_col]=df_mw.loc[:,name_col].to_numpy()

        # print(df_spec.loc[band.index, 'Flux'])

        # set fluxes for bands with mags
        if i in range(0,8):
            df_spec.loc[band.index, 'Mag']=df_mw.iloc[:,i].to_numpy()
            df_spec.loc[band.index, 'Flux']=zps[i]*pow(10,-df_spec.loc[band.index, 'Mag']/2.5)*widths[i]*1e-23

        # set fluxes for Chandra bands
        if i in range(8,11):
            df_spec.loc[band.index, 'Flux']=df_mw.iloc[:,i].to_numpy()
        # print(df_spec.loc[band.index, 'Flux'])

        # False if nan or 0 for a source in this band
        mask=~np.logical_or(np.isnan(df_spec['Flux']), df_spec['Flux']==0)

        # calculate density of source fluxes for each band, for each class
        for j, (classname, classgroup) in enumerate(df_spec.groupby(['Class'], sort=False)):

            # intersection between ith band and jth class
            idx=band.index.intersection(classgroup.index)
            # intersection between idx and sources with flux at ith band
            idx2=idx.intersection(mask.loc[mask==True].index)

            #print(bandname, classname, idx.shape, df_spec.loc[idx2].empty)

            # if only 1 source in class in band, set density=0.5
            if df_spec.loc[idx2].shape[0]==1:
                df_spec.loc[idx2, 'Source Density, Flux'] = 0.5

            # skip if no source in class in band
            elif not df_spec.loc[idx2].empty:
                df_spec.loc[idx2, 'Source Density, Flux'] = gaussian_kde(np.log10(df_spec.loc[idx2, 'Flux']))(np.log10(df_spec.loc[idx2, 'Flux']))

                # normalize by max density of each band, such that the densest region of each band will have value of 1
                df_spec.loc[idx2, 'Source Density, Flux'] = df_spec.loc[idx2, 'Source Density, Flux']/df_spec.loc[idx2, 'Source Density, Flux'].max()

    # Normalize fluxes to Chandra medium band
    for i, (bandname, band) in enumerate(df_spec.groupby(level=1, sort=False)):

        df_spec.loc[band.index, 'Flux Norm']=df_spec.loc[band.index, 'Flux']/df_spec.xs('medium', level=1).loc[:,'Flux']

        df_spec.loc[band.index, 'Flux Norm']

        # False if nan or 0 or inf for a source in this band
        mask=~((np.isnan(df_spec['Flux Norm'])) | (df_spec['Flux Norm']==0) | (np.isinf(df_spec['Flux Norm'])) )

        # calculate density of source normalized fluxes for each band, for each class
        for j, (classname, classgroup) in enumerate(df_spec.groupby(['Class'], sort=False)):

            # intersection between ith band and jth class
            idx=band.index.intersection(classgroup.index)
            # intersection between idx and sources with flux at ith band
            idx2=idx.intersection(mask.loc[mask==True].index)

            #print(bandname, classname, idx.shape, df_spec.loc[idx2].empty)
            # skip if medium band
            if bandname=='medium':
                df_spec.loc[idx2, 'Source Density, Flux Norm'] = 0.5

            # if only 1 source in class in band, set density=0.5
            elif df_spec.loc[idx2].shape[0]==1:
                df_spec.loc[idx2, 'Source Density, Flux Norm'] = 0.5

            # skip if no source in class in band
            elif not df_spec.loc[idx2].empty:
                df_spec.loc[idx2, 'Source Density, Flux Norm'] = gaussian_kde(np.log10(df_spec.loc[idx2, 'Flux Norm']))(np.log10(df_spec.loc[idx2, 'Flux Norm']))

                # normalize by max density of each band, such that the densest region of each band will have value of 1
                df_spec.loc[idx2, 'Source Density, Flux Norm'] = df_spec.loc[idx2, 'Source Density, Flux Norm']/df_spec.loc[idx2, 'Source Density, Flux Norm'].max()


    df_spec=df_spec.sort_index(1, 'Frequency')
    df_spec['Log Flux']=np.log10(df_spec['Flux'])
    df_spec['Source Density, Flux']=df_spec['Source Density, Flux'].replace(0,np.nan)
    df_spec['Flux Norm']=df_spec['Flux Norm'].replace(0,np.nan)
    df_spec['Log Flux Norm']=np.log10(df_spec['Flux Norm'])                
                
    return df_mw, df_spec
    
def plot_sed(TD_spec, field_spec, dir_plot, plot_class='YSO', save_html=False, name_col=False):
    
    #print(plot_class)
    #%%
    #'''
    #classes = ['AGN', 'NS', 'BINARY-NS', 'CV', 'LM-STAR', 'HM-STAR', 'LMXB', 'HMXB', 'YSO']
    #classes = ['AGN', 'NS', 'CV', 'LM-STAR', 'HM-STAR', 'LMXB', 'HMXB', 'YSO']

    scale_down=2

    #for c in classes:
    TD_hover_cols = ['Flux', 'Band']
    
    TD_hover_cols.append(name_col) if name_col else None
    spectrum = (TD_spec.loc[TD_spec['Class']==plot_class]).hvplot.scatter('Frequency','Flux Norm',
            logx=True,
            logy=True,
            # ylim=(1e-17, 1e-10),
            c='Source Density, Flux Norm',
            cmap="gist_rainbow",
            clabel='Source Density',
            xlabel='Frequency (Hz)',
            ylabel='Flux, Normalized to m-band',
            title='Normalized Spectra, '+plot_class,
            size=100/scale_down,
            width=int(2000/scale_down),
            height=int(2000/scale_down),
            fontscale=4/scale_down,
            hover_cols=TD_hover_cols
            ).opts(
            colorbar_opts={'width': int(70/scale_down), "label_standoff": int(20/scale_down)}
            )

#     spectrum = (df_spec.loc[df_spec['Class']==c]).hvplot.violin(y='Log Flux Norm', by=['Frequency', 'Band'],
#             title='Violin Plot of Normalized Fluxes, '+c,
#             xlabel='Frequency (Hz)',
#             ylabel='Log(Flux), Normalized to m-band',
#             width=int(2000/scale_down),
#             height=int(2000/scale_down),
#             fontscale=4/(1.25*scale_down),
#             ).sort(by='Frequency')
    
    field_scatter = (field_spec.loc[field_spec['Class']==plot_class]).hvplot.scatter('Frequency','Flux Norm',
            logx=True,
            logy=True,
            #ylim=(1e-17, 1e-10),
            color='k',
            size=200/scale_down,                                                                   
            marker='d',                                                                      
            #cmap="gist_rainbow",
            #clabel='Source Density',
            #xlabel='Frequency (Hz)',
            #ylabel='Flux, Normalized to m-band',
            #title='Normalized Spectra, '+c,
            #size=100/scale_down,
            #width=int(2000/scale_down),
            #height=int(2000/scale_down),
            #fontscale=4/scale_down,
            hover_cols=['name','Flux', 'Band']
            )#.opts(
            #colorbar_opts={'width': int(70/scale_down), "label_standoff": int(20/scale_down)}
            #)

    overlay = spectrum * field_scatter
    
    if save_html:
        hv.save(overlay, f'{dir_plot}/{plot_class}_spectrum.html')
        
    return overlay
    #'''
    
def plot_bbsed(TD, field, dir_plot, plot_class='YSO', save_class=['YSO','AGN'], confidence=True, TD_name_col=False):
    
    TD_mw, TD_spec = prepare_sed(TD, name_col=TD_name_col)
    #print(TD_spec.loc[0])
    if confidence:
        field = field[field.conf_flag>0].reset_index(drop=True)
    field_mw, field_spec = prepare_sed(field, name_col='name')
    
    #print(field_spec.loc[0])
    
    for s_class in save_class:
        plot_sed(TD_spec, field_spec, dir_plot, plot_class=s_class, save_html=True)
        
    return plot_sed(TD_spec, field_spec, dir_plot, plot_class=plot_class, save_html=False)
    
    #return None#plot
    #%%

def plot_class_matrix(field_name, df, dir_plot, class_labels):

    probs_ave = df[[ 'P_'+clas for clas in class_labels] ]
    probs_std = df[[ 'e_P_'+clas for clas in class_labels]]
    sources   = df['name'].values
    sources_plot = [str(df.index[i]+1)+'. '+sources[i] for i in range(len(sources))]
    preds = df['Class']

    fig = plot_classifier_matrix_withSTD(np.array(probs_ave), np.array(probs_std), preds, yaxis=np.array(sources_plot) #np.arange(field_probs.shape[0])
                    , classes=class_labels, normalize=True,title=field_name, nocmap=True,cmap=plt.get_cmap('YlOrRd'))

    plt.savefig(f'{dir_plot}/{field_name}.png', bbox_inches='tight')
    plt.close(fig)

    df_conf = df[df.conf_flag>0]

    if len(df_conf)>0:

        probs_ave = df_conf[[ 'P_'+clas for clas in class_labels] ]
        probs_std = df_conf[[ 'e_P_'+clas for clas in class_labels]]
        sources   = df_conf['name'].values
        sources_plot = [str(df_conf.index[i]+1)+'. '+sources[i] for i in range(len(sources))]
        preds = df_conf['Class']

        fig = plot_classifier_matrix_withSTD(np.array(probs_ave), np.array(probs_std), preds, yaxis=np.array(sources_plot) #np.arange(field_probs.shape[0])
                        , classes=class_labels, normalize=True,title=field_name, nocmap=True,cmap=plt.get_cmap('YlOrRd'))

        plt.savefig(f'{dir_plot}/{field_name}_conf.png', bbox_inches='tight')
        plt.close(fig)

  
def prepare_evts_plot_xray_class(field_name, ra_field, dec_field, radius, data_dir, dir_out):

    dir_plot = dir_out+'/plot'
    evt2_dir = dir_plot+'/evt2'
    Path(evt2_dir).mkdir(parents=True, exist_ok=True)
    obsids_info = {}
    obj_info = {}
    merge_script = []


    df_per = pd.read_csv(f'{data_dir}/{field_name}_per.csv')
    obsids = find_obs(df_per,ra_field,dec_field)#.astype(str)


    os.system('rm -rf to_merge.sh')
    if len(obsids) == 1:
        url, fn = get_evt2_file(obsids[0], path=evt2_dir) 
        obj_info[field_name] = fn
    else:
        merged_fn = f'{field_name}_merged_evt.fits'
        obj_info[field_name] = evt2_dir + '/' + merged_fn

        if merged_fn not in os.listdir(evt2_dir):             

            merge_script += [
                f"download_chandra_obsid {','.join(str(obs) for obs in obsids)}",
                'punlearn merge_obs',
                f"merge_obs {'/,'.join(str(obs) for obs in obsids)}/ tmp clobber=yes",
                f'mv tmp_merged_evt.fits ../{merged_fn}',
                'rm tmp*' 
            ]

    if len(merge_script):
        merge_script = [f'mkdir -p {evt2_dir}/merged', f'cd {evt2_dir}/merged'] + merge_script                
        open('to_merge.sh', 'w').write('\n'.join(merge_script)) 
        print("run 'bash to_merge.sh' to get and merge obsids")
 
    os.system('bash to_merge.sh')

    wolfram_colors = [[0., 0.52, 1.], [1., 0.72, 0.], [1., 0., 0.], [0., 0.7, 0.7], 
                  [0.51, 0.5, 1.], [0.784, 0.8, 0.], [0.8, 0., 0.48], 
                  [0.9, 0.6858, 0.27], [0., 0.34, 1.], [1., 0.48, 0.]]

    #sns.palplot(wolfram_colors)

    colors_markers = {
        'HM-STAR': [wolfram_colors[2], '*'], 
        'AGN': [wolfram_colors[1], 'o'],         
        'YSO': ['lightgreen', 'p'], 
        'LMXB': [wolfram_colors[3], 's'],         
        'CV': [wolfram_colors[4], 'P'],         
        'HMXB': [wolfram_colors[5], 'D'], 
        'LM-STAR': [wolfram_colors[6], '^'],
        #'ATNF_BIN': [wolfram_colors[0], 'x'], 
        #'ATNF': [wolfram_colors[8], 's'],
        #'WR': [wolfram_colors[9], '^'],
        'NS': [wolfram_colors[9], 'v']
    }

    handles = [mlines.Line2D([], [], 
                            color=color, 
                            marker=marker, 
                            linestyle='None', 
                            markerfacecolor='None',
                            markeredgewidth=2, 
                            markersize=20, 
                            label=label) 
            for label, (color, marker) in colors_markers.items()]

    '''
    fig, ax = plt.subplots()

    legend = plt.legend(handles=handles, loc='lower left', fontsize=25, facecolor='k')

    for text, v in zip(legend.get_texts(), colors_markers.values()):
        text.set_color(v[0])

    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    plt.show()
    '''

    for conf in ['', '_conf']:

        obj = field_name

        fn_evt2 = obj_info[obj]
        # fits.info(fn_evt2)
        evt2_data, head = process_fits(fn_evt2)

        evt2_data = xy_filter_evt2(evt2_data)
        
        #print(evt2_data)

        dat_csv = pd.read_csv(f'{dir_out}/{field_name}_class.csv')
        if conf == '_conf':
            dat_csv = dat_csv[dat_csv.conf_flag==1]#.reset_index(drop=True)
        
        x_min, x_max, y_min, y_max = evt2_data[0].min(), evt2_data[0].max(), evt2_data[1].min(), evt2_data[1].max()    
        w, h = x_max - x_min, y_max - y_min     
        cntr = [(x_max + x_min)/2, (y_max + y_min)/2]
        reg_phys = get_reg_phys(fn_evt2, dat_csv)

        #     sig = 25

        rmin = 5
        rmax = 20

        c_mx, c_mn = dat_csv['Class_prob'].max(), dat_csv['Class_prob'].min()
        if c_mx == c_mn: 
            c_mx = c_mn + 1

        s_mx, s_mn = dat_csv['significance'].max(), dat_csv['significance'].min()
        if s_mx == s_mn:
            s_mx = s_mn + 1     

        NBINS = (500, int(500 * h / w))

        # my_cmap = copy.copy(plt.cm.viridis)
        # my_cmap.set_under(1, 'k')

        fig, ax = plt.subplots(figsize=(15, 15 * h / w))
        
        #     GREEN CIRCLE
        icrs = SkyCoord(ra=ra_field*u.deg, dec=dec_field*u.deg)
        l, b = icrs.galactic.l.deg, icrs.galactic.b.deg
        #print(icrs,l,b)
        rad = radius
        #rad = np.max([summary[obj]['Size'], summary[obj]['Unc']])
        x, y, rad2 = wcs_to_physical(fn_evt2, [l, b], rad, 'galactic')  
        #print(x,y,rad2)
        
        #ax.hist2d(evt2_data[0], evt2_data[1], NBINS, cmap='viridis', range=[[x-rad2*2,x+rad2*2],[y-rad2*2,y+rad2*2]],norm=LogNorm())
        ax.hist2d(evt2_data[0], evt2_data[1], NBINS, cmap='viridis', norm=LogNorm())
        
        ax.set_facecolor('k')

        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 

        # for point in reg_phys:
        #     plt.plot(*point, markersize=10, c='y', zorder=1, marker='o', markerfacecolor='None')

        for (i, cat), j in zip(dat_csv.iterrows(), range(len(dat_csv))): 

            ms = ((cat['significance'] - s_mn)/(s_mx - s_mn) * (rmax - rmin) + rmin )# * 3
        #         ms = (0.5 * (rmax - rmin) + rmin) * 2.2
            width = 1* (cat['Class_prob'] - c_mn) / (c_mx - c_mn) + 1
            color = colors_markers[cat['Class']][0]
            marker = colors_markers[cat['Class']][1]
            plt.plot(*reg_phys[j], 
                    marker, 
                    ms=ms, 
                    markerfacecolor='None',
                    markeredgecolor=color, 
                    markeredgewidth=width, 
                    label=cat['Class'], 
                    zorder=1)
            plt.text(reg_phys[j][0]+0.7*ms, 
                    reg_phys[j][1]+1.1*ms, 
                    f'{i+1}', 
                    fontsize=10, 
                    color=color)

        labels = list(np.sort(np.unique(dat_csv['Class'])))

        handles = [mlines.Line2D([], [], 
                                color=colors_markers[label][0], 
                                marker=colors_markers[label][1], 
                                linestyle='None', 
                                markerfacecolor='None',
                                markeredgewidth=2, 
                                markersize=20, 
                                label=label) 
                for label in labels]

        legend = plt.legend(handles=handles, loc='lower left', fontsize=25)

        for text, label in zip(legend.get_texts(), labels):
            text.set_color(colors_markers[label][0])

        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0, 0, 1, 0.1))

        
        draw_circle = plt.Circle((x, y), radius=rad2, color='g', zorder=1, lw=4, fill=False)    
        ax.add_artist(draw_circle)

        plt.savefig(f'{dir_plot}/{field_name}{conf}.jpeg', bbox_inches='tight', pad_inches=0) 
        plt.close(fig)
        #plt.show()

        # d.set('exit')  
    return evt2_data,  fn_evt2

import holoviews as hv
from bokeh.plotting import figure, show
import hvplot.pandas
import scipy.ndimage
import scipy as sp
hv.extension('bokeh')

def hook(plot,element):
    # plot.handles['plot'].min_border = 100
    plot.handles['plot'].min_border_top = 200
    plot.handles['plot'].min_border_bottom = 200
    plot.handles['plot'].min_border_left = 200
    plot.handles['plot'].min_border_right = 200
    plot.handles['plot'].outline_line_width  = 1
    plot.handles['plot'].outline_line_alpha = 1
    plot.handles['plot'].outline_line_color = "black"
    plot.handles['xaxis'].axis_label_text_font = 'times'
    plot.handles['xaxis'].axis_label_text_font_style = "normal"
    plot.handles['yaxis'].axis_label_text_font = 'times'
    plot.handles['yaxis'].axis_label_text_font_style = "normal"

def interactive_Ximg_class(field_name, evt2_data, fn_evt2, dir_out):

    x_min, x_max, y_min, y_max = evt2_data[0].min(), evt2_data[0].max(), evt2_data[1].min(), evt2_data[1].max()    
    w, h = x_max - x_min, y_max - y_min     
    cntr = [(x_max + x_min)/2, (y_max + y_min)/2]
    NBINS = (4000, int(4000 * h / w))



    H, xe, ye = np.histogram2d(evt2_data[0], evt2_data[1],bins=NBINS)

    sigma_y = 2.0
    sigma_x = 2.0



    # Apply gaussian filter
    sigma = [sigma_y, sigma_x]
    H = sp.ndimage.filters.gaussian_filter(H, sigma, mode='constant')

    # produce an image of the 2d histogram
    cxo_obs = hv.Image(np.flip(H.T, axis=0), bounds=(x_min, y_min,x_max, y_max)).opts(logz=True, cmap='viridis', clim=(0.01,H.max()), width=600, height=600)

    dat_csv = pd.read_csv(f'{dir_out}/{field_name}_class.csv')
    #dat_csv = dat_csv[dat_csv.conf_flag>0].reset_index(drop=True)
    dat_csv[['reg_phys_x', 'reg_phys_y']] = get_reg_phys(fn_evt2, dat_csv)

    rmin = 5
    rmax = 20

    c_mx, c_mn = dat_csv['Class_prob'].max(), dat_csv['Class_prob'].min()
    if c_mx == c_mn: 
        c_mx = c_mn + 1

    s_mx, s_mn = dat_csv['significance'].max(), dat_csv['significance'].min()
    if s_mx == s_mn:
        s_mx = s_mn + 1  
    dat_csv['ms'] = ((dat_csv['significance'] - s_mn)/(s_mx - s_mn) * (rmax - rmin) + rmin )*20
    dat_csv['wd'] = 1* (dat_csv['Class_prob'] - c_mn) / (c_mx - c_mn) + 1
    #dat_csv['wd'] = dat_csv['Class_prob']*100
    dat_conf = dat_csv[dat_csv.conf_flag>0].reset_index(drop=True)


    markers=hv.dim("Class").categorize({'AGN': 'circle', 'NS': 'inverted_triangle', 'CV': 'hex', 'LM-STAR': 'star', 'HM-STAR': 'star', 'LMXB': 'circle_dot', 'YSO': 'diamond', 'Unconfident Classification': 'triangle'}, default="circle")

    class_scatter = dat_conf.hvplot.scatter('reg_phys_x', 'reg_phys_y', color="Class", marker=markers, size="ms",line_width='wd',hover_cols=['name', 'Class_prob','significance','wd'],#'wd','ms']
        ).opts(
        cmap={'AGN': 'blueviolet', 'NS': 'gold', 'CV': 'blue', 'LM-STAR': 'crimson', 'HM-STAR': 'deepskyblue', 'LMXB': 'black', 'YSO': 'lime', 'Unconfident Classification': 'gray'},
        #size=hv.dim("Class").categorize({'Unconfident Classification': 20}, default=10),
        #line_width=hv.dim("wd"),
        alpha=0.,
        line_alpha=1
        #muted_alpha=1
        )
    '''
    class_scatter2 = dat_conf.hvplot.scatter('reg_phys_x', 'reg_phys_y', color="Class", marker=markers, hover_cols=['name', 'Class_prob']
        ).opts(
        cmap={'AGN': 'blueviolet', 'NS': 'gold', 'CV': 'blue', 'LM-STAR': 'crimson', 'HM-STAR': 'deepskyblue', 'LMXB': 'black', 'YSO': 'lime', 'Unconfident Classification': 'gray'},
        size=hv.dim("Class").categorize({'Unconfident Classification': 20}, default=10),
        line_width=3,
        alpha=0.,
        line_alpha=1
        #muted_alpha=1
        )
    '''
    plot = (cxo_obs*class_scatter).opts(
            hooks=[hook],
            #logx=True,
            #logy=True,
            # xlim=(1e30,1e36),
            # ylim=(1e28,1e32),
            xlabel="pixel",
            ylabel="pixel",
            width=int(1000),
            height=int(1000),
            fontscale=1,
            legend_position='top_left',
            fontsize={
                'legend': int(10),
                # 'a': 20,
            },
            title="",
            ) 

    return plot#()