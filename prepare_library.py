 #!/usr/bin/env python
 # coding: utf-8
'''
python libaray for producing the training dataset

@author: Hui Yang huiyang@gwmail.gwu.edu

Created on Tue Oct 12 2021

version 1.0 the minimum version of library for preparation of TD
'''

import numpy as np, pandas as pd, astropy.units as u, pickle
from os import path
from pathlib import Path
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
import os
import scipy.special as sc
from astroquery.vizier import Vizier
import sys
from collections import Counter
import math
from astropy import coordinates
#from math import *
from astropy.io import fits 
from astroML.crossmatch import crossmatch_angular
from astroquery.gaia import Gaia
from uncertainties import unumpy
from astropy.time import Time

exnum = 975318642

def atnf_pos(coord, e_coord, coord_format='hms', out='pos'):
    '''
        calculate the coordinates and their uncertainties from ATNF catalog
    '''
    
    if coord[0] == "-":
        sign = -1.
    else:
        sign = 1.

    colon_count = coord.count(':')

    if out == 'pos':
        if colon_count == 2:
            h,m,s = pd.to_numeric(coord.split(':'))
            deg = ((s/60. + m)/60. + np.abs(h))
        elif colon_count == 1:
            h,m = pd.to_numeric(coord.split(':'))
            deg = (m/60. + np.abs(h))
        elif colon_count == 0:
            deg = np.abs(pd.to_numeric(coord))
        if coord_format == 'hms':
            return float(sign*deg*15.)
        elif coord_format == 'dms':
            return float(sign*deg)

    elif out == 'err':
        if colon_count == 2:
            e_deg = float(e_coord)
        elif colon_count == 1:
                e_deg = float(e_coord)*60.
        elif colon_count == 0:
            e_deg = float(e_coord)*60.**2
        if coord_format == 'hms':
                return float(e_deg)*15.
        elif coord_format == 'dms':
            return float(e_deg)

def create_perobs_data(data, query_dir, data_dir,  name_type='CSCview', name_col='name', ra_col='ra',dec_col='dec',coord_format='hms'):
    '''
    description:
        extract the per-observation CSC 2.0 data using ADQL from http://cda.cfa.harvard.edu/csccli/getProperties URL  

    input:
        data: the DataFrame containing the master-level information of CSC 2.0 sources, including names, coordinates
        query_dir: the directory to store the per-obs data
        data_dir:  
        name_type: 'CSCview' has prefix 'CXO' while 'VizierCSC' does not for their CSC 2.0 names
        name_col & ra_col & dec_col: column name of CSC 2.0 names, right ascension and declination
        coord_format: hms or deg format for the coordinates

    output:
        no output
        each individual per-obs data is saved as a txt file in query_dir 
        the combined per-obs data is saved as a csv file

    '''

    Path(query_dir).mkdir(parents=True, exist_ok=True)
    
    data['_q'] = data.index + 1
    if coord_format == 'hms':
        ras = Angle(data[ra_col], 'hourangle').degree
        decs = Angle(data[dec_col], 'deg').degree
    elif coord_format =='deg':
        ras = data[ra_col].values
        decs = data[dec_col].values
    
    pu = 0.1
    for source, ra, dec, usrid in zip(data[name_col], ras, decs, range(len(ras))):
        #print(source, ra, dec)
        if name_type == 'CSCview':
            src = source[5:].strip()
        elif name_type == 'VizierCSC':
            src = source[2:-1]#.decode('utf-8') 
        
        if not (path.exists(f'{query_dir}/{src}.txt')):
            print(src)
        
            ra_low  = ra - pu/3600
            ra_upp  = ra + pu/3600
            dec_low = dec - pu/3600
            dec_upp = dec + pu/3600
            rad_cone = pu/60
            
            f = open(f'{query_dir}/csc_query_cnt_template.adql', "r")
            adql = f.readline()
            ra_temp = '266.599396'
            dec_temp = '-28.87594'
            ra_low_temp = '266.5898794490786'
            ra_upp_temp = '266.60891255092145'
            dec_low_temp = '-28.884273333333333'
            dec_upp_temp = '-28.867606666666667'
            rad_cone_temp = '0.543215'
            
            for [str1, str2] in [[rad_cone, rad_cone_temp], [ra, ra_temp], [dec, dec_temp], [ra_low, ra_low_temp], [ra_upp, ra_upp_temp], [dec_low, dec_low_temp], [dec_upp, dec_upp_temp]]:
                adql = adql.replace(str2, str(str1))
            
            text_file = open(f'{query_dir}/{src}.adql', "w")
            text_file.write(adql)
            text_file.close()

            os.system("curl -o "+query_dir+'/'+src+".txt \
                --form query=@"+query_dir+'/'+src+".adql \
                http://cda.cfa.harvard.edu/csccli/getProperties")
    
    df_pers = pd.DataFrame()
    for source, usrid in zip(data[name_col], range(len(ras))):
        #print(usrid,source)
        if name_type == 'CSCview':
            src = source[5:].strip()
        elif name_type == 'VizierCSC':
            src = source[2:-1]#.decode('utf-8') 
        
        df = pd.read_csv(f'{query_dir}/{src}.txt', header=154, sep='\t')
        df['usrid'] = usrid+1
        df_pers = df_pers.append(df, ignore_index=True)
    #'''
    #df_pers.to_csv(query_dir+'../'+field_name+'_per.csv', index=False)

    return df_pers 

def stats(df, flx='flux_aper90_mean_', end='.1', drop=False):
    print("Run stats......")
    df = df.fillna(exnum)
    s0 = np.where( (df[flx+'h'+end]!=0) & (df[flx+'h'+end]!=exnum) & (df[flx+'m'+end]!=0) & (df[flx+'m'+end]!=exnum) & (df[flx+'s'+end]!=0) & (df[flx+'s'+end]!=exnum) )[0]
    s1 = np.where( (df[flx+'h'+end]!=exnum) & (df['e_'+flx+'h'+end]!=exnum)  & (df[flx+'m'+end]!=exnum) & (df['e_'+flx+'m'+end]!=exnum)  & (df[flx+'s'+end]!=exnum) & (df['e_'+flx+'s'+end]!=exnum) )[0]
    s2 = np.where(((df[flx+'h'+end]!=exnum) & (df['e_'+flx+'h'+end]!=exnum) )&((df[flx+'m'+end]!=exnum) & (df['e_'+flx+'m'+end]!=exnum) )&((df[flx+'s'+end]==exnum) | (df['e_'+flx+'s'+end]==exnum) ))[0]
    s3 = np.where(((df[flx+'h'+end]==exnum) | (df['e_'+flx+'h'+end]==exnum) )&((df[flx+'m'+end]!=exnum) & (df['e_'+flx+'m'+end]!=exnum) )&((df[flx+'s'+end]!=exnum) & (df['e_'+flx+'s'+end]!=exnum) ))[0]
    s4 = np.where(((df[flx+'h'+end]!=exnum) & (df['e_'+flx+'h'+end]!=exnum) )&((df[flx+'m'+end]==exnum) | (df['e_'+flx+'m'+end]==exnum) )&((df[flx+'s'+end]!=exnum) & (df['e_'+flx+'s'+end]!=exnum) ))[0]
    s5 = np.where(((df[flx+'h'+end]==exnum) | (df['e_'+flx+'h'+end]==exnum) )&((df[flx+'m'+end]!=exnum) & (df['e_'+flx+'m'+end]!=exnum) )&((df[flx+'s'+end]==exnum) | (df['e_'+flx+'s'+end]==exnum) ))[0]
    s6 = np.where(((df[flx+'h'+end]!=exnum) & (df['e_'+flx+'h'+end]!=exnum) )&((df[flx+'m'+end]==exnum) | (df['e_'+flx+'m'+end]==exnum) )&((df[flx+'s'+end]==exnum) | (df['e_'+flx+'s'+end]==exnum) ))[0]
    s7 = np.where(((df[flx+'h'+end]==exnum) | (df['e_'+flx+'h'+end]==exnum) )&((df[flx+'m'+end]==exnum) | (df['e_'+flx+'m'+end]==exnum) )&((df[flx+'s'+end]!=exnum) & (df['e_'+flx+'s'+end]!=exnum) ))[0]
    s8 = np.where(((df[flx+'h'+end]==exnum) | (df['e_'+flx+'h'+end]==exnum) )&((df[flx+'m'+end]==exnum) | (df['e_'+flx+'m'+end]==exnum) )&((df[flx+'s'+end]==exnum) | (df['e_'+flx+'s'+end]==exnum)))[0]
    s9 = np.where( (df[flx+'h'+end]==exnum) | (df['e_'+flx+'h'+end]==exnum) | (df[flx+'m'+end]==exnum) | (df['e_'+flx+'m'+end]==exnum) | (df[flx+'s'+end]==exnum) | (df['e_'+flx+'s'+end]==exnum) )[0]
    df = df.replace(exnum, np.nan)

    tot = len(df)
    df_rows = [('Y','Y','Y',len(s1),int(len(s1)/tot*1000)/10.),
                 ('Y','Y','N',len(s2),int(len(s2)/tot*1000)/10.),
                 ('N','Y','Y',len(s3),int(len(s3)/tot*1000)/10.),
                 ('Y','N','Y',len(s4),int(len(s4)/tot*1000)/10.),
                 ('N','Y','N',len(s5),int(len(s5)/tot*1000)/10.),
                 ('Y','N','N',len(s6),int(len(s6)/tot*1000)/10.),
                 ('N','N','Y',len(s7),int(len(s7)/tot*1000)/10.),
                 ('N','N','N',len(s8),int(len(s8)/tot*1000)/10.),
                 ('~Y','Y','Y',len(s9),int(len(s9)/tot*1000)/10.)]

    tt = Table(rows=df_rows, names=('H', 'M', 'S','#','%'))
    print(tt)
    print('-----------------')
    print('total:     ',tot)
     
    print("Only ", len(s1), " detections have valid fluxes at all bands.")
     
    if drop==True:
         #print("Only ", len(s1), " detections have valid fluxes at all bands.")
         df.loc[s9, 'per_remove_code'] = df.loc[s9, 'per_remove_code']+64
         print('After dropping', str(len(s9)),'detections with NaNs,', len(df[df['per_remove_code']==0]),'remain.')
         return df

    elif drop == False:
        return df

def flux2symmetric(df, flx='flux_aper90_',bands=['s', 'm','h'],end='.1'):
    # calculate the left & right uncertainties, the mean, the variance of the Fechner distribution for band fluxes
    print("Run flux2symmetric......")
    
    for band in bands:
        df['e_'+flx+'hilim_'+band+end] = df[flx+'hilim_'+band+end] - df[flx+''+band+end]
        df['e_'+flx+'lolim_'+band+end] = df[flx+''+band+end] - df[flx+'lolim_'+band+end]
        df[flx+'mean_'+band+end] = df[flx+''+band+end] + np.sqrt(2/np.pi) * (df['e_'+flx+'hilim_'+band+end] - df['e_'+flx+'lolim_'+band+end])
        df['e_'+flx+'mean_'+band+end] = np.sqrt((1.- 2./np.pi)* (df['e_'+flx+'hilim_'+band+end] - df['e_'+flx+'lolim_'+band+end])**2 + df['e_'+flx+'hilim_'+band+end]*df['e_'+flx+'lolim_'+band+end])
        df = df.drop(['e_'+flx+'hilim_'+band+end, 'e_'+flx+'lolim_'+band+end], axis=1)
    
    return df

def cal_bflux(df, flx='flux_aper90_',end='.1'):
    # calculate the mean and the variance of the broad band flux

    df[flx+'b'+end] = df[flx+'s'+end]+df[flx+'m'+end]+df[flx+'h'+end]
    df['e_'+flx+'b'+end] = np.sqrt(df['e_'+flx+'s'+end]**2+df['e_'+flx+'m'+end]**2+df['e_'+flx+'h'+end]**2)
    
    return df

def powlaw2symmetric(df, cols=['flux_powlaw','powlaw_gamma','powlaw_nh','powlaw_ampl'],end='.1'):
    # calculate the left & right uncertainties, the mean, the variance of the Fechner distribution for band fluxes
    print("Run powlaw2symmetric......")
    
    for col in cols:
        df['e_'+col+'_hilim'+end] = df[col+'_hilim'+end] - df[col+end]
        df['e_'+col+'_lolim'+end] = df[col+end] - df[col+'_lolim'+end]
        df[col+'_mean'+end] = df[col+end] + np.sqrt(2/np.pi) * (df['e_'+col+'_hilim'+end] - df['e_'+col+'_lolim'+end])
        df['e_'+col+'_mean'+end] = np.sqrt((1.- 2./np.pi)* (df['e_'+col+'_hilim'+end] - df['e_'+col+'_lolim'+end])**2 + df['e_'+col+'_hilim'+end]*df['e_'+col+'_lolim'+end])
        df = df.drop(['e_'+col+'_hilim'+end, 'e_'+col+'_lolim'+end], axis=1)
    
    return df

def add_newdata(data, data_dir):
    print("Run add_newdata......")
    
    # Adding new data
    data['obs_reg'] = data['obsid']*10000+data['region_id']
    
    print('Before adding new data:')
    stats(data)
    #stats(data[data['per_remove_code']==0])
    
    bands = ['s', 'm', 'h']
    files = [f'{data_dir}/newdata/output_gwu_snull.txt_May_29_2020_15_32_39_clean.csv', f'{data_dir}/newdata/output_gwu_mnull.txt_Jun_01_2020_09_36_59_clean.csv', f'{data_dir}/newdata/gwu_hnull_output.txt_May_01_2020_13_16_39_clean.csv']
    
    for band, fil in zip(bands, files):
        
        data_new = pd.read_csv(fil)
        
        data_new['obs'] = data_new['#current_obsid'].str[:5].astype(int)
        data_new['obs_reg'] = data_new['obs']*10000+data_new['reg']
        
        data_new = data_new.rename(columns={'mode':'flux_aper90_'+band+'.1', 'lolim':'flux_aper90_lolim_'+band+'.1','hilim':'flux_aper90_hilim_'+band+'.1'})
        data_new = flux2symmetric(data_new, flx='flux_aper90_',bands=[band],end='.1')
        data_new = data_new.set_index('obs_reg')

        data     = data.set_index('obs_reg')
        #df = data.copy()
        
        data.update(data_new)
        
        #print(np.count_nonzero(data!=df)/3)
        
        data.reset_index(inplace=True)
        
        print('After adding new ',str(len(data_new)), band, 'band data:')
        data = stats(data)
        #stats(data[data['per_remove_code']==0])
    
    return data

def apply_flags_filter(data, instrument=True,sig=False,theta_flag=True, dup=True, sat_flag=True, pileup_warning=True, streak_flag=True,pu_signa_fil=False,verb=False):
    print("Run apply_flags_filter......")
    
    data= data.fillna(exnum)
    
    if verb:
        stats(data[data['per_remove_code']==0])

    if instrument:
        s = np.where(data['instrument']==' HRC')[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+1
        print('After dropping', str(len(s)),'detections with HRC instrument,', len(data[data['per_remove_code']==0]),'remain.')        

    if theta_flag:
        s = np.where(data['theta']> 10)[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+2
        print('After dropping', str(len(s)),'detections with theta larger than 10\',', len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code']==0])
    
    if sat_flag:
        s = np.where(data['sat_src_flag.1'] == True)[0]
        #print(str(sorted(Counter(data['Class'].iloc[s]).items())))
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+4
        print("After dropping", len(s), " detections with sat_src_flag = TRUE,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code']==0])

    if pileup_warning:
        s = np.where((data['pileup_warning'] > 0.3) & (data['pileup_warning'] != exnum))[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+8
        print("After dropping", len(s), " detections with pile_warning>0.3,", len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code']==0])

    if streak_flag:
        s = np.where(data['streak_src_flag.1'] == True)[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+16
        print("After dropping", len(s), " detections with streak_src_flag = TRUE,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code']==0])
    
    if dup:
        #print(data.groupby(['obsid','region_id','obi']).filter(lambda g: len(g['name'].unique()) > 1) )
        s = np.where(data.set_index(['obsid','region_id','obi']).index.isin(data.groupby(['obsid','region_id','obi']).filter(lambda g:  len(g['name'].unique()) > 1 ).set_index(['obsid','region_id','obi']).index))[0]
        #data.iloc[s].to_csv('dup.csv',index=False)
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+32
        print("After dropping", len(s), " detections assigned to different sources,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code']==0])

    if pu_signa_fil:
        #s = np.where( (data['flux_significance_b']==exnum) | (data['flux_significance_b']==0)  | (np.isinf(data['PU'])) | (data['PU']==exnum))[0]
        s = np.where( (np.isinf(data['PU'])) | (data['PU']==exnum))[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+64
        print("After dropping", len(s), " detections having nan sig or inf PU,",  len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code']==0])
    
    if sig:
        s = np.where(data['flux_significance_b']< sig)[0]
        data.loc[s, 'per_remove_code'] = data.loc[s, 'per_remove_code']+1
        print('After dropping', str(len(s)),'detections with flux_significance_b less than', sig, len(data[data['per_remove_code']==0]),'remain.')
        if verb:
            stats(data[data['per_remove_code']==0])

    data = data.replace(exnum, np.nan)

    return data

def cal_theta_counts(df, df_ave, theta, net_count, err_count):
    print("Run cal_theta_counts......")

    for col in [theta+'_mean', theta+'_median', 'e_'+theta, net_count, err_count]:
        df_ave[col] = np.nan
    
    for src in df.usrid.unique():
        idx = np.where( (df.usrid == src))[0] # & (df.per_remove_code==0) )[0]
        idx2 = np.where(df_ave.usrid == src)[0]
        
 
        theta_mean = np.nanmean(df.loc[idx, theta].values)
        theta_median = np.nanmedian(df.loc[idx, theta].values)
        theta_std = np.nanstd(df.loc[idx, theta].values)
        counts = np.nansum(df.loc[idx, net_count].values)
        e_counts = np.sqrt(np.nansum( [ e**2 for e in df.loc[idx, err_count].values]   ))
        
        df_ave.loc[idx2, theta+'_mean']  = theta_mean
        df_ave.loc[idx2, theta+'_median'] = theta_median
        df_ave.loc[idx2, 'e_'+theta]  = theta_std
        df_ave.loc[idx2, net_count] = counts
        df_ave.loc[idx2, err_count]  = e_counts
        df_ave.loc[idx2, theta+'_median'] = theta_median

    return df, df_ave


def cal_sig(df, df_ave, sig):
    print("Run cal_sig......")

    df_ave[sig+'_max'] = np.nan
    
    for src in df.usrid.unique():
        idx = np.where( (df.usrid == src) & (df.per_remove_code==0) )[0]
        idx2 = np.where(df_ave.usrid == src)[0]

        if len(idx)==0:
            continue
        elif len(idx)==1:
            sig_max = df.loc[idx,sig].values
        else:
            sig_max =  np.nanmax(df.loc[idx,sig]) 

        df_ave.loc[idx2, sig+'_max'] = sig_max
    
    return df, df_ave

def cal_cnt(df, df_ave, cnt, cnt_hi, cnt_lo):
    df_ave[cnt+'_max'] = np.nan
    
    for src in df.usrid.unique():
        idx = np.where( (df.usrid == src) & (df.per_remove_code==0) )[0]
        idx2 = np.where(df_ave.usrid == src)[0]

        if len(idx)==0:
            continue
        elif len(idx)==1:
            cnt_max = df.loc[idx,cnt].values
            cnt_max_hi = df.loc[idx,cnt_hi].values
            cnt_max_lo = df.loc[idx,cnt_lo].values
        else:
            max_ind =  np.nanargmax(df.loc[idx,cnt])
            cnt_max = df.loc[max_ind, cnt]
            cnt_max_hi = df.loc[max_idx, cnt_hi] 
            cnt_max_lo = df.loc[max_idx, cnt_lo]
            print(cnt_max, np.nanmax(df.loc[idx,cnt])) 

        df_ave.loc[idx2, cnt+'_max'] = cnt_max
        df_ave.loc[idx2, cnt+'_max_hi'] = cnt_max_hi
        df_ave.loc[idx2, cnt+'_max_lo'] = cnt_max_lo
    
    return df, df_ave



def cal_aveflux(df, df_ave, bands, flux_name, per_flux_name, fil = False, add2df=False):
    print("Run cal_aveflux......")
    
    for band in bands:
        col = flux_name+band
        p_col = per_flux_name+band+'.1'
        df_ave[col]      = np.nan
        df_ave['e_'+col] = np.nan
        if add2df:
            df[col]      = np.nan
            df['e_'+col] = np.nan
        
        for uid in df.usrid.unique():
            
            if fil==True:
                idx = np.where( (~df[p_col].isna()) & (~df['e_'+p_col].isna()) & (df.per_remove_code==0) & (df.usrid==uid) & (df.theta<=10) & (df['sat_src_flag.1']!=True) & (df.pileup_warning<=0.3))[0]
            elif fil=='strict':
                idx = np.where( (~df[p_col].isna()) & (~df['e_'+p_col].isna()) & (df.per_remove_code==0) & (df.usird==uid) & (df.theta<=10) & (df['sat_src_flag.1']==False) & (df.conf_code<=7) & (df.pileup_warning<=0.1) & (df.edge_code<=1) & (df.extent_code<=0) & (df['streak_src_flag.1']==False) )[0]
            else:
                idx = np.where( (~df[p_col].isna()) & (~df['e_'+p_col].isna()) & (df.per_remove_code==0) & (df.usrid==uid) )[0]
            
            idx2 = np.where(df_ave.usrid==uid)[0]
            
            if len(idx)==0:
                #df_ave.loc[idx2, 'remove_code'] = 1
                continue
            
            elif len(idx) ==1:
                ave      = df.loc[idx, p_col].values
                err = df.loc[idx, 'e_'+p_col].values
                #df_ave.loc[idx2, col]      = ave
                #df_ave.loc[idx2, 'e_'+col] = err
            
            else:
                ave = np.average(df.loc[idx, p_col].values, weights=1./(df.loc[idx, 'e_'+p_col].values)**2)
                err =  np.sqrt(1./sum(1./(df.loc[idx, 'e_'+p_col].values)**2))
                
            df_ave.loc[idx2, col]  = ave
            df_ave.loc[idx2, 'e_'+col] = err
            if add2df:
                df.loc[idx, col]      = ave
                df.loc[idx, 'e_'+col] = err

    return df, df_ave

def cal_var(df, df_ave, b_ave,b_per):
    print("Run cal_var......")
    
    new_cols = ['chisqr', 'dof', 'kp_prob_b_max','var_inter_prob','significance_max']
    for col in new_cols:
        df_ave[col] = np.nan
    
    #b_per     = 'flux_aper90_mean_b.1'

    for uid in sorted(df.usrid.unique()):
        idx = np.where( (~df[b_per].isna()) & (~df['e_'+b_per].isna()) & (df.per_remove_code==0) & (df.usrid==uid) )[0]
    
        dof = len(idx)-1.
        
        idx2 = np.where(df_ave.usrid==uid)[0]
        df_ave.loc[idx2, 'dof'] = dof
        
        if dof >-1:
            df_ave.loc[idx2, 'kp_prob_b_max'] = np.nanmax(df.loc[idx,'kp_prob_b'].values)
            df_ave.loc[idx2, 'significance_max'] = np.nanmax(df.loc[idx,'flux_significance_b'].values)
        
        if (dof == 0) or (dof == -1):
            continue
    
        chisqr = np.sum((df.loc[idx,b_per].values-df.loc[idx,b_ave].values)**2/(df.loc[idx,'e_'+b_per].values)**2)
        df_ave.loc[idx2, 'chisqr']= chisqr

    df_ave['var_inter_prob'] = df_ave.apply(lambda row: sc.chdtr(row.dof, row.chisqr), axis=1)
    df_ave = df_ave.round({'var_inter_prob': 3})

    return df, df_ave

def combine_master(df_ave, bands=['s','m','h']):
    print("run combine_master......")
    
    # one revision can be considered is that to change the "remove_code" if some master data replaced are good 

    for band in bands:
        s = np.where((df_ave['flux_aper90_ave_'+band].isna()) & (df_ave['e_flux_aper90_ave_'+band].isna()) & (~df_ave['flux_aper90_mean_'+band].isna()) & (~df_ave['e_flux_aper90_mean_'+band].isna()) & (df_ave['sat_src_flag']!=True) & (df_ave['streak_src_flag']!=True) )[0]#  
        df_ave.loc[s, 'flux_aper90_ave_'+band] = df_ave.loc[s, 'flux_aper90_mean_'+band] 
        df_ave.loc[s, 'e_flux_aper90_ave_'+band] = df_ave.loc[s, 'e_flux_aper90_mean_'+band] 
        print(len(s), "sources at ", band, " band are saved by combing master data.")
        df_ave = stats(df_ave, flx='flux_aper90_ave_', end='')
        stats(df_ave[df_ave['remove_code']==0], flx='flux_aper90_ave_', end='')
    return df_ave

def nan_flux(df_ave, flux_name):
    print("Run nan_flux......")

    df_ave['flux_flag'] = 0

    for band, code, flux_hilim in zip(['s', 'm', 'h'], [1, 2, 4], [1e-17, 1e-17, 1e-17]):
        col = flux_name+band
        idx = np.where( (df_ave[col].isna()) | (df_ave['e_'+col].isna()) )[0]
        
        df_ave.loc[idx, col] = np.sqrt(2/np.pi) * flux_hilim
        df_ave.loc[idx, 'e_'+col] = np.sqrt((1.- 2./np.pi))* flux_hilim
        df_ave.loc[idx, 'flux_flag'] = df_ave.loc[idx, 'flux_flag'] + code

    return df_ave

def cal_ave(df, data_dir, dtype='TD', Chandratype='CSC',PU=False,cnt=False,plot=False, verb=False):
    '''
    description:
        calculate the averaged data from per-observation CSC data

    input: 
        df: the DataFrame of per-observation data
        dtype: 'TD' for training dataset and 'field' for field (testing) dataset
        plot: plot mode

    output: 
        df_ave: the DataFrame of averaged data from per-observation CSC data
        df: the per-observation data used to calculate the averaged data

    '''
    
    print("Run cal_ave......")
    print('There are',str(len(df)),'per-obs data.')

    df = df.fillna(exnum)
    df = df.replace(r'^\s*$', exnum, regex=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.replace({' TRUE': True, 'False': False, 'FALSE':False})
    df = df.replace(exnum, np.nan)
    
    # convert asymmetric fluxes to symmetric fluxes
    df = flux2symmetric(df, end='.1')
    
    if Chandratype == 'CSC' or  Chandratype=='CSC-CXO':
        df = flux2symmetric(df, end='')
        df = cal_bflux(df, flx='flux_aper90_mean_',end='')
        df = powlaw2symmetric(df, end='')
    
    if dtype == 'TD' and Chandratype=='CSC':
        # Adding new data
        df = add_newdata(df, data_dir)
    
    df = cal_bflux(df, flx='flux_aper90_mean_',end='.1')
    
    # Apply with some filters on sat_src_flag and pile_warning at per-obs level
    if Chandratype=='CSC' or  Chandratype=='CSC-CXO':
        df = apply_flags_filter(df, verb=verb)# theta_flag=True,dup=True,sat_flag=True,pileup_warning=True,streak_flag=True
    elif Chandratype=='CXO':
        df = apply_flags_filter(df,theta_flag=True,dup=False,sat_flag=False,pileup_warning=False,streak_flag=False,pu_signa_fil=False,verb=verb)
    #'''
    #df.to_csv('TD_test.csv',index=False)

    if Chandratype=='CSC':
        cols_copy = ['name', 'usrid', 'ra', 'dec', 'err_ellipse_r0', 'err_ellipse_r1', 'err_ellipse_ang', 'significance',
                  'extent_flag', 'pileup_flag','sat_src_flag', 'streak_src_flag','conf_flag',
                  'flux_aper90_mean_b', 'e_flux_aper90_mean_b', 'flux_aper90_mean_h', 'e_flux_aper90_mean_h', 
                  'flux_aper90_mean_m', 'e_flux_aper90_mean_m', 'flux_aper90_mean_s', 'e_flux_aper90_mean_s', 
                  'kp_intra_prob_b','ks_intra_prob_b','var_inter_prob_b',
                  'nh_gal','flux_powlaw_mean','e_flux_powlaw_mean','powlaw_gamma_mean','e_powlaw_gamma_mean',
                  'powlaw_nh_mean','e_powlaw_nh_mean','powlaw_ampl_mean','e_powlaw_ampl_mean','powlaw_stat']
   
    elif Chandratype=='CXO':
          cols_copy = ['usrid']
    elif Chandratype=='CSC-CXO':
        cols_copy = ['COMPONENT','name', 'usrid', 'ra', 'dec', 'err_ellipse_r0', 'err_ellipse_r1', 'err_ellipse_ang', 'significance',
                  'extent_flag', 'pileup_flag','sat_src_flag', 'streak_src_flag','conf_flag',
                  'flux_aper90_mean_b', 'e_flux_aper90_mean_b', 'flux_aper90_mean_h', 'e_flux_aper90_mean_h', 
                  'flux_aper90_mean_m', 'e_flux_aper90_mean_m', 'flux_aper90_mean_s', 'e_flux_aper90_mean_s', 
                  'kp_intra_prob_b','ks_intra_prob_b','var_inter_prob_b',
                  'nh_gal','flux_powlaw_mean','e_flux_powlaw_mean','powlaw_gamma_mean','e_powlaw_gamma_mean',
                  'powlaw_nh_mean','e_powlaw_nh_mean','powlaw_ampl_mean','e_powlaw_ampl_mean','powlaw_stat']
   
    df = df[df['per_remove_code']==0].reset_index(drop=True)
    #df.to_csv('TD_test.csv',index=False)
    if PU:
        df_ave = df[cols_copy+[PU]].copy()
    else:
        df_ave = df[cols_copy].copy()
    
    df_ave['prod_per_remove_code'] = 0
    for uid in df_ave.usrid.unique():
        idx = np.where(df.usrid==uid)[0]
        idx2 = np.where(df_ave.usrid==uid)[0]
        df_ave.loc[idx2, 'prod_per_remove_code'] = np.prod(df.loc[idx, 'per_remove_code'].values)

    df_ave = df_ave.drop_duplicates(subset =['usrid'], keep = 'first')
    df_ave = df_ave.reset_index(drop=True)
    df_ave['remove_code'] = 0
    df_ave.loc[df_ave.prod_per_remove_code>0, 'remove_code'] = 1
    df_ave = df_ave.drop('prod_per_remove_code', axis=1)
   
    df, df_ave = cal_sig(df, df_ave, 'flux_significance_b')
    if Chandratype=='CXO' or Chandratype=='CSC-CXO':
        df, df_ave = cal_theta_counts(df, df_ave, 'theta', 'NET_COUNTS_broad','NET_ERR_broad')

    # Calculating average fluxes
    df, df_ave = cal_aveflux(df, df_ave,['s','m','h'],'flux_aper90_ave_','flux_aper90_mean_')#fil =False)
    df, df_ave = cal_aveflux(df, df_ave,['b'],'flux_aper90_ave2_','flux_aper90_mean_',add2df=True)

    # Calculating inter-variability
    df, df_ave = cal_var(df, df_ave, 'flux_aper90_ave2_b','flux_aper90_mean_b.1')
    df_ave = df_ave.drop(['flux_aper90_ave2_b','e_flux_aper90_ave2_b'],axis=1)

    if Chandratype=='CSC' or  Chandratype=='CSC-CXO':
        # combine additional useful master flux
        #df_ave = combine_master(df_ave)
        df_ave['ra']= Angle(df_ave['ra'], 'hourangle').degree
        df_ave['dec'] = Angle(df_ave['dec'], 'deg').degree
    
    df_ave = nan_flux(df_ave, 'flux_aper90_ave_')
    
    df_ave = cal_bflux(df_ave, 'flux_aper90_ave_',end='')

    if cnt:
        df, df_ave = cal_cnt(df, df_ave, 'src_cnts_aper90_b','src_cnts_aper90_hilim_b','src_cnts_aper90_lolim_b')
    #df_ave.to_csv('ave_test.csv',index=False)
    #'''
    return df_ave, df

def MW_counterpart_confusion(ras, decs, R, Es=[], N=10, catalog='wise',ref_mjd=5.e4, pm_cor=False, confusion=True, second_nearest=False):
    '''
        input:
        
        radec:  [[Ra1, Dec1], [Ra2, Dec2], ..., [Ran, Decn]] or [Ra, Dec] (degrees)
        R: searching radius for estimating density (arcs)
        E: source position uncertainties [E1, E2, ..., En] (arcs, optional)
        N: multiple of the nearest source outside of error circle defined as the outer boundary of the area to estimate field density
        catalog: '2mass' or 'gaiadr2' or 'wise'
        
        output: a DataFrame table containing
        
        d_nr: distance to the nearest source outside error circle
        d_out: outer boundary of the circle to estimate field density
        num: number of sources within the circle to estimate density
        rho: field density
        _q: index of the source
        prob: chance coincidence probability
        prob_log: logarithmic of chance coincidence probability
        
        '''
    
    cats = {
        'gaia':     'I/350/gaiaedr3',
        'gaiadist': 'I/352/gedr3dis',
        '2mass':    'II/246/out',
        'catwise':  'II/365/catwise',
        'unwise':   'II/363/unwise',
        'allwise':  'II/328/allwise',
        'vphas':   'II/341'
    }
    
    cols = {
        'gaia': ['RAJ2000', 'DEJ2000', 'Gmag', 'e_Gmag', 'BPmag', 'e_BPmag', 'RPmag', 'e_RPmag','Plx', 'e_Plx', 'PM', 'pmRA', 'e_pmRA', 'pmDE'],#, 'e_pmDE'],
        'gaiadist': ['RAJ2000', 'DEJ2000', 'rgeo', 'rpgeo'],
        '2mass': ['RAJ2000', 'DEJ2000', 'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag'],
        'catwise': ['RAJ2000', 'DEJ2000','W1mproPM','e_W1mproPM','W2mproPM','e_W2mproPM','pmRA','e_pmRA','pmDE','e_pmDE'],
        'unwise': ['RAJ2000', 'DEJ2000','FW1','e_FW1','FW2','e_FW2'],
        'allwise': ['RAJ2000', 'DEJ2000', 'W1mag', 'e_W1mag',  'W2mag', 'e_W2mag',  'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag']
    }
    # astroquery match 
    # gaia: RA_ICRS DE_ICRS at Ep=2016.0; gaiadist: RA_ICRS DE_ICRS at Ep=2016.0
    # 2mass: RAJ2000 DEJ2000 at Ep=2000.0; catwise: RA_ICRS DE_ICRS at unknown epoch, change to RAPMdeg DEPMdeg at Ep=2015.4 
    # unwise: RAJ2000 DEJ2000 at Ep=2000.0; allwise: RAJ2000 DEJ2000 at Ep=2000.0
    # vphas: RAJ2000 DEJ2000 at Ep=2000.0
    pu_cols = {
            'gaia':'e_DE_ICRS', # in mas
            '2mass':'errMaj', # in arcsec
            'catwise': 'e_DE_ICRS', # in arcsec
            'allwise': 'eeMaj' }# in arcsec 

    
    if catalog not in cats:
        sys.exit(f"wrong catalog name, should be {' or '.join(cats)}")

    viz = Vizier(row_limit=-1,  timeout=5000, columns=["**", "+_r"], catalog=cats[catalog])


    df_rho = pd.DataFrame(columns=['_rs'])#'d_nr','d_out','num','rho','prob','_q','_rs'])
    df_MW = pd.DataFrame(columns=['_q'])#+cols[catalog])
    if len(Es)==1:
        Es = Es*len(ras)
    
        
    radec = [[ras[i], decs[i]] for i in range(len(ras))]
    rd = Table(Angle(radec, 'deg'), names=('_RAJ2000', '_DEJ2000'))
    #print(rd)
    print('cross-matching to',catalog)
    if confusion:
        query_res = viz.query_region(rd, radius=R*u.arcsec)[0]
    else:
        query_radius = 10.
        query = viz.query_region(rd, radius=query_radius*u.arcsec)
        while len(query)==0:
            query_radius +=10
            print('updating query radius to',query_radius)
            query = viz.query_region(rd, radius=query_radius*u.arcsec)
        query_res = query[0]
    
    df = query_res.to_pandas()

    if catalog == 'gaia':
        df['ra_X'] = df.apply(lambda row: ras[row._q-1],axis=1)
        df['dec_X'] = df.apply(lambda row: decs[row._q-1],axis=1)

        print('cross-matching to gaiadist')
        viz = Vizier(row_limit=-1,  timeout=5000, columns=["**", "+_r"], catalog=cats['gaiadist'])
        query_radius = 10.
        query = viz.query_region(rd, radius=query_radius*u.arcsec)
        while len(query)==0:
            query_radius +=10
            print('updating query radius to',query_radius)
            query = viz.query_region(rd, radius=query_radius*u.arcsec)
        query_res = query[0]

        df_gaiadist = query_res.to_pandas()
    
    #'''
    if pm_cor == True and catalog == 'gaia':
        gaia_ref_mjd = 57388.
        delta_yr = (ref_mjd - gaia_ref_mjd)/365.
        df['RA_pmcor'] = df.apply(lambda row:row.RA_ICRS+delta_yr*row.pmRA/(np.cos(row.DE_ICRS*np.pi/180.)*3.6e6),axis=1)
        df['DEC_pmcor'] = df.apply(lambda row:row.DE_ICRS+delta_yr*row.pmDE/3.6e6,axis=1)
        df.loc[df['RA_pmcor'].isnull(),'RA_pmcor'] = df['RA_ICRS']
        df.loc[df['DEC_pmcor'].isnull(),'DEC_pmcor'] = df['DE_ICRS']
        df.loc[:, '_r_nopmcor'] = df['_r']
        df['_r'] = df.apply(lambda row:  SkyCoord(row.RA_pmcor*u.deg, row.DEC_pmcor*u.deg, frame='icrs').separation(SkyCoord(row.ra_X*u.deg, row.dec_X*u.deg, frame='icrs')).arcsecond,axis=1)

    if pm_cor ==False and catalog == 'gaia':
        df['_r'] = df.apply(lambda row:  SkyCoord(row.RA_ICRS*u.deg, row.DE_ICRS*u.deg, frame='icrs').separation(SkyCoord(row.ra_X*u.deg, row.dec_X*u.deg, frame='icrs')).arcsecond,axis=1)
        

    #if catalog == 'catwise':
        #df['_r'] = df.apply(lambda row:  SkyCoord(row.RAPMdeg*u.deg, row.DEPMdeg*u.deg, frame='icrs').separation(SkyCoord(row.ra_X*u.deg, row.dec_X*u.deg, frame='icrs')).arcsecond,axis=1)

    '''
    if pm_cor == True and catalog == 'catwise':
        catwise_ref_mjd = 57170.
        delta_yr = (ref_mjd - catwise_ref_mjd)/365.
        df['RA_pmcor'] = df.apply(lambda row:row.RAPMdeg+delta_yr*row.pmRA/(np.cos(row.DEPMdeg*np.pi/180.)*3.6e3),axis=1)
        df['DEC_pmcor'] = df.apply(lambda row:row.DEPMdeg+delta_yr*row.pmDE/3.6e3,axis=1)
        df['_r'] = df.apply(lambda row:  SkyCoord(row.RA_pmcor*u.deg, row.DEC_pmcor*u.deg, frame='icrs').separation(SkyCoord(row.ra_X*u.deg, row.dec_X*u.deg, frame='icrs')).arcsecond,axis=1)
        df.loc[:, '_r_nopmcor'] = df['_r']
        df['_r'] = df.apply(lambda row:  SkyCoord(row.RA_pmcor*u.deg, row.DEC_pmcor*u.deg, frame='icrs').separation(SkyCoord(row.ra_X*u.deg, row.dec_X*u.deg, frame='icrs')).arcsecond,axis=1)
    
    if pm_cor == True and catalog == 'allwise':
        allwise_ref_mjd = 55400.
        delta_yr = (ref_mjd - allwise_ref_mjd)/365.
        df['RA_pmcor'] = df.apply(lambda row:row.RA_pm+delta_yr*row.pmRA/(np.cos(row.DE_pm*np.pi/180.)*3.6e6),axis=1)
        df['DEC_pmcor'] = df.apply(lambda row:row.DE_pm+delta_yr*row.pmDE/3.6e6,axis=1)
        df.loc[:, '_r_nopmcor'] = df['_r']
        df['_r'] = df.apply(lambda row:  SkyCoord(row.RA_pmcor*u.deg, row.DEC_pmcor*u.deg, frame='icrs').separation(SkyCoord(row.ra_X*u.deg, row.dec_X*u.deg, frame='icrs')).arcsecond,axis=1)
    '''

    for i, e in zip(range(len(ras)), Es):
        #print(i,e)
        df_sub = df.loc[df['_q']==i+1].reset_index(drop=True)
        df_sub = df_sub.sort_values(by=['_r']).reset_index(drop=True)
        #nr_bk = min(df_sub.loc[df_sub['_r']> e, '_r'])

        if confusion:
        
            try:
                nr_bk  = min(df_sub.loc[df_sub['_r']> e, '_r'])
            except ValueError:
                #df_sub.to_csv('MW_test.csv')
                #print(i,e)
                continue

            bk_out = max(min(R, nr_bk*N), 30)
            #bk_out = min(R, nr_bk*N)
            num_bk = len(df_sub.loc[df_sub['_r']< bk_out])
            rho_bk = num_bk/(np.pi*(bk_out**2))
            
            chance_prob = 1. - np.exp(-rho_bk*(np.pi*e**2))
            
            '''
            
            if catalog in pu_cols:
                cat_pu_median = 2.*np.median(df_sub[pu_cols[catalog]].values)
                if catalog == 'gaia':
                    cat_pu_median = cat_pu_median/1.e3

                df_sub['pu_median'] = cat_pu_median
                e = np.sqrt(e**2+cat_pu_median**2)
            else:
                print(catalog, ' does not count PU in X-ray PUs.')

            '''

                
            rs = df_sub.loc[df_sub['_r']<10.,'_r'].tolist()
            new_row = {'d_nr':nr_bk,'d_out':bk_out,'num':num_bk,'rho':rho_bk,'prob':chance_prob,'_q':i+1,'_rs':rs}
                
        #if len(df_sub.loc[df_sub['_r']<=2.*e]) >0:

        else:
            rs = df_sub.loc[df_sub['_r']<10.,'_r'].tolist()
            new_row = {'_q':i+1,'_rs':rs}
        
        df_rho = df_rho.append(new_row, ignore_index=True)


        if second_nearest:
            try:
                df_cp = df_sub.loc[df_sub['_r']==sorted(df_sub['_r'].values)[1]]
            except IndexError:
                continue
        else:
            try:
                df_cp = df_sub.loc[df_sub['_r']==min(df_sub['_r'].values)]
            except ValueError:
                #print(i,e)         
                continue
        
        if catalog == 'gaia':
            df_cp = df_cp.drop(columns=['ra_X','dec_X'])
        
        #print(df_cp)
        #df_MW.append(df_counter, ignore_index=True)
        df_MW = pd.concat([df_MW, df_cp])

    df_MW = df_MW.drop_duplicates(subset=['_q'])
    
    if confusion:
        df_rho['prob_log'] = np.log10(df_rho['prob'])

    df_MWs = pd.merge(df_rho, df_MW, how='outer', on=['_q', '_q'])
    
    df_MWs = df_MWs.add_suffix('_'+catalog)
    df_MWs = df_MWs.rename(columns= {'_q_'+catalog: '_q'} )

    if catalog == 'gaia':
        df_gaiadist = df_gaiadist.add_suffix('_gaiadist')
        df_gaiadist = df_gaiadist.rename(columns= {'Source_gaiadist':'Source_gaia','_q_gaiadist':'_q'})

        df_MWs = df_MWs[df_MWs['EDR3Name_gaia'].notna()]
        df_MWs['Source_gaia'] = df_MWs.apply(lambda row: np.int64(row.EDR3Name_gaia[10:]), axis=1)
        #print(df_MWs[['Source_gaia','_q']], df_gaiadist[['Source_gaia','_q']])
        df_MWs = pd.merge(df_MWs, df_gaiadist, how="left", on=["Source_gaia","_q"])
    #print(df_MWs)
    #df_MWs = df_MWs.rename(columns= {'RAJ2000':'RAJ2000_'+catalog, 'DEJ2000':'DEJ2000_'+catalog, '_r': '_r_'+catalog} )
    
    return df_MWs

def add_MW(df, file_dir, field_name, Chandratype='CSC',ref_mjd=5.e4,pm_cor=False,confusion=False):

    #data = data[0:3]
    if path.exists(f'{file_dir}/{field_name}_MW.csv') == True:
        data_MW_old = pd.read_csv(f'{file_dir}/{field_name}_MW.csv')
        data = df.loc[~df.name.isin(data_MW_old.name)].reset_index(drop=True)
        idx_old = data_MW_old.index[data_MW_old.name.isin(df.name)]
        data_old = data_MW_old.loc[idx_old].reset_index(drop=True)
        
    else:
        data = df

    #print(len(data)) 
    #print(data)

    if (len(data) == 0) and (len(data_MW_old) > len(idx_old)):
        print(len(data), len(data_MW_old), len(idx_old))
        for cat in ['gaia','2mass','catwise','unwise','allwise']:
            df_MW_old = pd.read_csv(f'{file_dir}/{field_name}_{cat}.csv')
            print(len(df_MW_old))
            df_MW_old = df_MW_old.loc[df_MW_old._q.isin(data_old._q)].reset_index(drop=True)
            #df_MW.to_csv(file_dir+'/'+field_name+'_'+cat+'_new.csv', index=False)
            df_MW_old.to_csv(f'{file_dir}/{field_name}_{cat}.csv', index=False)   
        
        data_old.to_csv(file_dir+'/'+field_name+'_MW.csv', index=False)   

    if len(data) > 0:
        if Chandratype == 'CSC':
            ras = data['ra'].values
            decs = data['dec'].values
            Es = data['err_ellipse_r0'].values
        
        elif Chandratype == 'CXO' or Chandratype=='CSC-CXO':
            ras = data['RA'].values
            decs = data['DEC'].values
            Es = data['PU'].values
        
        data = data.reset_index(drop=True)
        if path.exists(f'{file_dir}/{field_name}_MW.csv') == True:
            data['_q'] = data.index + 1 + len(data_MW_old)
        else:
            data['_q'] = data.index + 1
        search_radius = 300 # arcsec
        sig_nr = 10
    #print(ras,decs)        
#for cat, confusion,second_nearest in zip(['gaia','gaiadist','2mass','catwise','unwise','allwise'], [False]*6,[True]+5*[False]):
        for cat, conf in zip(['gaia','2mass','catwise','unwise','allwise'], [confusion]*5):#,True,True,True,True,True,False]):
            # For field data NGC 3532, we should not match to gaiadist so that we don't have to remove those match to gaiadist but no gaia matched later.
            #print(cat, confusion)
            
            df_MW = MW_counterpart_confusion(ras, decs, search_radius, Es=Es, N=sig_nr, catalog=cat,ref_mjd=ref_mjd,pm_cor=pm_cor,confusion=conf)
            
            if path.exists(f'{file_dir}/{field_name}_{cat}.csv') == False:
                #df_MW = MW_counterpart_confusion(ras, decs, search_radius, Es=Es, N=sig_nr, catalog=cat,ref_mjd=ref_mjd,pm_cor=pm_cor,confusion=confusion)
                df_MW.to_csv(file_dir+'/'+field_name+'_'+cat+'.csv', index=False) 
            else:
                #if len(ras)
                df_MW_old = pd.read_csv(f'{file_dir}/{field_name}_{cat}.csv')
                df_MW_old = df_MW_old.loc[df_MW_old._q.isin(data_old._q)].reset_index(drop=True)
                #df_MW_old = df_MW_old.loc[idx_old].reset_index(drop=True)
                df_MW = df_MW.drop_duplicates(subset=['_q'])
                df_MW['_q'] = df_MW['_q']+len(data_MW_old)
                #df_MW.to_csv(file_dir+'/'+field_name+'_'+cat+'_new.csv', index=False)
                df_MW_all = df_MW_old.append(df_MW, ignore_index=True)
                #df_MW_all['_q'] = df_MW_all.index+1
                df_MW_all.to_csv(f'{file_dir}/{field_name}_{cat}.csv', index=False)   
            
            data = pd.merge(data, df_MW, how='outer', on=['_q', '_q'])
        
        if path.exists(f'{file_dir}/{field_name}_MW.csv') == True:
            #data.to_csv(file_dir+'/'+field_name+'_MW_new.csv', index=False)
            data_all = data_old.append(data,ignore_index=True)
            #data_all['_q'] = data_all.index+1
            data_all.to_csv(f'{file_dir}/{field_name}_MW.csv', index=False)   
        else:
            data.to_csv(f'{file_dir}/{field_name}_MW.csv', index=False)


def counterpart_clean(df, X_PU='PU',catalog='gaia',X_mjd=57388.,pu_factor=1.5,pm_cor=True,r2=False):
    '''
    description
        drop counterparts outside pu_factor * X-ray PUs in 95% confidence level where X-ray PUs are taking into account of MU PUs, proper motion uncertainties, parallaxes (and their uncertainties), astrometric noises, if any of those are available; flag suspicious counterparts if they are outside 95% PUs or other counterparts are also near the X-ray sources 
    '''
    # astroquery match 
    # gaia: RA_ICRS DE_ICRS at Ep=2016.0 (MJD 57388); gaiadist: RA_ICRS DE_ICRS at Ep=2016.0
    # 2mass: RAJ2000 DEJ2000 between 1997 June (MJD 50600) and 2001 Feb 15 (MJD 51955); catwise: RA_ICRS DE_ICRS at unknown epoch, change to RAPMdeg DEPMdeg at Ep=2015.4 (MJD 57170) 
    # unwise: RAJ2000 DEJ2000 between 2009 Dec (MJD 55166) and 2011 Feb (MJD 55593), 2013 Dec (MJD 56627) and 2017 Dec (58088); allwise: RAJ2000 DEJ2000 between 2010 Jan 7 (MJD 55203) and 2010 Aug 6 (MJD 55414), and 2010 Sep 29 (MJD 55468) and 2011 Feb 1 (MJD 55593)
    # vphas: RAJ2000 DEJ2000 between 2011 Dec 28 (MJD 55923) and 2013 Sep (MJD 56536). 

    mjd_difs = {'gaia':X_mjd-57388.,'gaiadist':X_mjd-57388.,'2mass':max(abs(X_mjd-50600),(X_mjd-51955)),'catwise':X_mjd-57170.0,
              'unwise':max(abs(X_mjd-55203.),abs(X_mjd-55593.),abs(X_mjd-56627),abs(X_mjd-58088)),
              'allwise':max(abs(X_mjd-55203.),abs(X_mjd-55414.),abs(X_mjd-55468),abs(X_mjd-55593)),
              'vphas':max(abs(X_mjd-55923),abs(X_mjd-56536))
            }
    
    df['PU_'+catalog] = 0.
    
    # CU: Coordinate uncertainty 
    if catalog == 'gaia':
        df['PU_CU_'+catalog] = df.apply(lambda row: max(row.e_RA_ICRS_gaia, row.e_DE_ICRS_gaia)*2./1.e3, axis=1)
    elif catalog == 'gaiadist':
        df['PU_CU_'+catalog] = df['PU_CU_gaia']
    elif catalog == '2mass':
        df['PU_CU_'+catalog] = df['errMaj_2mass']*2.
    elif catalog == 'catwise':
        df['PU_CU_'+catalog] = df.apply(lambda row: max(row.e_RA_ICRS_catwise, row.e_DE_ICRS_catwise)*2., axis=1)
    elif catalog == 'unwise':
        df['PU_CU_'+catalog] = df.apply(lambda row: max(row.e_XposW1_unwise,row.e_XposW2_unwise,row.e_YposW1_unwise,row.e_YposW2_unwise)*2.75*2., axis=1)
    elif catalog =='allwise':
        df['PU_CU_'+catalog] = df['eeMaj_allwise']*2.
    elif catalog =='vphas':
        df['PU_CU_'+catalog] = 0.1
    
    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_CU_'+catalog].fillna(0.)**2 

    # Proper motion uncertainty: 
    df['e_PM_gaia'] = np.sqrt(df['e_pmRA_gaia']**2+df['e_pmDE_gaia']**2)

    if catalog == '2mass':
        df['MJD_2mass'] = df.apply(lambda row: Time(row.Date_2mass, format='isot').to_value('mjd', 'long') if pd.notnull(row.Date_2mass) else row, axis=1)
        df['MJD_2mass']  = pd.to_numeric(df['MJD_2mass'], errors='coerce')
        df['PU_PM_'+catalog] = df.apply(lambda row: row.PM_gaia*(row.MJD_2mass-X_mjd)/(365.*1e3),axis=1)
        df['PU_e_PM_'+catalog] = df.apply(lambda row: row.e_PM_gaia*(row.MJD_2mass-X_mjd)*2/(365.*1e3),axis=1)
    elif catalog=='catwise':
        df['PU_PM_'+catalog] = df.apply(lambda row: row.PM_gaia*(float(row.MJD_catwise)-X_mjd)/(365.*1e3),axis=1)
        df['PU_e_PM_'+catalog] = df.apply(lambda row: row.e_PM_gaia*(row.MJD_catwise-X_mjd)*2/(365.*1e3),axis=1)
    elif catalog=='vphas':
        df['PU_PM_'+catalog] = df.apply(lambda row: row.PM_gaia*(float(row.MJDu_vphas)-X_mjd)/(365.*1e3),axis=1)
        df['PU_e_PM_'+catalog] = df.apply(lambda row: row.e_PM_gaia*(row.MJDu_vphas-X_mjd)*2/(365.*1e3),axis=1)
    else:
        #if catalog != 'gaia':
        df['PU_PM_'+catalog] = df['PM_gaia']*mjd_difs[catalog]/(365.*1e3)
        df['PU_e_PM_'+catalog] = df['e_PM_gaia']*2.*mjd_difs[catalog]/(365.*1e3)
    
    if catalog != 'gaia':
        df['PU_'+catalog] = df['PU_'+catalog] + df['PU_PM_'+catalog].fillna(0.)**2 
    if catalog == 'gaia' and pm_cor==False:
        df['PU_'+catalog] = df['PU_'+catalog] + df['PU_PM_'+catalog].fillna(0.)**2 

    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_e_PM_'+catalog].fillna(0.)**2 

    # Parallax 
    df['PU_plx_'+catalog] = df['Plx_gaia']*2./1.e3
    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_plx_'+catalog].fillna(0.)**2
    df['PU_e_plx_'+catalog] = df['e_Plx_gaia']*2./1.e3
    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_e_plx_'+catalog].fillna(0.)**2
    # epsi (Excess noise)
    df['PU_epsi_'+catalog] = df['epsi_gaia']*2./1.e3
    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_epsi_'+catalog].fillna(0.)**2

    df['X_PU_'+catalog] = np.sqrt(df['PU_'+catalog]+df[X_PU]**2)
    df['PU_'+catalog] = np.sqrt(df['PU_'+catalog])
    
    df['cp_flag_'+catalog] = -8
   
    s = np.where((df['_r_'+catalog] > df['X_PU_'+catalog] ) & (df['_r_'+catalog] <= pu_factor*df['X_PU_'+catalog]))[0]
    df.loc[s,'cp_flag_'+catalog] = df.loc[s,'cp_flag_'+catalog] +4


    #'''
    if r2:
        # rs = np.array([ np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df['_rs_'+catalog]])
        #print(df['_rs_'+catalog])
        #df['_r2_'+catalog] = df.apply(lambda row: np.fromstring(row['_rs_'+catalog][1:-1], dtype=np.float, sep=',')[1] if len(np.fromstring(row['_rs_'+catalog][1:-1], dtype=np.float, sep=','))>1 else np.nan, axis=1)
        df['_r2_'+catalog] = df.apply(lambda row: np.sort(np.array(row['_rs_'+catalog]))[1] if len(np.array(row['_rs_'+catalog]))>1 else np.nan, axis=1)
        #print(df['_r2_'+catalog])

        s = np.where(( (df['_r2_'+catalog]<pu_factor*df['X_PU_'+catalog])) & (df['cp_flag_'+catalog]==-8) )[0]
        df.loc[s,'cp_flag_'+catalog] = df.loc[s,'cp_flag_'+catalog] +2
    
    #s = np.where((df['_r_'+catalog] > pu_factor*df['X_PU_'+catalog] ) )[0]
    #df.loc[s,'cp_flag_'+catalog] = df.loc[s,'cp_flag_'+catalog] + 1 

    s = np.where(df['_r_'+catalog]<=df['X_PU_'+catalog])[0]
    print(len(s), 'MW counterparts remained for',catalog)
    df.loc[s, 'cp_flag_'+catalog] = df.loc[s, 'cp_flag_'+catalog]+8
    
    #'''
    if 'rho_'+catalog in df.columns:
        df['prob_'+catalog]     = 1. - np.exp(-df['rho_'+catalog]*(np.pi*df['X_PU_'+catalog]**2))
        df['prob_log_'+catalog] = np.log10(df['prob_'+catalog])
    
    return df 

def CSC_counterpart_clean(df, X_PU='err_ellipse_r0',catalog='gaia',pu_factor=1.5):
    '''
    description
        drop counterparts outside pu_factor * X-ray PUs in 95% confidence level where X-ray PUs are taking into account of MU PUs, proper motion uncertainties, parallaxes (and their uncertainties), astrometric noises, if any of those are available; flag suspicious counterparts if they are outside 95% PUs or other counterparts are also near the X-ray sources 
    '''

    print(catalog)
    
    df['PU_'+catalog] = 0.

    # CU: Coordinate uncertainty 
    if catalog == 'gaia':
        df['PU_CU_'+catalog] = df.apply(lambda row: max(row.e_RA_ICRS_gaia, row.e_DE_ICRS_gaia)*2./1.e3, axis=1)
    elif catalog == 'gaiadist':
        df['PU_CU_'+catalog] = df['PU_CU_gaia']
    elif catalog == '2mass':
        df['PU_CU_'+catalog] = df['errMaj_2mass']*2.
    elif catalog == 'catwise':
        df['PU_CU_'+catalog] = df.apply(lambda row: max(row.e_RA_ICRS_catwise, row.e_DE_ICRS_catwise)*2., axis=1)
    elif catalog == 'unwise':
        df['PU_CU_'+catalog] = df.apply(lambda row: max(row.e_XposW1_unwise,row.e_XposW2_unwise,row.e_YposW1_unwise,row.e_YposW2_unwise)*2.75*2., axis=1)
    elif catalog =='allwise':
        df['PU_CU_'+catalog] = df['eeMaj_allwise']*2.
    elif catalog =='vphas':
        df['PU_CU_'+catalog] = 0.1

    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_CU_'+catalog].fillna(0.)**2 

    # Parallax 
    df['PU_plx_'+catalog] = df['Plx_gaia']*2./1.e3
    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_plx_'+catalog].fillna(0.)**2
    df['PU_e_plx_'+catalog] = df['e_Plx_gaia']*2./1.e3
    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_e_plx_'+catalog].fillna(0.)**2
    # epsi (Excess noise)
    df['PU_epsi_'+catalog] = df['epsi_gaia']*2./1.e3
    df['PU_'+catalog] = df['PU_'+catalog] + df['PU_epsi_'+catalog].fillna(0.)**2

    df['X_PU_'+catalog] = np.sqrt(df['PU_'+catalog]+df[X_PU]**2)
    df['PU_'+catalog] = np.sqrt(df['PU_'+catalog])

    
    df['cp_flag_'+catalog] = -8
   
    s = np.where((df['_r_'+catalog] > df['X_PU_'+catalog] ) & (df['_r_'+catalog] <= pu_factor*df['X_PU_'+catalog]))[0]
    df.loc[s,'cp_flag_'+catalog] = df.loc[s,'cp_flag_'+catalog] +4

    #s = np.where((df['_r_'+catalog] > pu_factor*df['X_PU_'+catalog] ) )[0]
    #df.loc[s,'cp_flag_'+catalog] = df.loc[s,'cp_flag_'+catalog] +2

    #df['_r2_'+catalog] = df.apply(lambda row: np.sort(np.array(row['_rs_'+catalog]))[1] if len(np.fromstring(np.array(row['_rs_'+catalog])))>1 else np.nan, axis=1)
        #print(df['_r2_'+catalog])

    df[['_rs_'+catalog]] = df[['_rs_'+catalog]].fillna('[]')
    df['_r2_'+catalog] = df.apply(lambda row: np.fromstring(row['_rs_'+catalog][1:-1], dtype=np.float, sep=', ')[1] if len(np.fromstring(row['_rs_'+catalog][1:-1], dtype=np.float, sep=', '))>1 else np.nan, axis=1)
    #s = np.where(((df['_r2_'+catalog]<1.5*df['_r_'+catalog]) | (df['_r2_'+catalog]<pu_factor*df['X_PU_'+catalog])) & (df['cp_flag_'+catalog]==0) )[0]
    #df.loc[s,'cp_flag_'+catalog] = df.loc[s,'cp_flag_'+catalog] +1 

    s = np.where(( (df['_r2_'+catalog]<=pu_factor*df['X_PU_'+catalog])) & (df['cp_flag_'+catalog]==-8) )[0]
    df.loc[s,'cp_flag_'+catalog] = df.loc[s,'cp_flag_'+catalog] +2

    s = np.where(df['_r_'+catalog]<=df['X_PU_'+catalog])[0]
    print(len(s), 'MW counterparts remained for',catalog)
    df.loc[s, 'cp_flag_'+catalog] = df.loc[s, 'cp_flag_'+catalog]+8
    
    if 'rho_'+catalog in df.columns:
        df['prob_'+catalog]     = 1. - np.exp(-df['rho_'+catalog]*(np.pi*df['X_PU_'+catalog]**2))
        df['prob_log_'+catalog] = np.log10(df['prob_'+catalog])
    
    return df 

def confusion_clean(df, X_PU='err_ellipse_r0',X_mjd=57388.,Chandratype='CSC'):
    if X_PU == 'err_ellipse_r0':
        df.loc[:,'PU'] = df['err_ellipse_r0']
    if Chandratype=='CSC-CXO':
        for cat in ['gaia','2mass','catwise','unwise','allwise','vphas']:
            df = counterpart_clean(df, X_PU=X_PU, catalog=cat,X_mjd=X_mjd,pu_factor=1.5)
    elif Chandratype=='CSC':
        for cat in ['gaia','2mass','catwise','unwise','allwise']:
            df = CSC_counterpart_clean(df, X_PU=X_PU, catalog=cat,pu_factor=1.5)
    return df   

def remove_sources(CSC, remove_codes=[1, 2, 4, 8, 16, 32, 64], dtype='TD'):
    
    if 1 in remove_codes:
        print("remove_code = 1: CSC flags")
        flags = ['extent_flag', 'pileup_flag','sat_src_flag', 'streak_src_flag','conf_flag']
        flags_remove = ['sat_src_flag', 'streak_src_flag']
        flags_record = ['extent_flag','conf_flag','pileup_flag']

        for flag in flags:
            print(len(CSC[CSC[flag]!=False]), 'sources with True ', flag)
            print(flag, sorted(Counter(CSC[flag]).items()))

        s = np.where( (CSC['sat_src_flag']==1) | (CSC['streak_src_flag']==True)  )[0]
        CSC.loc[s, 'remove_code'] = CSC.loc[s, 'remove_code']+1

        for i in CSC.index:
            CSC_flags = []
            if CSC.loc[i, "extent_flag"] == True:
                CSC_flags.append("extent")
            if CSC.loc[i, "conf_flag"] == 1:
                CSC_flags.append("conf")
            if CSC.loc[i, "pileup_flag"] == 1:
                CSC_flags.append("pileup")
            CSC.loc[i,"CSC_flags"] = "|".join(CSC_flags)
        
        if dtype =='TD':
            print('Remove', len(s), sorted(Counter(CSC.loc[s, 'Class']).items()))
            print('Left', len(CSC[CSC['remove_code']==0]), sorted(Counter(CSC[CSC['remove_code']==0]['Class']).items()))
        elif dtype=='CSC':
            print('Remove', len(s))
            print('Left', len(CSC[CSC['remove_code']==0]))

    if 32 in remove_codes:
        print("remove_code = 32: NaN and/or zero fluxes in multiple bands")
        
        print(sorted(Counter(CSC['flux_flag']).items()))
        if dtype =='TD':
            print(sorted(Counter(CSC.loc[CSC['flux_flag']==7,'Class']).items()))
        CSC = CSC.replace(np.nan, exnum)
        fn = 'flux_aper90_ave_'
        #s = np.where(((CSC[fn+'h']==exnum) | (CSC[fn+'m']==exnum) | (CSC[fn+'s']==exnum)) | ((CSC[fn+'h']==0) & (CSC[fn+'m']==0) & (CSC[fn+'s']>0)) | ((CSC[fn+'h']==0) & (CSC[fn+'m']>0) & (CSC[fn+'s']==0)) | ((CSC[fn+'h']>0) & (CSC[fn+'m']==0) & (CSC[fn+'s']==0)) | ((CSC[fn+'h']==0) & (CSC[fn+'m']==0) & (CSC[fn+'s']==0)) )[0]
        #s = np.where( (CSC[fn+'h']==exnum) | (CSC['e_'+fn+'h']==exnum) | (CSC[fn+'m']==exnum) | (CSC['e_'+fn+'m']==exnum) | (CSC[fn+'s']==exnum) | (CSC['e_'+fn+'s']==exnum) | (CSC[fn+'b']==exnum)  | (CSC['e_'+fn+'b']==exnum) )[0]
        s = np.where( CSC['flux_flag']==7 )[0]
        CSC.loc[s, 'remove_code'] = CSC.loc[s, 'remove_code'] + 32

        if dtype == 'CXO' or dtype == 'CSC':
            print('Remove', len(s), 'sources.')
            print('Left', len(CSC[CSC['remove_code']==0]))
        elif dtype == 'TD': 
            print('Remove', len(s), sorted(Counter(CSC.loc[s, 'Class']).items()), 'sources removed.')
            print('Left', len(CSC[CSC['remove_code']==0]), sorted(Counter(CSC[CSC['remove_code']==0]['Class']).items()))


        CSC = CSC.replace(exnum, np.nan)
        #s= np.where(((CSC[fn+'b']==exnum) | (CSC[fn+'b']==0)) &  ((CSC[fn+'h']>=0) | (CSC[fn+'m']>=0) | (CSC[fn+'s']>=0))  )[0]
        #CSC.loc[s, fn+'b'] = CSC.loc[s, fn+'h'] + CSC.loc[s, fn+'m'] + CSC.loc[s, fn+'s']
        #print(len(s), ' flux_b re-calculated.')

    if 64 in remove_codes:
        print("remove_code = 64: STAR and YSO without any MW counterparts")
        CSC = CSC.replace(np.nan, exnum)

        MW_features = ['Gmag','BPmag','RPmag','Jmag','Hmag','Kmag','W1mag_comb','W2mag_comb','W3mag_allwise']


        MW_classes = ['HM-STAR', 'LM-STAR','YSO']

        s1 = pd.Index(np.where(CSC.Class.isin(MW_classes))[0])
        s2 = pd.Index(np.where(~CSC[MW_features].isin([exnum]))[0])
        s = s1.difference(s2)

        CSC.loc[s, 'remove_code'] = CSC.loc[s, 'remove_code'] + 64

        print('Remove', len(s), sorted(Counter(CSC.loc[s, 'Class']).items()), 'sources removed.')
        print('Left', len(CSC[CSC['remove_code']==0]), sorted(Counter(CSC[CSC['remove_code']==0]['Class']).items()))
        CSC = CSC.replace(exnum, np.nan)

    return CSC

def TD_clean_vizier(TD, NS_clean=True, remove_codes = [1, 32, 64]):


    # take the subset and calculate UnWISE magnitudes
    fn = 'flux_aper90_ave_'
    cols = ['name_cat','ra_cat','dec_cat','e_Pos','Class','SubClass','ref','remove_code','sep','name','ra','dec','err_ellipse_r0','err_ellipse_r1','err_ellipse_ang','significance','significance_max',
    'extent_flag', 'pileup_flag','sat_src_flag', 'streak_src_flag','conf_flag',
    fn+'b',fn+'h',fn+'m',fn+'s','e_'+fn+'b','e_'+fn+'h','e_'+fn+'m','e_'+fn+'s','flux_flag',
    'ks_intra_prob_b','kp_intra_prob_b','var_inter_prob_b','kp_prob_b_max','var_inter_prob','dof','chisqr',
    'EDR3Name_gaia','Plx_gaia','e_Plx_gaia','X_PU_gaia','Gmag_gaia','e_Gmag_gaia','BPmag_gaia','e_BPmag_gaia','RPmag_gaia','e_RPmag_gaia','PM_gaia','pmRA_gaia',
    'e_pmRA_gaia','pmDE_gaia','e_pmDE_gaia','_r_gaia','prob_log_gaia',
    'X_PU_2mass','Jmag_2mass','Hmag_2mass','Kmag_2mass','e_Jmag_2mass','e_Hmag_2mass','e_Kmag_2mass','_r_2mass','prob_log_2mass',
    'X_PU_catwise','W1mproPM_catwise','e_W1mproPM_catwise','W2mproPM_catwise','e_W2mproPM_catwise','pmRA_catwise','pmDE_catwise','_r_catwise','prob_log_catwise',
    'X_PU_unwise','FW1_unwise','e_FW1_unwise','FW2_unwise','e_FW2_unwise','_r_unwise','prob_log_unwise','W1mag_allwise','W2mag_allwise','W3mag_allwise','W4mag_allwise','e_W1mag_allwise','e_W2mag_allwise','e_W3mag_allwise',
    'X_PU_allwise','e_W4mag_allwise','_r_allwise','prob_log_allwise','X_PU_gaiadist','rgeo_gaiadist','rpgeo_gaiadist','_r_gaiadist','prob_log_gaiadist',
    'b_rgeo_gaiadist','B_rgeo_gaiadist','b_rpgeo_gaiadist','B_rpgeo_gaiadist','Flag_gaiadist',
    'main_id','main_type','_r_simbad']
    cols_sub = ['prob_log_gaia', 'prob_log_2mass', 'prob_log_catwise', 'prob_log_unwise', 'prob_log_allwise', 'prob_log_gaiadist']
    cols_f = [item for item in cols if item not in cols_sub]
    CSC = TD[cols_f]


    CSC.loc[CSC['Plx_gaia']<0, ['Plx_gaia', 'e_Plx_gaia']]=np.nan
    CSC.loc[CSC['Plx_gaia']/CSC['e_Plx_gaia']<2, ['Plx_gaia', 'e_Plx_gaia']]=np.nan

    CSC.loc[:, 'W1mag_unwise'], CSC.loc[:, 'W2mag_unwise'] = np.nan, np.nan
    CSC.loc[CSC['FW1_unwise']>0,'W1mag_unwise'] = 22.5-2.5*np.log10(CSC.loc[CSC['FW1_unwise']>0,'FW1_unwise']) - 0.004
    CSC.loc[CSC['FW2_unwise']>0,'W2mag_unwise'] = 22.5-2.5*np.log10(CSC.loc[CSC['FW2_unwise']>0,'FW2_unwise']) - 0.032
    CSC.loc[:, 'e_W1mag_unwise'], CSC.loc[:, 'e_W2mag_unwise'] = np.nan, np.nan
    CSC.loc[CSC['FW1_unwise']>0,'e_W1mag_unwise'] = -2.5*np.log10(math.e)*CSC.loc[CSC['FW1_unwise']>0,'e_FW1_unwise']/CSC.loc[CSC['FW1_unwise']>0,'FW1_unwise'] 
    CSC.loc[CSC['FW2_unwise']>0,'e_W2mag_unwise'] = -2.5*np.log10(math.e)*CSC.loc[CSC['FW2_unwise']>0,'e_FW2_unwise']/CSC.loc[CSC['FW2_unwise']>0,'FW2_unwise']


    CSC = CSC.rename(columns={'Gmag_gaia':'Gmag','e_Gmag_gaia':'e_Gmag',
            'BPmag_gaia':'BPmag','e_BPmag_gaia':'e_BPmag','RPmag_gaia':'RPmag','e_RPmag_gaia':'e_RPmag',
            'Jmag_2mass':'Jmag','Hmag_2mass':'Hmag','Kmag_2mass':'Kmag','e_Jmag_2mass':'e_Jmag','e_Hmag_2mass':'e_Hmag','e_Kmag_2mass':'e_Kmag',
            'W1mproPM_catwise':'W1mag_catwise','W2mproPM_catwise':'W2mag_catwise','e_W1mproPM_catwise':'e_W1mag_catwise','e_W2mproPM_catwise':'e_W2mag_catwise',
            'rgeo_gaiadist':'rgeo','rpgeo_gaiadist':'rpgeo','b_rgeo_gaiadist':'b_rgeo','B_rgeo_gaiadist':'B_rgeo','b_rpgeo_gaiadist':'b_rpgeo','B_rpgeo_gaiadist':'B_rpgeo','main_id_simbad':'main_id','main_type_simbad':'main_type'})
    
    CSC = CSC.replace(r'^\s*$', np.nan, regex=True)
    CSC = CSC.apply(pd.to_numeric, errors='ignore')
    CSC = CSC.replace(np.nan, exnum)

    # Removing unreliable MW features from large PUs

    for MW_feature, sep in zip([['EDR3Name_gaia','Plx_gaia','e_Plx_gaia','Gmag', 'BPmag', 'RPmag','e_Gmag', 'e_BPmag', 'e_RPmag','pm_gaia','pmRA_gaia','pmDE_gaia'], ['rgeo','rpgeo','b_rgeo','B_rgeo','b_rpgeo','B_rpgeo','Flag_gaiadist'], ['Jmag', 'Hmag', 'Kmag','e_Jmag', 'e_Hmag', 'e_Kmag'],['W1mag_catwise','W2mag_catwise','e_W1mag_catwise','e_W2mag_catwise','pmRA_catwise','pmDE_catwise'],['W1mag_unwise', 'W2mag_unwise','e_W1mag_unwise', 'e_W2mag_unwise'], ['W1mag_allwise','W2mag_allwise','W3mag_allwise','W4mag_allwise','e_W1mag_allwise','e_W2mag_allwise','e_W3mag_allwise','e_W4mag_allwise'],['main_id','main_type']], ['_gaia','_gaiadist', '_2mass','_catwise','_unwise','_allwise','_simbad']):
        print('angDist'+sep)
        if sep != '_simbad':
            s = np.where(CSC['_r'+sep]>CSC['X_PU'+sep])[0]
        elif sep == '_simbad':
            s = np.where(CSC['_r'+sep]>CSC['err_ellipse_r0'])[0]
        #print(CSC.loc[s, ['err_ellipse_r0','_r'+sep]])
        print(sorted(Counter(CSC['Class'][s]).items()), 'MW counterparts dropped.')
        CSC.loc[s, MW_feature] = exnum

    # Use Simbad name is name_cat is missing, use CSC name if both Simbad name and name_cat are missing    
    
    #print(len(CSC[CSC['name_cat']==exnum]))
    #print(len(CSC))
    CSC.loc[CSC['name_cat'] == exnum, 'name_cat'] = CSC.loc[CSC['name_cat'] == exnum, 'main_id']
#print(CSC.loc[CSC['name'] == exnum, 'main_id']) # CSC.apply(lambda row: row.main_id if row.name == exnum else row.name, axis=1)
    #print(len(CSC[CSC['name_cat']==exnum]))
    CSC.loc[CSC['name_cat'] == exnum, 'name_cat'] = CSC.loc[CSC['name_cat'] == exnum, 'name']

    # obtain combined WISE magnitudes from CatWISE2020 and unWISE
    
    #CSC['W1mag_comb'], CSC['W2mag_comb']  = CSC['W1mag_catwise'], CSC['W1mag_catwise']
    #CSC['e_W1mag_comb'], CSC['e_W2mag_comb']  = CSC['e_W1mag_catwise'], CSC['e_W1mag_catwise']

    for w in ['W1', 'W2']:
        CSC[w+'mag_comb'] = CSC[w+'mag_catwise']
        CSC['e_'+w+'mag_comb'] = CSC['e_'+w+'mag_catwise']
        s= np.where((CSC[w+'mag_comb']==exnum) & (CSC[w+'mag_unwise']!=exnum))[0]
        print(sorted(Counter(CSC.loc[s,'Class']).items()), 'use ',w,' from UnWISE rather than CatWISE2020.')
        CSC.loc[s, w+'mag_comb'] = CSC.loc[s, w+'mag_unwise']
        CSC.loc[s, 'e_'+w+'mag_comb'] = CSC.loc[s, 'e_'+w+'mag_unwise']

    # clean PSR MW
    #'''
    counterpart_features = ['EDR3Name_gaia','Plx_gaia','e_Plx_gaia','Gmag','BPmag','RPmag','e_Gmag','e_BPmag','e_RPmag','pm_gaia','pmRA_gaia','pmDE_gaia','_r_gaia','Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag','_r_2mass','W1mag_catwise','W2mag_catwise','e_W1mag_catwise','e_W2mag_catwise','pmRA_catwise','pmDE_catwise','_r_catwise','W1mag_unwise','W2mag_unwise','e_W1mag_unwise','e_W2mag_unwise','_r_unwise','W1mag_comb','W2mag_comb','e_W1mag_comb','e_W2mag_comb','W1mag_allwise','W2mag_allwise','W3mag_allwise','W4mag_allwise','e_W1mag_allwise','e_W2mag_allwise','e_W3mag_allwise','e_W4mag_allwise','_r_allwise','rgeo','rpgeo','_r_gaiadist','b_rgeo','B_rgeo','b_rpgeo','B_rpgeo','Flag_gaiadist']
    
    print("NS confusion:")
    for col in counterpart_features:
        numopt = len(CSC.loc[(CSC.Class=='NS') & (CSC[col] != exnum)])
        numpsr = len(CSC.loc[(CSC.Class=='NS')])
        print(str(numopt), col, "has a fraction of ",str(float(numopt)/numpsr))
    if NS_clean:    
        CSC.loc[CSC.Class=='NS',counterpart_features]=exnum
    #'''
    CSC = remove_sources(CSC, remove_codes) 

    CSC = CSC.replace(exnum, np.nan)
    print(len(CSC))
    print(sorted(Counter(CSC['Class']).items()))
    #CSC = CSC.drop(columns=['FW1_unwise','FW2_unwise','sep_LMC','sep_SMC','sep_Wd1','_r_gaia','_r_gaiadist', '_r_2mass','_r_catwise','_r_unwise','_r_allwise','_r_simbad'])
    #CSC = CSC.drop(columns=['extent_flag', 'pileup_flag','sat_src_flag', 'streak_src_flag','conf_flag','FW1_unwise','FW2_unwise','sep_LMC','sep_SMC','sep_Wd1','_r_gaia','_r_gaiadist', '_r_2mass','_r_catwise','_r_unwise','_r_allwise','_r_simbad'])
    CSC = CSC.drop(columns=['extent_flag', 'pileup_flag','sat_src_flag', 'streak_src_flag','conf_flag','FW1_unwise','FW2_unwise'])#,'_r_gaia','_r_gaiadist', '_r_2mass','_r_catwise','_r_unwise','_r_allwise','_r_simbad'])
    #CSC.to_csv('CSC_TD_v5_Topcat_removecode_v9.csv',index=False)

    print(len(CSC[CSC['remove_code']==0]))
    print(sorted(Counter(CSC[CSC['remove_code']==0]['Class']).items()))

    return CSC     


def create_field_csc_data(data_dir,field_name,ra,dec,radius):#,name_type='CSCview', name_col='name', ra_col='ra',dec_col='dec',coord_format='hms'):
    '''
    description:
        extract the CSC 2.0 data for the field using ADQL from http://cda.cfa.harvard.edu/csccli/getProperties URL

    input:
        data_dir: the directory to store the per-obs data
        field_name: the field name
        ra,dec,radius: center coordinate of the field and searching radius in arcmin
        name_col & ra_col & dec_col: column name of CSC 2.0 names, right ascension and declination
        coord_format: hms or deg format for the coordinates

    output:
        no output
        each individual per-obs data is saved as a txt file in query_dir
        the combined per-obs data is saved as a csv file

    '''


    ra_low  = ra  - radius/60.
    ra_upp  = ra  + radius/60.
    dec_low = dec - radius/60.
    dec_upp = dec + radius/60.
    rad_cone = radius
    
    f = open(f'{data_dir}/cscfield_query_template.adql', "r")
    adql = f.readline()
    ra_temp = '266.599396'
    dec_temp = '-28.87594'
    ra_low_temp = '266.5898794490786'
    ra_upp_temp = '266.60891255092145'
    dec_low_temp = '-28.884273333333333'
    dec_upp_temp = '-28.867606666666667'
    rad_cone_temp = '0.5'
    
    for [str1, str2] in [[rad_cone, rad_cone_temp], [ra, ra_temp], [dec, dec_temp], [ra_low, ra_low_temp], [ra_upp, ra_upp_temp], [dec_low, dec_low_temp], [dec_upp, dec_upp_temp]]:
        adql = adql.replace(str2, str(str1))
    
    text_file = open(f'{data_dir}/{field_name}.adql', "w")
    text_file.write(adql)
    text_file.close()

    os.system("curl -o "+data_dir+'/'+field_name+"_csc.txt \
        --form query=@"+data_dir+'/'+field_name+".adql \
        http://cda.cfa.harvard.edu/csccli/getProperties")

    df = pd.read_csv(f'{data_dir}/{field_name}_csc.txt', header=154, sep='\t')
    #df['usrid'] = usrid+1
    df.name = df.name.str.lstrip()
    #df.to_csv(f'{data_dir}/{field_name}_csc_per.csv', index=False)

    return df

def CSC_clean_keepcols(CSC, remove_codes = [1, 32], withvphas=False):

    print(len(CSC), ' sources in total.')
    # take the subset and calculate UnWISE magnitudes
    #fn = 'flux_aper90_ave_'

    CSC.loc[:, 'W1mag_unwise'], CSC.loc[:, 'W2mag_unwise'] = np.nan, np.nan
    CSC.loc[CSC['FW1_unwise']>0,'W1mag_unwise'] = 22.5-2.5*np.log10(CSC.loc[CSC['FW1_unwise']>0,'FW1_unwise']) - 0.004
    CSC.loc[CSC['FW2_unwise']>0,'W2mag_unwise'] = 22.5-2.5*np.log10(CSC.loc[CSC['FW2_unwise']>0,'FW2_unwise']) - 0.032
    CSC.loc[:, 'e_W1mag_unwise'], CSC.loc[:, 'e_W2mag_unwise'] = np.nan, np.nan
    CSC.loc[CSC['FW1_unwise']>0,'e_W1mag_unwise'] = -2.5*np.log10(math.e)*CSC.loc[CSC['FW1_unwise']>0,'e_FW1_unwise']/CSC.loc[CSC['FW1_unwise']>0,'FW1_unwise']
    CSC.loc[CSC['FW2_unwise']>0,'e_W2mag_unwise'] = -2.5*np.log10(math.e)*CSC.loc[CSC['FW2_unwise']>0,'e_FW2_unwise']/CSC.loc[CSC['FW2_unwise']>0,'FW2_unwise']

    CSC = CSC.replace(r'^\s*$', exnum, regex=True)
    CSC = CSC.apply(pd.to_numeric, errors='ignore')
    CSC = CSC.replace(np.nan, exnum)

    '''
    CSC = CSC.rename(columns={'Gmag_gaia':'Gmag','e_Gmag_gaia':'e_Gmag',
            'BPmag_gaia':'BPmag','e_BPmag_gaia':'e_BPmag','RPmag_gaia':'RPmag','e_RPmag_gaia':'e_RPmag',
            'Jmag_2mass':'Jmag','Hmag_2mass':'Hmag','Kmag_2mass':'Kmag','e_Jmag_2mass':'e_Jmag','e_Hmag_2mass':'e_Hmag','e_Kmag_2mass':'e_Kmag',
            'W1mproPM_catwise':'W1mag_catwise','W2mproPM_catwise':'W2mag_catwise','e_W1mproPM_catwise':'e_W1mag_catwise','e_W2mproPM_catwise':'e_W2mag_catwise',
            'rgeo_gaiadist':'rgeo','rpgeo_gaiadist':'rpgeo','b_rgeo_gaiadist':'b_rgeo','B_rgeo_gaiadist':'B_rgeo','b_rpgeo_gaiadist':'b_rpgeo','B_rpgeo_gaiadist':'B_rpgeo'})#,
            #'main_id_simbad':'main_id','main_type_simbad':'main_type'})
    if withvphas:
        CSC = CSC.rename(columns={'gmag_vphas':'gmag','e_gmag_vphas':'e_gmag','umag_vphas':'umag','e_umag_vphas':'e_umag',
            'rmag_vphas':'rmag','e_rmag_vphas':'e_rmag', 'imag_vphas':'imag', 'e_imag_vphas':'e_imag'})
    '''

    '''
    # Removing unreliable MW features from large PUs

    for cat in ['gaia','2mass','catwise','unwise','allwise']:
        print('angDist_'+cat)
        s = np.where(CSC['_r_'+cat]<=CSC['X_PU_'+cat])[0]
        #print(CSC.loc[s, ['err_ellipse_r0','_r'+sep]])
        print(len(s), 'MW counterparts dropped.')
        CSC.loc[s, 'cp_flag_'+cat] = CSC.loc[s, 'cp_flag_'+cat]+8

    if withvphas:
        print('angDist_vphas')
        s = np.where(CSC['_r_vphas']<=CSC['X_PU_vphas'])[0]
        print(len(s), 'MW counterparts dropped.')
        CSC.loc[s, 'cp_flag_vphas'] = CSC.loc[s, 'cp_flag_vphas']+8
    '''    
    # obtain combined WISE magnitudes from CatWISE2020 and unWISE

    #'''
    CSC = remove_sources(CSC, [1, 32], dtype='CSC')

    CSC = CSC.replace(exnum, np.nan)
    #CSC = CSC.drop(columns=['FW1_unwise','FW2_unwise','_r_gaia','_r_gaiadist', '_r_2mass','_r_catwise','_r_unwise','_r_allwise'])
    #CSC.to_csv('CSC_TD_v5_Topcat_removecode_v9.csv',index=False)

    print(len(CSC[CSC['remove_code']==0]))

    return CSC


def CSC_clean(data, remove_codes = [1, 32], withvphas=False):

    print(len(data), ' sources in total.')
    # take the subset and calculate UnWISE magnitudes
    fn = 'flux_aper90_ave_'
    cols = ['usrid','name','remove_code','ra','dec','err_ellipse_r0','err_ellipse_r1','err_ellipse_ang','significance',
    'extent_flag', 'pileup_flag','sat_src_flag', 'streak_src_flag','conf_flag',
    fn+'b',fn+'h',fn+'m',fn+'s','e_'+fn+'b','e_'+fn+'h','e_'+fn+'m','e_'+fn+'s','flux_flag',
    'ks_intra_prob_b','kp_intra_prob_b','var_inter_prob_b','kp_prob_b_max','var_inter_prob','dof','chisqr',
    'X_PU_gaia','EDR3Name_gaia','Plx_gaia','e_Plx_gaia','Gmag_gaia','e_Gmag_gaia','BPmag_gaia','e_BPmag_gaia','RPmag_gaia','e_RPmag_gaia','PM_gaia','pmRA_gaia',
    'e_pmRA_gaia','pmDE_gaia','e_pmDE_gaia','_r_gaia','prob_log_gaia',
    'X_PU_2mass','Jmag_2mass','Hmag_2mass','Kmag_2mass','e_Jmag_2mass','e_Hmag_2mass','e_Kmag_2mass','_r_2mass','prob_log_2mass',
    'X_PU_catwise','RA_ICRS_catwise','DE_ICRS_catwise','W1mproPM_catwise','e_W1mproPM_catwise','W2mproPM_catwise','e_W2mproPM_catwise','pmRA_catwise','pmDE_catwise','_r_catwise','prob_log_catwise',
    'X_PU_unwise','RAJ2000_unwise','DEJ2000_unwise','FW1_unwise','e_FW1_unwise','FW2_unwise','e_FW2_unwise','_r_unwise','prob_log_unwise','X_PU_allwise','W1mag_allwise','W2mag_allwise','W3mag_allwise','W4mag_allwise','e_W1mag_allwise','e_W2mag_allwise','e_W3mag_allwise',
    'e_W4mag_allwise','_r_allwise','prob_log_allwise','X_PU_gaiadist','rgeo_gaiadist','rpgeo_gaiadist','_r_gaiadist','prob_log_gaiadist',
    'b_rgeo_gaiadist','B_rgeo_gaiadist','b_rpgeo_gaiadist','B_rpgeo_gaiadist','Flag_gaiadist']

    CSC = data#[cols]

    CSC.loc[:, 'W1mag_unwise'], CSC.loc[:, 'W2mag_unwise'] = np.nan, np.nan
    CSC.loc[CSC['FW1_unwise']>0,'W1mag_unwise'] = 22.5-2.5*np.log10(CSC.loc[CSC['FW1_unwise']>0,'FW1_unwise']) - 0.004
    CSC.loc[CSC['FW2_unwise']>0,'W2mag_unwise'] = 22.5-2.5*np.log10(CSC.loc[CSC['FW2_unwise']>0,'FW2_unwise']) - 0.032
    CSC.loc[:, 'e_W1mag_unwise'], CSC.loc[:, 'e_W2mag_unwise'] = np.nan, np.nan
    CSC.loc[CSC['FW1_unwise']>0,'e_W1mag_unwise'] = -2.5*np.log10(math.e)*CSC.loc[CSC['FW1_unwise']>0,'e_FW1_unwise']/CSC.loc[CSC['FW1_unwise']>0,'FW1_unwise']
    CSC.loc[CSC['FW2_unwise']>0,'e_W2mag_unwise'] = -2.5*np.log10(math.e)*CSC.loc[CSC['FW2_unwise']>0,'e_FW2_unwise']/CSC.loc[CSC['FW2_unwise']>0,'FW2_unwise']


    CSC = CSC.rename(columns={'Gmag_gaia':'Gmag','e_Gmag_gaia':'e_Gmag',
            'BPmag_gaia':'BPmag','e_BPmag_gaia':'e_BPmag','RPmag_gaia':'RPmag','e_RPmag_gaia':'e_RPmag',
            'Jmag_2mass':'Jmag','Hmag_2mass':'Hmag','Kmag_2mass':'Kmag','e_Jmag_2mass':'e_Jmag','e_Hmag_2mass':'e_Hmag','e_Kmag_2mass':'e_Kmag',
            'W1mproPM_catwise':'W1mag_catwise','W2mproPM_catwise':'W2mag_catwise','e_W1mproPM_catwise':'e_W1mag_catwise','e_W2mproPM_catwise':'e_W2mag_catwise',
            'rgeo_gaiadist':'rgeo','rpgeo_gaiadist':'rpgeo','b_rgeo_gaiadist':'b_rgeo','B_rgeo_gaiadist':'B_rgeo','b_rpgeo_gaiadist':'b_rpgeo','B_rpgeo_gaiadist':'B_rpgeo'})#,
            #'main_id_simbad':'main_id','main_type_simbad':'main_type'})
    if withvphas:
        CSC = CSC.rename(columns={'gmag_vphas':'gmag','e_gmag_vphas':'e_gmag','umag_vphas':'umag','e_umag_vphas':'e_umag',
            'rmag_vphas':'rmag','e_rmag_vphas':'e_rmag', 'imag_vphas':'imag', 'e_imag_vphas':'e_imag'})

    CSC = CSC.replace(r'^\s*$', exnum, regex=True)
    CSC = CSC.apply(pd.to_numeric, errors='ignore')
    CSC = CSC.replace(np.nan, exnum)

    # Removing unreliable MW features from large PUs

    for MW_feature, sep in zip([['EDR3Name_gaia','Source_gaia','Plx_gaia','e_Plx_gaia','Gmag', 'BPmag', 'RPmag','e_Gmag', 'e_BPmag', 'e_RPmag','pm_gaia','pmRA_gaia','pmDE_gaia'], ['rgeo','rpgeo','b_rgeo','B_rgeo','b_rpgeo','B_rpgeo','Flag_gaiadist','Source_gaiadist'], ['Jmag', 'Hmag', 'Kmag','e_Jmag', 'e_Hmag', 'e_Kmag','_2MASS_2mass'],['W1mag_catwise','W2mag_catwise','e_W1mag_catwise','e_W2mag_catwise','pmRA_catwise','pmDE_catwise','objID_catwise'],['W1mag_unwise', 'W2mag_unwise','e_W1mag_unwise', 'e_W2mag_unwise','objID_unwise'], ['W1mag_allwise','W2mag_allwise','W3mag_allwise','W4mag_allwise','e_W1mag_allwise','e_W2mag_allwise','e_W3mag_allwise','e_W4mag_allwise','AllWISE_allwise']], ['_gaia','_gaiadist', '_2mass','_catwise','_unwise','_allwise']):
        print('angDist'+sep)
        s = np.where(CSC['_r'+sep]>CSC['X_PU'+sep])[0]
        #print(CSC.loc[s, ['err_ellipse_r0','_r'+sep]])
        print(len(s), 'MW counterparts dropped.')
        CSC.loc[s, MW_feature] = exnum
    if withvphas:
        print('angDist_vphas')
        s = np.where(CSC['_r_vphas']>CSC['X_PU_vphas'])[0]
        print(len(s), 'MW counterparts dropped.')
        CSC.loc[s, ['sourceID', 'VPHASDR2_vphas', 'gmag','e_gmag','umag','e_umag','rmag','e_rmag','sourceID_vphas']] = exnum

    # obtain combined WISE magnitudes from CatWISE2020 and unWISE

    #CSC['W1mag_comb'], CSC['W2mag_comb']  = CSC['W1mag_catwise'], CSC['W1mag_catwise']
    #CSC['e_W1mag_comb'], CSC['e_W2mag_comb']  = CSC['e_W1mag_catwise'], CSC['e_W1mag_catwise']

    for w in ['W1', 'W2']:
        CSC[w+'mag_comb'] = CSC[w+'mag_catwise']
        CSC['e_'+w+'mag_comb'] = CSC['e_'+w+'mag_catwise']
        s= np.where((CSC[w+'mag_comb']==exnum) & (CSC[w+'mag_unwise']!=exnum))[0]
        print(len(s), 'use ',w,' from UnWISE rather than CatWISE2020.')
        CSC.loc[s, w+'mag_comb'] = CSC.loc[s, w+'mag_unwise']
        CSC.loc[s, 'e_'+w+'mag_comb'] = CSC.loc[s, 'e_'+w+'mag_unwise']
    for ps in ['RA','DE']:
        CSC[ps+'deg_comb'] = CSC[ps+'_ICRS_catwise']
        s= np.where((CSC[ps+'_ICRS_catwise']==exnum) & (CSC[ps+'J2000_unwise']!=exnum))[0]
        print(len(s), 'use ',ps,' from UnWISE rather than CatWISE2020.')
        CSC.loc[s, ps+'deg_comb'] = CSC.loc[s, ps+'J2000_unwise']

    #'''
    CSC = remove_sources(CSC, [1, 32], dtype='CSC')

    CSC = CSC.replace(exnum, np.nan)
    #CSC = CSC.drop(columns=['FW1_unwise','FW2_unwise','_r_gaia','_r_gaiadist', '_r_2mass','_r_catwise','_r_unwise','_r_allwise'])
    #CSC.to_csv('CSC_TD_v5_Topcat_removecode_v9.csv',index=False)

    print(len(CSC[CSC['remove_code']==0]))

    return CSC


def Gaia_counterparts(df_gaia, file_dir, field_name):

    for cat in ['2mass','allwise']:

        if cat=='gaiadist':
            job = Gaia.launch_job_async("""select source_id, d.r_med_geo, d.r_lo_geo, d.r_hi_geo, d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo
            from gaiaedr3.gaia_source as gaia
            join external.gaiaedr3_distance as d using (source_id)
            where
            1 = contains(
                point('', 166.28875, -58.85),
                circle('', gaia.ra, gaia.dec, 0.5)
            )""",
            dump_to_file=False, output_format='csv')
        if cat=='2mass':
            job = Gaia.launch_job_async("""select source_id, n.original_ext_source_id, t.ra, t.dec, t.j_m, t.j_msigcom, t.h_m, t.h_msigcom, t.ks_m, t.ks_msigcom
            from gaiaedr3.gaia_source as gaia
            join gaiaedr3.tmass_psc_xsc_best_neighbour as n using (source_id)
            INNER join gaiadr1.tmass_original_valid as t on (n.original_ext_source_id = t.designation)
            where
            1 = contains(
                point('', 166.28875, -58.85),
                circle('', gaia.ra, gaia.dec, 0.5)
            )""",
            dump_to_file=False, output_format='csv')
        if cat=='allwise':
            job = Gaia.launch_job_async("""select source_id, original_ext_source_id, a.ra, a.dec, a.ra_error, a.dec_error, a.radec_co_error, a.w1mpro, a.w1mpro_error, a.w2mpro, a.w2mpro_error, a.w3mpro, a.w3mpro_error, a.w4mpro, a.w4mpro_error, cc_flags, ext_flag, var_flag, ph_qual
            from gaiaedr3.gaia_source as gaia
            join gaiaedr3.allwise_best_neighbour using (source_id)
            join gaiadr1.allwise_original_valid as a using (allwise_oid)
            where
            1 = contains(
                point('', 166.28875, -58.85),
                circle('', gaia.ra, gaia.dec, 0.5)
            )""",
            dump_to_file=False, output_format='csv')

        r = job.get_results()
        df_MW = r.to_pandas()

        # rename columns to vizier names
        if cat=='gaiadist':
            df_MW = df_MW.rename(columns={
                #'original_ext_source_id': 'gaiadist_id',
                'r_med_geo':'rgeo',
                'r_med_photogeo':'rpgeo', 
                'r_lo_geo':'b_rgeo',
                'r_hi_geo':'B_rgeo',
                'r_lo_photogeo':'b_rpgeo',
                'r_hi_photogeo':'B_rpgeo'})
            # calculate 1 sigma errors based on 84 and 16 percentile distance values
            #df_MW['e_rgeo'] = (df_MW['B_rgeo'] - df_MW['b_rgeo'])/2
            #df_MW['e_rpgeo'] = (df_MW['B_rpgeo'] - df_MW['b_rpgeo'])/2

            for dist in ['rgeo', 'rpgeo']:
                df_MW['e_'+dist+'_hi'] = df_MW['B_'+dist] - df_MW[dist]
                df_MW['e_'+dist+'_lo'] = df_MW[dist] - df_MW['b_'+dist]
                df_MW[dist] = df_MW[dist] + np.sqrt(2/np.pi) * (df_MW['e_'+dist+'_hi'] - df_MW['e_'+dist+'_lo'])
                df_MW['e_'+dist] = np.sqrt((1.- 2./np.pi)* (df_MW['e_'+dist+'_hi'] - df_MW['e_'+dist+'_lo'])**2 + df_MW['e_'+dist+'_hi']*df_MW['e_'+dist+'_lo'])
                df_MW = df_MW.drop(['e_'+dist+'_hi','e_'+dist+'_lo'], axis=1)
   

        if cat=='2mass':
            df_MW = df_MW.rename(columns={
                'original_ext_source_id': '_2MASS',
                'ra':'RAJ2000',
                'dec':'DEJ2000',
                'j_m':'Jmag',
                'h_m':'Hmag',
                'ks_m':'Kmag',
                'j_msigcom':'e_Jmag',
                'h_msigcom':'e_Hmag',
                'ks_msigcom':'e_Kmag'})

        if cat=='allwise':
            df_MW = df_MW.rename(columns={
                'original_ext_source_id':'AllWISE',
                'ra':'RAJ2000',
                'dec':'DEJ2000',
                'w1mpro':'W1mag',
                'w2mpro':'W2mag',
                'w3mpro':'W3mag',
                'w4mpro':'W4mag',
                'w1mpro_error':'e_W1mag',
                'w2mpro_error':'e_W2mag',
                'w3mpro_error':'e_W3mag',
                'w4mpro_error':'e_W4mag'})

        #df_gaia = pd.read_csv(f'{file_dir}/{field_name}_gaia.csv')
        print(df_MW.shape)
        #print(df_gaia.dtypes)
        #print(df_gaia['Source_gaia'][:3])
        print(len(df_gaia))
        df_gaia = df_gaia[df_gaia['EDR3Name_gaia'].notna()]
        print(len(df_gaia))
        df_gaia['Source_gaia'] = df_gaia.apply(lambda row: np.int64(row.EDR3Name_gaia[10:]), axis=1)
        df_MW = df_MW.add_suffix('_'+cat)
        #df_gaia['Source_gaia'] = df_gaia.apply(lambda row: np.nan if pd.isnull(row.EDR3Name_gaia) else np.int64(row.EDR3Name_gaia[10:]), axis=1)
        df_MW = df_MW.merge(df_gaia[['Source_gaia', '_q','cp_flag_gaia']], how="inner", left_on="source_id_"+cat, right_on="Source_gaia")
        #print(df_MW)
        print(df_MW.shape)
        
        df_MW = df_MW.rename(columns={'cp_flag_gaia':'cp_flag_'+cat})
        df_MW['Source_gaia'] = df_MW['Source_gaia'].apply(str)
        df_MW['source_id_'+cat] = df_MW['source_id_'+cat].apply(str)


        df_MW.to_csv(f'{file_dir}/{field_name}_'+cat+'_gaia.csv', index=False)
        
        #return None

        '''
        # for CXO sources without gaia counterparts, run crossmatching on allwise and 2mass based on CXO Sources
        if cat=='2mass' or cat=='allwise':
            df_MW_no_gaia = df.merge(df_gaia['_q'], how="outer", on='_q', indicator=True).loc[lambda x : x['_merge']=='left_only']

            df_MW_no_gaia = MW_counterpart_confusion(df_MW_no_gaia['ra'].values, df_MW_no_gaia['dec'].values, search_radius, Es=df_MW_no_gaia['err_ellipse_r0'].values, N=sig_nr, catalog=cat,ref_mjd=ref_mjd,pm_cor=pm_cor,confusion=confusion)
            df_MW = pd.concat([df_MW, df_MW_no_gaia])
        '''

def Gaia_counterparts_update(df, file_dir, field_name):

    df = df.fillna(exnum)
    df_gaia = df.loc[df['Source_gaia']!=exnum, ['Source_gaia','_q']].copy()
    print(len(df_gaia))
    df_MWs = pd.DataFrame()

    for cat, id_col, MW_feature in zip(['2mass','allwise'],['_2MASS_2mass','AllWISE_allwise'], [['Jmag', 'Hmag', 'Kmag','e_Jmag', 'e_Hmag', 'e_Kmag','_2MASS_2mass'], ['W1mag_allwise','W2mag_allwise','W3mag_allwise','W4mag_allwise','e_W1mag_allwise','e_W2mag_allwise','e_W3mag_allwise','e_W4mag_allwise','AllWISE_allwise']]):


        df_MW = pd.read_csv(f'{file_dir}/{field_name}_'+cat+'_gaia.csv')
        if cat =='gaiadist':
            df_MW = df_MW.rename(columns={'Source_gaia':'Source_gaiadist'})
        
        #df_MW['_q'] = df_MW['_q'].astype('int')

        df_gaia = pd.merge(df_gaia, df_MW[MW_feature+['_q']], how='outer', on=['_q', '_q'])
        
        
        print(len(df_MW))
        
        #print(df_MW[MW_feature].head(30))
        #print(df_gaia[MW_feature].head(30))
    #df_MW = df_MW.replace(r'^\s*$', exnum, regex=True)
    #df_MW = df_MW.replace(np.nan, exnum)
    #df_MW = df_MW.fillna(exnum) 
    df_gaia = df_gaia.fillna(exnum)
    #df_gaia.to_csv('update.csv',index=False)
    df.set_index('_q', inplace=True)
    df_gaia.set_index('_q', inplace=True)
    
    df.update(df_gaia) 
    df = df.reset_index() 
    df = df.replace(exnum, np.nan)
    
    return df

def vphasp_to_gaia_mags(df_mw):

    # index of sources with all of these three vphas magnitudes
    idx = df_mw.loc[(df_mw['gmag_vphas'].notna()) & (df_mw['rmag_vphas'].notna()) & (df_mw['imag_vphas'].notna())].index

    # convert each vphas mag to gaia mag
    Gmag_vphas = 0.16973702*unumpy.uarray(df_mw.loc[idx, 'gmag_vphas'], df_mw.loc[idx, 'e_gmag_vphas']) \
    + 0.13965751*unumpy.uarray(df_mw.loc[idx, 'rmag_vphas'], df_mw.loc[idx, 'e_rmag_vphas']) \
    + 0.71393298*unumpy.uarray(df_mw.loc[idx, 'imag_vphas'], df_mw.loc[idx, 'e_imag_vphas']) \

    RPmag_vphas = -0.04373623*unumpy.uarray(df_mw.loc[idx, 'rmag_vphas'], df_mw.loc[idx, 'e_rmag_vphas']) \
    + 1.04600377*unumpy.uarray(df_mw.loc[idx, 'imag_vphas'], df_mw.loc[idx, 'e_imag_vphas'])

    BPmag_vphas = 0.44898906*unumpy.uarray(df_mw.loc[idx, 'gmag_vphas'], df_mw.loc[idx, 'e_gmag_vphas']) \
    + 0.55972776*unumpy.uarray(df_mw.loc[idx, 'rmag_vphas'], df_mw.loc[idx, 'e_rmag_vphas']) \

    df_mw['Gmag_vphas']=pd.Series(unumpy.nominal_values(Gmag_vphas), idx)
    df_mw['RPmag_vphas']=pd.Series(unumpy.nominal_values(RPmag_vphas), idx)
    df_mw['BPmag_vphas']=pd.Series(unumpy.nominal_values(BPmag_vphas), idx)

    #df_mw['Gmag_comb']=df_mw['Gmag'].fillna(df_mw['Gmag_vphas'])
    #df_mw['RPmag_comb']=df_mw['RPmag'].fillna(df_mw['RPmag_vphas'])
    #df_mw['BPmag_comb']=df_mw['BPmag'].fillna(df_mw['BPmag_vphas'])

    # df_mw['e_Gmag_vphas']=pd.Series(unumpy.std_devs(Gmag_vphas), idx)
    # df_mw['e_RPmag_vphas']=pd.Series(unumpy.std_devs(RPmag_vphas), idx)
    # df_mw['e_BPmag_vphas']=pd.Series(unumpy.std_devs(BPmag_vphas), idx)

    # add systematic uncertainties obtained from fitting standard deviation of the difference between vphas converted mags and gaia mags as a function of gaia mags. Gmag fitted with quadratic function from vphas converted Gmag>=16, RPmag and BPmag fitted with linear functions
    df_mw['e_Gmag_vphas']=pd.Series(unumpy.std_devs(Gmag_vphas), idx)

    df_mw.loc[df_mw['Gmag_vphas']>=16, 'e_Gmag_vphas'] = df_mw.loc[df_mw['Gmag_vphas']>=16, 'e_Gmag_vphas'] + 0.00768757*df_mw.loc[df_mw['Gmag_vphas']>=16, 'Gmag_vphas']**2-0.2657567*df_mw.loc[df_mw['Gmag_vphas']>=16, 'Gmag_vphas']+2.33631045

    df_mw['e_RPmag_vphas']=pd.Series(unumpy.std_devs(RPmag_vphas), idx) + 0.02083715*df_mw['RPmag_vphas']-0.18252657
    df_mw['e_BPmag_vphas']=pd.Series(unumpy.std_devs(BPmag_vphas), idx) + 0.03513391*df_mw['BPmag_vphas']-0.42604082

    print(df_mw['e_Gmag_vphas'].dropna().shape, df_mw['e_RPmag_vphas'].dropna().shape, df_mw['e_BPmag_vphas'].dropna().shape)

    
    #df_mw['e_Gmag_comb']=df_mw['e_Gmag'].fillna(df_mw['e_Gmag_vphas'])
    #df_mw['e_RPmag_comb']=df_mw['e_RPmag'].fillna(df_mw['e_RPmag_vphas'])
    #df_mw['e_BPmag_comb']=df_mw['e_BPmag'].fillna(df_mw['e_BPmag_vphas'])
    

    return df_mw

def cal_PU(df, theta, N_counts, PU_name, ver='kim95', sigma=2.):

    df[PU_name] = np.nan

    if ver == 'kim95':

        s1 = np.where((df[N_counts] <= 10**2.1393) & (df[N_counts]> 1))[0]
        df.loc[s1, PU_name] =  10.**(0.1145 * df.loc[s1, theta]-0.4958*np.log10( df.loc[s1, N_counts])+0.1932) *sigma/1.96

        s2 = np.where((df[N_counts] > 10**2.1393) & (df[N_counts] <= 10**3.30))[0]
        df.loc[s2, PU_name] =  10.**(0.0968 * df.loc[s2, theta]-0.2064*np.log10(df.loc[s2, N_counts])-0.4260) *sigma/1.96

    elif ver == 'kim68':

        s1 = np.where((df[N_counts] <= 10**2.1227) & (df[N_counts]> 1))[0]
        df.loc[s1, PU_name] =  10.**(0.1137 * df.loc[s1, theta]-0.460*np.log10( df.loc[s1, N_counts])-0.2398)*sigma

        s2 = np.where((df[N_counts] > 10**2.1227) & (df[N_counts] <= 10**3.30))[0]
        df.loc[s2, PU_name] =  10.**(0.1031 * df.loc[s2, theta]-0.1945*np.log10(df.loc[s2, N_counts])-0.8034)*sigma

    elif ver == 'csc90':

        df[PU_name] =  10.**(0.173 * df[theta]-0.526*np.log10(df[N_counts])-0.023* df[theta]*np.log10(df[N_counts])-0.031) * sigma/1.645

    return df

def refsrc_gaia(field_name, field_dir, ref_mjd, ra=167.8665, dec=-60.66655, R=12., exclude_center=False, Plx_limits=[-2.,2.], e_Plx_limit=1., e_PM_limit=1., PU_limits=1., PM_limit=False, RUWE_limit=False):
    '''
    description:
        prepare the Gaia astrometric reference catalog
    
    inputs:
        field_name: field name
        field_dir: directory to save field data
        ra, dec: center position of the field
        R: the size of the field used to extract Gaia sources
        exclude_center: False if no center area will be excluded, otherwise it will be set to the radius of the center region in arcmin that will be excluded 
        PM_limits: proper motion cut in mas/yr when filtering the astrometric reference sources from gaia 
        PU_limits: positional error cut in mas 

    '''
    gaia_ref_mjd = 57388.
    delta_yr = (ref_mjd - gaia_ref_mjd)/365.
    #Path(field_dir).mkdir(parents=True, exist_ok=True)
    #os.chdir(field_dir)
    Path(f'{field_dir}/Astrometry/').mkdir(parents=True, exist_ok=True)
    #os.chdir(field_dir+'/Astrometry/')
    if path.exists(field_name+'_Gaia_eDR3_clean.txt') == False:
        gaia_cat = 'I/350/gaiaedr3'
        viz = Vizier(row_limit=-1,  timeout=1000, columns=["**", "+_r"], column_filters={'Gmag':'<23.', 'e_RA_ICRS':'<1.', 'e_DE_ICRS':'<1.'}, catalog=gaia_cat)
        c = coordinates.SkyCoord(ra, dec, unit=('deg', 'deg'), frame='icrs')
        res = viz.query_region(c, radius=R*u.arcmin)[0]
        df = res.to_pandas()
        df['e_PM'] = np.sqrt(df['e_pmRA']**2+df['e_pmDE']**2)
        df.to_csv(f'{field_dir}/Astrometry/{field_name}_Gaia_eDR3.csv', index=False)
        epsi_90 = df['epsi'].quantile(0.9)
        print('The 90% percentile of epsi for', field_name,' field is ', epsi_90,'.')
        df_sub = df.loc[(df.Plx>Plx_limits[0]) & (df.Plx<Plx_limits[1]) & (df.e_Plx<e_Plx_limit) & (df.e_PM < e_PM_limit) & (df.epsi<epsi_90)]
        #df_sub = df.loc[(df.Plx>Plx_limits[0]) & (df.Plx<Plx_limits[1]) & (df.PM < PM_limits) & (df.e_RA_ICRS < PU_limits) & (df.e_DE_ICRS < PU_limits)]
        if PM_limit:
            df_sub = df_sub.loc[df.PM < PM_limit]
        if RUWE_limit:
            df_sub = df_sub.loc[df.RUWE < RUWE_limit]
        if exclude_center:
            df_sub = df_sub.loc[df._r > exclude_center]
        df_sub['RA_ERR'] = df_sub['e_RA_ICRS']/3.6e6
        df_sub['DEC_ERR'] = df_sub['e_DE_ICRS']/3.6e6
        df_sub['RA'] = df_sub.apply(lambda row:row.RA_ICRS+delta_yr*row.pmRA/(np.cos(row.DE_ICRS*np.pi/180.)*3.6e6),axis=1)
        df_sub['DEC'] = df_sub.apply(lambda row:row.DE_ICRS+delta_yr*row.pmDE/3.6e6,axis=1)
        df_sub.loc[df_sub['RA'].isnull(),'RA'] = df_sub['RA_ICRS']
        df_sub.loc[df_sub['DEC'].isnull(),'DEC'] = df_sub['DE_ICRS']
        df_save = df_sub[['RA','RA_ERR','DEC','DEC_ERR']]
        df_save = df_save.rename(columns={'RA':'#RA'})
        #df_save = df_save.rename(columns={'RA_ICRS':'#RA','e_RA_ICRS':'RA_ERR','DE_ICRS':'DEC','e_DE_ICRS':'DEC_ERR'})
        df_save.to_csv(f'{field_dir}/Astrometry/{field_name}_Gaia_eDR3_clean.txt',header=True,index=None, sep='\t')

    return df_save

def cal_astrometric_correction(xfm):#field_name,data_dir,residlim,sig_astro,count_astro):

    #xfm = fits.open(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_xfm.fits')
    #logfile=data_dir+'/Astrometry/'+field_name+'_'+str(residlim)+'_'+str(sig_astro)+'_'+str(count_astro)+'_wcs_Xmatch.log')
    
    trs_pars = xfm[1].data[0]
    t1, t2, ra0, dec0, roll_ref, xscale, yscale = trs_pars[4], trs_pars[5], trs_pars[6], trs_pars[7], trs_pars[8], trs_pars[11], trs_pars[12]
    print(t1, t2)
    x = xscale*t1*np.pi/180
    y = yscale*t2*np.pi/180
    phi = math.atan(x/-y)
    theta = math.atan(1./np.sqrt(x**2+y**2))

    ra = ra0+ math.atan((-math.cos(theta)*math.sin(phi))/(math.sin(theta)*math.cos(dec0*np.pi/180.)-math.cos(theta)*math.sin(dec0*np.pi/180.)*math.cos(phi)))*180/np.pi
    #ra0_new = 2*ra0-ra
    #print("ra:", ra, " true ra:",(ra1-(2*ra0-ra)))
    dec = np.arcsin(math.sin(theta)*math.sin(dec0*np.pi/180.)+math.cos(theta)*math.cos(dec0*np.pi/180.)*math.cos(phi))*180/np.pi
    #dec0_new = 2*dec0-dec
    #print("dec:", dec, " true dec:",dec1-(2*dec0-dec))
    if t1<0 and t2<0:
        del_ra, del_dec = ra0-ra, dec0-dec
    else:
        del_ra, del_dec = ra-ra0, dec-dec0
    print('The correction is {:.2f}" in ra and {:.2f}" in dec.'.format(del_ra*3.6e3, del_dec*3.6e3))
    return del_ra, del_dec

def alignment_uncertainty(coords, df_X, df_ref):
    
    df_X = df_X.iloc[coords['Dup'].values].reset_index(drop=True)
    #print(coords['Ref'].values)
    df_ref = df_ref.iloc[coords['#Ref'].values].reset_index(drop=True)

    ra_astro_pu = np.sum(1./(df_X['RA_ERR']*3600)**2+(df_ref['RA_ERR']*3600)**2)**(-0.5)
    dec_astro_pu = np.sum(1./(df_X['DEC_ERR']*3600)**2+(df_ref['DEC_ERR']*3600)**2)**(-0.5)
    #print(ra_astro_pu, dec_astro_pu)
    astro_pu = (ra_astro_pu +dec_astro_pu)/2.
    
    return astro_pu


def cal_astro_pu(field_name,data_dir,residlim,sig_astro,count_astro):

    #os.chdir(field_dir)
    lines = open(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_wcs_Xmatch.log').readlines()
    num_line = len(lines)
    line = lines[8]
    num_src = [int(s) for s in line.split() if s.isdigit()][0]
    outF = open(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_src.csv', "w")
    outF.write('#Ref Dup RA Dec prior_res transfm_res res_ratio')
    outF.write('\n')
    for li in lines[num_line-13-num_src:num_line-13]:
        if 'Y' in li:
            outF.write(li[:-4])
            outF.write('\n')
    outF.close()

    outF = open(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_residual.csv', "w")
    outF.write('before after percentage ')
    outF.write('\n')
    for li in lines[num_line-8:num_line-6]:
            outF.write(li[30:])
            #outF.write('\n')
    outF.close()

    res = pd.read_csv(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_residual.csv', header=0,sep="\s+")
    rms_res = res.loc[0, 'after']
    
    df_src = pd.read_csv(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_src.csv', header=0,sep="\s+")

    df_X = pd.read_csv(f'{data_dir}/Astrometry/{field_name}_bright.txt', header=0,sep="\s+")
    
    df_ref = pd.read_csv(f'{data_dir}/Astrometry/{field_name}_Gaia_eDR3_clean.txt', header=0,sep="\s+")


    astro_pu = alignment_uncertainty(df_src, df_X, df_ref)

    return astro_pu, rms_res, len(df_src) 

def apply_astro_correct(field_name,data_dir,del_ra, del_dec, residlim,sig_astro,count_astro, astro_pu):

    df = pd.read_csv(f'{data_dir}/{field_name}_ave.csv')

    df['ra_cor'] = df['ra'] + del_ra
    df['dec_cor'] = df['dec'] + del_dec 

    df['PU_astro_68'] = astro_pu

    df['PU'] = np.sqrt(df['PU_kim95']**2+(astro_pu*2)**2)

    df.to_csv(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_astro_correct.csv', index=False)

    df_X = pd.read_csv(f'{data_dir}/Astrometry/{field_name}_bright.txt', header=0,sep="\s+")

    df_X['ra_cor'] = df_X['#RA'] + del_ra
    df_X['dec_cor'] = df_X['DEC'] + del_dec 

    df_src = pd.read_csv(f'{data_dir}/Astrometry/{field_name}_{residlim}_{sig_astro}_{count_astro}_src.csv', header=0,sep="\s+")
    
    df_ref = pd.read_csv(f'{data_dir}/Astrometry/{field_name}_Gaia_eDR3_clean.txt', header=0,sep="\s+")

    df_X = df_X.iloc[df_src['Dup'].values].reset_index(drop=True)
    df_ref = df_ref.iloc[df_src['#Ref'].values].reset_index(drop=True)

    #astro_pu = alignment_uncertainty(df_src, df_X, df_ref)

    Xcat1 = np.empty((len(df_src), 2), dtype=np.float64)
    Xcat1[:, 0] = df_ref['#RA']
    Xcat1[:, 1] = df_ref['DEC']

    Xcat2 = np.empty((len(df_X), 2), dtype=np.float64)
    Xcat2[:, 0] = df_X['ra_cor']
    Xcat2[:, 1] = df_X['dec_cor']

    # crossmatch catalogs
    max_radius = 1. / 3600  # 1 arcsec
    dist1, ind1 = crossmatch_angular(Xcat1, Xcat2, max_radius)

    print(dist1*3600)

