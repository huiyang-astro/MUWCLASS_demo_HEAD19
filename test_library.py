#!/usr/bin/env python
# coding: utf-8

# version 1.0

from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
from gdpyc import GasMap, DustMap
from astropy.coordinates import SkyCoord
from scipy.interpolate import InterpolatedUnivariateSpline
import extinction
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import json
from random import randint
import matplotlib.pyplot as plt
from collections import Counter

#from cuml import RandomForestClassifier as cuRF

MW_names = {'gaia':  ['Gmag', 'BPmag', 'RPmag'], \
            '2mass': ['Jmag', 'Hmag', 'Kmag'], \
            'wise':  ['W1mag', 'W2mag', 'W3mag'], \
            'glimpse': ['3.6mag','4.5mag','5.8mag','8.0mag']}

gaia_features = ['Gmag','BPmag', 'RPmag']
gaia_limits   = [21.5,   21.5,   21.]
gaia_zeros    = [2.5e-9, 4.08e-9, 1.27e-9]#[3228.75, 3552.01, 2554.95]
gaia_waves    = [4052.97, 2157.50, 2924.44]
twomass_features = ['Jmag','Hmag','Kmag']
twomass_limits   = [18.5,   18.0,  17.0]
twomass_zeros    = [3.13e-10, 1.13e-10, 4.28e-11]#[1594.,  1024., 666.7]
twomass_waves    = [1624.32, 2509.40, 2618.87]
wise_features = ['W1mag','W2mag','W3mag']
wise_limits   = [18.5,   17.5   , 14.5]
wise_zeros    = [8.18e-12, 2.42e-12, 6.52e-14] #[309.54, 171.787, 31.674]
wise_waves    = [6626.42, 10422.66, 55055.71] #[34000., 46000., 120000.]

MW_features = gaia_features + twomass_features + wise_features
MW_limits = gaia_limits + twomass_limits + wise_limits  # limiting magnitudes
MW_zeros  = gaia_zeros + twomass_zeros + wise_zeros    # zero points to convert magnitude to flux in wavelength space
MW_waves  = gaia_waves + twomass_waves + wise_waves    # effective wavelength widths

 
CSC_flux_features = ['ACIS0512Flux','ACIS122Flux','ACIS27Flux','ACIS057Flux']
CSC_HR_features = ['HR_hm','HR_ms','HR_hms']
CSC_var_features = ['var_inter_prob','var_intra_prob']

CSC_features = CSC_flux_features + CSC_HR_features + CSC_var_features

colors = ['G-BP','G-RP','G-J','G-H','G-K','BP-RP','RP-J','J-H','J-K','H-K','W1-W2','W1-W3','W2-W3']

CSC_all_features = CSC_features + MW_features + colors

XMM_all_features = CSC_flux_features + CSC_HR_features + MW_features + colors

Flux_features = CSC_flux_features + MW_features

dist_features_dict = {'nodist': [], 
                    'rgeo': ['rgeo'], 
                    'rpgeo':  ['rpgeo'], 
                    'plx': ['Plx_dist'], 
                    'rgeo_lum': ['rgeo', 'ACIS057_lum', 'Gmag_lum', 'Jmag_lum'], 
                    'rpgeo_lum': ['rpgeo', 'ACIS057_lum', 'Gmag_lum', 'Jmag_lum'], 
                    'plx_lum': ['Plx_dist', 'ACIS057_lum', 'Gmag_lum', 'Jmag_lum']}

CSC_aveflux_prefix = 'flux_aper90_ave_'

exnum =  -9999999. # some extra-large negtive number to replace NULL

class_labels = {'AGN':'AGN','NS':'NS','CV':'CV','HMXB':'HMXB','LMXB':'LMXB','HM-STAR':'HM-STAR','LM-STAR':'LM-STAR','YSO':'YSO'}
#class_labels = {'AGN':'AGN','NS':'NS','NS_BIN':'NS_BIN','CV':'CV','HMXB':'HMXB','LMXB':'LMXB','HM-STAR':'HM-STAR','LM-STAR':'LM-STAR','YSO':'YSO'}
n_classes = 8#9

class_colors = ['blue','orange','red','c','g','purple','magenta','olive', 'Aqua']

MW_cols = {'xray':['name','ra','dec','PU','significance','flux_aper90_ave_s','e_flux_aper90_ave_s','flux_aper90_ave_m','e_flux_aper90_ave_m','flux_aper90_ave_h','e_flux_aper90_ave_h', \
                  'flux_aper90_ave_b','e_flux_aper90_ave_b','kp_prob_b_max','var_inter_prob' ],
           'gaia':['EDR3Name_gaia','RA_pmcor_gaia','DEC_pmcor_gaia','Gmag_gaia','e_Gmag_gaia','BPmag_gaia','e_BPmag_gaia','RPmag_gaia','e_RPmag_gaia','rgeo_gaiadist','b_rgeo_gaiadist','B_rgeo_gaiadist','rpgeo_gaiadist','b_rpgeo_gaiadist','B_rpgeo_gaiadist'], 
           '2mass':['_2MASS_2mass','Jmag_2mass','e_Jmag_2mass','Hmag_2mass','e_Hmag_2mass','Kmag_2mass','e_Kmag_2mass'], 
           'catwise':['Name_catwise','W1mag_catwise','e_W1mag_catwise','W2mag_catwise','e_W2mag_catwise'],
           'unwise':['objID_unwise','W1mag_unwise','e_W1mag_unwise','W2mag_unwise','e_W2mag_unwise'],
           'allwise':['AllWISE_allwise','W1mag_allwise','e_W1mag_allwise','W2mag_allwise','e_W2mag_allwise','W3mag_allwise','e_W3mag_allwise','W4mag_allwise','e_W4mag_allwise'],
           'vphas':['VPHASDR2_vphas','Gmag_vphas','RPmag_vphas','BPmag_vphas','e_Gmag_vphas','e_RPmag_vphas','e_BPmag_vphas'],
           '2mass_gaia':['_2MASS_2mass_gaia','Jmag_2mass_gaia','e_Jmag_2mass_gaia','Hmag_2mass_gaia','e_Hmag_2mass_gaia','Kmag_2mass_gaia','e_Kmag_2mass_gaia'],
           'allwise_gaia':['AllWISE_allwise_gaia','W1mag_allwise_gaia','e_W1mag_allwise_gaia','W2mag_allwise_gaia','e_W2mag_allwise_gaia','W3mag_allwise_gaia','e_W3mag_allwise_gaia','W4mag_allwise_gaia','e_W4mag_allwise_gaia']
           }

########################### Default Scaler  ####################################
from sklearn.preprocessing import StandardScaler
#   default = StandardScaler to remove the mean and scale to unit variance
standscaler = StandardScaler()
ML_scaler = standscaler # the scaler selected

scaler_switch = True # for ML_model = RFmodel

def col_rename(df):

    df = df[df.remove_code==0].reset_index(drop=True)

    df = df.rename(columns = {
        'W1mag_comb':'W1mag','W2mag_comb':'W2mag','W3mag_allwise':'W3mag','W4mag_allwise':'W4mag',
        'e_W1mag_comb':'e_W1mag','e_W2mag_comb':'e_W2mag','e_W3mag_allwise':'e_W3mag','e_W4mag_allwise':'e_W4mag',
        'kp_prob_b_max':'var_intra_prob'})

    for band in ['s','m','h']:
        df['Fcsc_' + band], df['e_Fcsc_' + band] = df['flux_aper90_ave_' + band], df['e_flux_aper90_ave_' + band]

    return df

def prepare_cols(df, cp_thres=0, vphas=False,gaiadata=False,cp_conf_flag=False, TD=False, NS_MWdrop=False, STAR_classremove=['HM-STAR','LM-STAR','YSO']):

    # clean X-ray sources
    #df = df[df.remove_code==0].reset_index(drop=True)

    df = df.rename(columns={
        'W1mproPM_catwise':'W1mag_catwise','W2mproPM_catwise':'W2mag_catwise','e_W1mproPM_catwise':'e_W1mag_catwise','e_W2mproPM_catwise':'e_W2mag_catwise'})
    
    MW_cats = ['gaia','2mass','catwise','unwise','allwise']
    MW_cats.append('vphas') if vphas==True else None 
    MW_cats.extend(['2mass_gaia','allwise_gaia']) if gaiadata==True else None
    #print(MW_cats)

    # clean TD 
    if TD == True:

        s = np.where((df.Class.isin(STAR_classremove)) & (df['cp_flag_gaia']<cp_thres) & (df['cp_flag_2mass']<cp_thres) & (df['cp_flag_catwise']<cp_thres) & (df['cp_flag_unwise']<cp_thres) & (df['cp_flag_allwise']<cp_thres))[0]
        df.loc[s, 'remove_code'] = df.loc[s, 'remove_code']+64
        print('Remove', len(s), sorted(Counter(df.loc[s, 'Class']).items()))

        if NS_MWdrop == True:
            for cat in MW_cats:
                df.loc[df['Class']=='NS', 'cp_flag_'+cat] = df.loc[df['Class']=='NS', 'cp_flag_'+cat] - 8

    # clean X-ray sources
    df = df[df.remove_code==0].reset_index(drop=True)
    if TD:
        print('Final breakdown', len(df), sorted(Counter(df['Class']).items()))

    # clean multiwavelength catalogs 
    for cat in MW_cats:
        df.loc[df['cp_flag_'+cat]<cp_thres, MW_cols[cat]] = np.nan 

    df_save = pd.DataFrame()

    MW_cats.insert(0, 'xray')
    for i, cat in enumerate(MW_cats):
        #print(i, cat)
        if i == 0:
            df_save = df[MW_cols[cat]]
        else:
            #print(MW_cols[cat]+['cp_flag_'+cat])
            df_save = df_save.join(df[MW_cols[cat]+['cp_flag_'+cat]])
    
    if cp_conf_flag == True:
        df_save = df_save.join(df['cp_conf_flag'])
    if TD == True:
        df_save = df_save.join(df['Class'])

    # update from Gaia database 
    if gaiadata==True:
        #print(df_save.loc[df_save['cp_flag_2mass_gaia']>=cp_thres, MW_cols['2mass']+MW_cols['2mass_gaia']])
        #df_save.loc[df_save['cp_flag_2mass_gaia']>=cp_thres, [MW_cols['2mass']+['cp_flag_2mass']]] = df_save.loc[df_save['cp_flag_2mass_gaia']>=cp_thres, MW_cols['2mass_gaia']+['cp_flag_2mass_gaia']]
        #df_save.loc[df_save['cp_flag_allwise_gaia']>=cp_thres, [MW_cols['allwise']+['cp_flag_allwise']]] = df_save.loc[df_save['cp_flag_allwise_gaia']>=cp_thres, MW_cols['allwise_gaia']+['cp_flag_allwise_gaia']]
        #'''
        for col in MW_cols['2mass']+['cp_flag_2mass']:
            df_save.loc[df_save['cp_flag_2mass_gaia']>=cp_thres, col] = df_save.loc[df_save['cp_flag_2mass_gaia']>=cp_thres, col+'_gaia']
        for col in MW_cols['allwise']+['cp_flag_allwise']:
            df_save.loc[df_save['cp_flag_allwise_gaia']>=cp_thres, col] = df_save.loc[df_save['cp_flag_allwise_gaia']>=cp_thres, col+'_gaia']
        #'''

    # combine ALLWISE, CatWISE and UnWISE

    df_save['cp_flag_wise12'],df_save['which_wise12'] = np.nan, np.nan
    df_save['W1mag_wise12'],df_save['e_W1mag_wise12'],df_save['W2mag_wise12'],df_save['e_W2mag_wise12'] =  np.nan, np.nan, np.nan, np.nan
    for wise in ['unwise','catwise','allwise']:
        df_save.loc[df_save['cp_flag_'+wise]>=cp_thres, 'cp_flag_wise12'] = df_save.loc[df_save['cp_flag_'+wise]>=cp_thres, 'cp_flag_'+wise]
        df_save.loc[df_save['cp_flag_'+wise]>=cp_thres, 'which_wise12'] = wise
        for w in ['W1', 'W2']:
            df_save.loc[df_save['cp_flag_'+wise]>=cp_thres, w+'mag_wise12'] = df_save.loc[df_save['cp_flag_allwise']>=cp_thres, w+'mag_'+wise]
            df_save.loc[df_save['cp_flag_'+wise]>=cp_thres, 'e_'+w+'mag_wise12'] = df_save.loc[df_save['cp_flag_allwise']>=cp_thres, 'e_'+w+'mag_'+wise]

    # Gaia and vphas and combine them 
    if vphas == True:
        df_save['which_gaia'], df_save['cp_flag_comb'] = np.nan, np.nan
        df_save.loc[(df_save['cp_flag_gaia']>=cp_thres), 'which_gaia'] = 'gaia'
        df_save.loc[(df_save['cp_flag_gaia']>=cp_thres), 'cp_flag_comb'] = df_save.loc[(df_save['cp_flag_gaia']>=cp_thres), 'cp_flag_gaia']
        df_save.loc[(df_save['cp_flag_gaia']<cp_thres) & (df_save['cp_flag_vphas']>=cp_thres), 'which_gaia'] = 'vphas'
        df_save.loc[(df_save['cp_flag_gaia']<cp_thres) & (df_save['cp_flag_vphas']>=cp_thres), 'cp_flag_comb'] = df_save.loc[(df_save['cp_flag_gaia']<cp_thres) & (df_save['cp_flag_vphas']>=cp_thres), 'cp_flag_vphas']
        for mag in ['Gmag', 'BPmag', 'RPmag']:
            df_save[mag+'_comb'], df_save['e_'+mag+'_comb'] = np.nan, np.nan
            df_save.loc[(df_save['cp_flag_gaia']>=cp_thres), mag+'_comb'] = df_save.loc[(df_save['cp_flag_gaia']>=cp_thres), mag+'_gaia'] 
            df_save.loc[(df_save['cp_flag_gaia']>=cp_thres), 'e_'+mag+'_comb'] = df_save.loc[(df_save['cp_flag_gaia']>=cp_thres), 'e_'+mag+'_gaia']
            df_save.loc[(df_save['cp_flag_gaia']<cp_thres) & (df_save['cp_flag_vphas']>=cp_thres), mag+'_comb'] = df_save.loc[(df_save['cp_flag_gaia']<cp_thres) & (df_save['cp_flag_vphas']>=cp_thres), mag+'_vphas'] 
            df_save.loc[(df_save['cp_flag_gaia']<cp_thres) & (df_save['cp_flag_vphas']>=cp_thres), 'e_'+mag+'_comb'] = df_save.loc[(df_save['cp_flag_gaia']<cp_thres) & (df_save['cp_flag_vphas']>=cp_thres), 'e_'+mag+'_vphas'] 
    
    #'''
    df_save = df_save.rename(columns = {
        'flux_aper90_ave_s':'Fcsc_s','e_flux_aper90_ave_s':'e_Fcsc_s','flux_aper90_ave_m':'Fcsc_m','e_flux_aper90_ave_m':'e_Fcsc_m',
        'flux_aper90_ave_h':'Fcsc_h','e_flux_aper90_ave_h':'e_Fcsc_h',
        'rgeo_gaiadist':'rgeo','b_rgeo_gaiadist':'b_rgeo','B_rgeo_gaiadist':'B_rgeo','rpgeo_gaiadist':'rpgeo','b_rpgeo_gaiadist':'b_rpgeo','B_rpgeo_gaiadist':'B_rpgeo',
        'Jmag_2mass':'Jmag','e_Jmag_2mass':'e_Jmag','Hmag_2mass':'Hmag','e_Hmag_2mass':'e_Hmag','Kmag_2mass':'Kmag','e_Kmag_2mass':'e_Kmag',
        'W1mag_wise12':'W1mag','W2mag_wise12':'W2mag','W3mag_allwise':'W3mag','W4mag_allwise':'W4mag',
        'e_W1mag_wise12':'e_W1mag','e_W2mag_wise12':'e_W2mag','e_W3mag_allwise':'e_W3mag','e_W4mag_allwise':'e_W4mag',
        'kp_prob_b_max':'var_intra_prob'})
    
    if vphas == True:
        df_save = df_save.rename(columns = {
            'Gmag_comb':'Gmag','BPmag_comb':'BPmag','RPmag_comb':'RPmag','e_Gmag_comb':'e_Gmag','e_BPmag_comb':'e_BPmag','e_RPmag_comb':'e_RPmag'
        })
    elif vphas == False:
        df_save = df_save.rename(columns = {
            'Gmag_gaia':'Gmag','BPmag_gaia':'BPmag','RPmag_gaia':'RPmag','e_Gmag_gaia':'e_Gmag','e_BPmag_gaia':'e_BPmag','e_RPmag_gaia':'e_RPmag'
        })

    return df_save

# MC sampling 

def nonzero_sample(df, col, out_col,random_state=None):
    '''
    description: 
        sampling the col column of df with its Gaussian uncertainty e_col column while making sure the sampled value is larger than zero (cases for fluxes)
    
    input:
        df: the dataframe 
        col: the sampled column name (the uncertainty column is e_col by default)
        out_col: output column name
    '''
    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)
    
    df['temp_'+col] = np.random.randn(df[col].size) * df['e_'+col] + df[col]
    s = df.loc[df['temp_'+col]<=0].index

    while len(s) >0:
        df.loc[s,'temp_'+col] = np.random.randn(df.loc[s,col].size) * df.loc[s,'e_'+col] + df.loc[s,col]
        s = df.loc[df['temp_'+col]<=0].index

    df[out_col] = df['temp_'+col]
    df = df.drop(columns='temp_'+col)
    
    return df

def asymmetric_errors(df,dist):
    # calculate the errors for distances based on 84% and 16% percentile values
    
    df['e_B_'+dist] = df['B_'+dist] - df[dist]
    df['e_b_'+dist] = df[dist] - df['b_'+dist] 

    # assume mode is median for this
    df['mean_'+dist] = df[dist] + np.sqrt(2/np.pi) * (df['e_B_'+dist] - df['e_b_'+dist])
    df['e_'+dist] = np.sqrt((1.- 2./np.pi)* (df['e_B_'+dist] - df['e_b_'+dist])**2 + df['e_B_'+dist]*df['e_b_'+dist])
    
    return df

def sample_data(df,Xray='CSC',distance='nodist',Uncer_flag=False,random_state=None,rep_num=False,verb=False):
    '''
    description: create sampled data from (Gaussian) distributions of measurements
    
    input:
        df: the dataframe 
        Xray: 'XMM' or 'CSC' X-ray data set 
        fIR: 'WISE' or 'Glimpse' far-Infrared data 
        Xray_level: 'ave' when averaged fluxes are sampled or 'obs' when per-observation fluxes are sampled
        distance: 
        verb: 
    '''

    if rep_num!=False:
        df = pd.DataFrame(np.repeat(df.values, rep_num, axis=0), columns=df.columns)
    
    if random_state is None:
        np.random.seed(randint(1,999999999)) 
    else:
        np.random.seed(random_state)

    if verb:
        print('Run......sample_data')
        print('Sampling '+ Xray + ' X-ray data.')

    if Uncer_flag == True:
        if Xray == 'XMM':
            # simulate fluxes assuming gaussian distribution of flux for XMM energy bands
            
            bands = ['2','3','4','5','8']
            for band in bands:
                df = nonzero_sample(df, 'Fxmm_'+band, 'Fxmm_'+band,random_state=random_state)

        if Xray == 'CSC':

            bands = ['s','m','h']
            
            for band in bands:
                    
                df = nonzero_sample(df, 'Fcsc_'+band, 'Fcsc_'+band,random_state=random_state)
            
            df['Fcsc_b'] = df['Fcsc_s'] + df['Fcsc_m'] + df['Fcsc_h']

        if verb:
            print('Sampling MW data.')

        MW_cats = ['gaia', '2mass', 'wise']

        for cat in MW_cats:
            for band in MW_names[cat]:
                df[band] = np.random.randn(df[band].size) * df['e_'+band] + df[band]

        if distance !='nodist':
            
            dist_feature = dist_features_dict[distance][0] 

            # set distances to zero for sources without EDR3Name_gaia 

            df.loc[df['EDR3Name_gaia'].isna(), [dist_feature, 'e_'+dist_feature]]=np.nan
            
            # set distance of sources with negative parallaxes and parallaxes with large errors (fpu<2) to nan, already done in making of TD?
            # the cleaning of features should be done when creating the test data
            df.loc[df['Plx_gaia']<0, [dist_feature, 'e_'+dist_feature]]=np.nan
            df.loc[df['Plx_gaia']/df['e_Plx_gaia']<2, [dist_feature, 'e_'+dist_feature]]=np.nan

            df = asymmetric_errors(df, dist_feature)

            if dist_feature == 'Plx_dist':
                df = nonzero_sample(df, 'Plx_gaia', 'Plx_gaia',random_state=random_state)
                df['Plx_dist'] = 1000./df['Plx_gaia'] # Plx_gaia in units of mas
            else:
                # set distance of sources with no parallax measurements to nan. 
                df.loc[df['Plx_gaia'].isna(), [dist_feature, 'e_'+dist_feature]]=np.nan
                df = nonzero_sample(df, dist_feature, dist_feature,random_state=random_state)

    elif Uncer_flag == False:   

        if Xray == 'XMM':
            bands = ['2','3','4','5','8']
            for band in bands:
                df.loc[df['Fxmm_'+band]==0, 'Fxmm_'+band] = 1e-22 # can be implemented when producing the XMM TD
        
        if Xray == 'CSC':
            df['Fcsc_b'] = df['Fcsc_s'] + df['Fcsc_m'] + df['Fcsc_h']
        
        if verb:
            print('Copying MW data where FIR is from ', fIR, ' and distance feature ', distance,'.')
    
        if distance !='nodist':
            
            dist_feature = dist_features_dict[distance][0] 

            df.loc[df['EDR3Name_gaia'].isna(), [dist_feature, 'e_'+dist_feature]]=np.nan 

            # set distance of sources with negative parallaxes and parallaxes with large errors (fpu<2) to nan, already done in making of TD?
            df.loc[df['Plx_gaia']<0, [dist_feature, 'e_'+dist_feature]]=np.nan
            df.loc[df['Plx_gaia']/df['e_Plx_gaia']<2, [dist_feature, 'e_'+dist_feature]]=np.nan

            df = asymmetric_errors(df, dist_feature)
            
            if dist_feature == 'Plx_dist':
                # set distance of sources with no parallax measurements to nan. 
                df['Plx_dist'] = 1000./df['Plx_gaia'] # Plx_gaia in units of mas
            else:
                # set distance of sources with no parallax measurements to nan. 
                df.loc[df['Plx_gaia'].isna(), [dist_feature, 'e_'+dist_feature]]=np.nan

    return df

def convert2csc(data, method = 'simple', Gamma =2.,verb=False):
    # Convert XMM fluxes to CSC fluxes with method='simple' using simple scaling factors assuming Gamma=2
    # or method='LR' with linear regression using paramters from fitting the same sources from XMM and CSC TD
    
    CSC_fluxs, XMM_fluxs = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h'], ['Fxmm_2',   'Fxmm_3',  'Fxmm_4',   'Fxmm_5']    
    CSC_bands, XMM_bands = [[0.5,1.2], [1.2,2.],  [2.,7.]],   [[0.5,1.] ,[1.,2.], [2.,4.5], [4.5,12.]]
    
    simple_coefs = [[(np.log(1.2)-np.log(0.5))/(np.log(1.0)-np.log(0.5))], #[0.5, 1.0] keV -> (0.5, 1.2) keV
                    [(np.log(2.0)-np.log(1.2))/(np.log(2.0)-np.log(1.0))], # [1.0, 2.0] -> (1.2, 2.0) keV
                    [1.0, (np.log(7.0)-np.log(4.5))/(np.log(12.0)-np.log(4.5))]] # [2.0, 4.5] [4.5, 12.0] -> [2.0, 7.0]
    if Gamma !=2.:
        simple_coefs = [[(1.2**(2.-Gamma)-0.5**(2.-Gamma))/(1.**(2.-Gamma)-0.5**(2.-Gamma))],
                        [(2.**(2.-Gamma) -1.2**(2.-Gamma))/(2.**(2.-Gamma)-1.)],
                        [1., (7.**(2.-Gamma)-4.5**(2.-Gamma))/(12.**(2.-Gamma)-4.5**(2.-Gamma))]]
    
    LR_coefs = [[ 0.95141998],
                [ 0.54679595, 0.44769412],
                [ 1.00513222],
                [ 0.66209453, 0.31911049]]
    
    XMMfluxs = [[data['Fxmm_2']],
                [data['Fxmm_3']],
                [data['Fxmm_4'], data['Fxmm_5']]]
    CSCfluxs = ['Fcsc_s','Fcsc_m','Fcsc_h']
       
       
    if method=='simple':
        if verb:
            print("Run convert2csc with simple method and Gamma = "+str(Gamma)+".")
    
        # Simple scaling assuming a flat spectrum (default Gamma=2)
        for col_n, Xflux, coef in zip(CSCfluxs, XMMfluxs, simple_coefs):
            #print("Converbting to", col_n, "with simple method.")
            data[col_n] = sum([ flux*c for (flux, c) in zip(Xflux,np.array(coef))])

            
    if method=='LR': 
        if verb:
            print("Run convert2csc with LR method......")

        # Linear Regression 
        for col_n, Xflux, LR_coef, sim_coef in zip(cols_new, XMMfluxs, LR_coefs, simple_coefs):
            #print("Converbting to", col_n, "with LR method.")
            data[col_n] = np.prod([flux**c for (flux, c) in zip(Xflux,np.array(LR_coef))],axis=0)
            if col_n == 'Fcsc_s_lr2':
                data.loc[(data['xmm_f2'] == 0) | (data['xmm_f3'] == 0),col_n]= \
                    sum([ flux*c for (flux, c) in zip([data.loc[(data['xmm_f2'] == 0) | (data['xmm_f3'] == 0),'xmm_f2'],\
                    data.loc[(data['xmm_f2'] == 0) | (data['xmm_f3'] == 0),'xmm_f3']],np.array(sim_coef))])

            if col_n == 'Fcsc_h_lr2':
                data.loc[(data['xmm_f4'] == 0) | (data['xmm_f5'] == 0),col_n]= \
                    sum([ flux*c for (flux, c) in zip([data.loc[(data['xmm_f4'] == 0) | (data['xmm_f5'] == 0),'xmm_f4'],\
                    data.loc[(data['xmm_f4'] == 0) | (data['xmm_f5'] == 0),'xmm_f5']],np.array(sim_coef))])

    data['Fcsc_b']=data['Fcsc_s']+data['Fcsc_m']+data['Fcsc_h']

    return data                  
   

def get_red_par(ra, dec, dustmap='SFD', nhmap='LAB'):

    coords = SkyCoord(ra, dec, unit='deg')
    ebv = DustMap.ebv(coords, dustmap=dustmap) * 0.86 # 0.86 is the correction described in Schlafly et al. 2010 and Schlafly & Finkbeiner 2011
    nH_from_AV = 2.21 * 3.1 * ebv
    nH  = GasMap.nh(coords, nhmap=nhmap).value / 1.e21 # nH in unit of 1.e21 atoms /cm2
    
    return ebv, nH_from_AV 

def red_factor(ene, nH, Gamma, tbabs_ene, tbabs_cross):

    if Gamma == 2:
        flux_unred_int = np.log(ene[1]) - np.log(ene[0])
    else:
        flux_unred_int   = (ene[1]**(2.-Gamma)-ene[0]**(2.-Gamma))/(2.-Gamma)
            
    _ = np.array([_**(1 - Gamma) for _ in tbabs_ene])            
    tbabs_flux_red = _ * np.exp(-nH * 1e-3 * tbabs_cross)
    
    finterp = InterpolatedUnivariateSpline(tbabs_ene, tbabs_flux_red, k=1)
    
    flux_red_int = finterp.integral(*ene)
        
    return flux_red_int / flux_unred_int

def apply_red2csc(data, nh, tbabs_ene, tbabs_cross, red_class='AGN', self_unred=False, Gamma=2):
    bands = ['Fcsc_s', 'Fcsc_m', 'Fcsc_h', 'Fcsc_b']
    enes  = [[0.5,1.2], [1.2,2.0], [2.0,7.0], [0.5, 7.0]]
    for ene, band in zip(enes, bands):
        red_fact = red_factor(ene, nh, Gamma, tbabs_ene, tbabs_cross)
        if self_unred == True:
            for idx in data.loc[data['Class'] == red_class].index.tolist():
                data[band][idx] = data[band][idx] * red_factor(ene, nh - data['nH'][idx], Gamma, tbabs_ene, tbabs_cross)
        if self_unred == False:
            data.loc[data.Class == red_class, band] = data[band]*red_fact
    return data

def apply_red2mw(data, ebv, red_class='AGN', self_unred = False):
    # extinction.fitzpatrick99 https://extinction.readthedocs.io/en/latest/
    ### wavelengths of B, R, I (in USNO-B1), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms
    # wavelengths of G, Gbp, Grp (in Gaia), J, H, K (in 2MASS), W1, W2, W3 (in WISE) bands in Angstroms
    
    waves = gaia_waves + twomass_waves + wise_waves 
    bands = gaia_features + twomass_features + wise_features
       
    for wave, band in zip(waves, bands):
        if self_unred == True:
            for idx in data.loc[data['Class'] == red_class].index.tolist():
                data[band][idx] = data[band][idx] + extinction.fitzpatrick99(np.array([wave]), 3.1*(ebv-data['ebv'][idx]))
        if self_unred == False:
            data.loc[data.Class == red_class, band] = data.loc[data.Class == red_class, band] + extinction.fitzpatrick99(np.array([wave]), 3.1*ebv)
    return data

def mw2limit(data):

    # remove any magnitudes that are larger than the limiting magnitudes
    #bands = ['Gmag','BPmag','RPmag','Jmag', 'Hmag', 'Kmag', 'W1mag','W2mag']#,'W3mag']
    #limits =[ 21.5,  21.5,  21.0,  18.5,  18.0,   17.0,   18.5,   17.5]   #, 14.5]
    for band, limit in zip(MW_features, MW_limits):
        data.loc[ data[band] >= limit, band] = np.nan
    return data

def create_colors(data, apply_limit=True):
    
    if apply_limit:
        # data = mw2limit(data, verb=verb)
        for band, limit in zip(MW_features, MW_limits):
            data.loc[data[band]>=limit, band] = np.nan

    MW_features2 = MW_features.copy()
    for col1 in MW_features:
        MW_features2.remove(col1)
        for col2 in MW_features2:
            color = col1[:-3] + "-" + col2[:-3]
            data[color] = data[col1] - data[col2]

    return data

def create_Xfeatures(data):
    '''
    create X-ray features including EP052Flux, EP127Flux, EP057Flux, in erg/cm^2/s
    and hardness ratios EPHR4, EPHR2
    '''
    
    data['ACIS0512Flux'] = data['Fcsc_s']
    data['ACIS122Flux'] = data['Fcsc_m']
    data['ACIS27Flux'] = data['Fcsc_h']
    data['ACIS052Flux'] = data['Fcsc_s'] + data['Fcsc_m']
    data['ACIS127Flux'] = data['Fcsc_m'] + data['Fcsc_h']
    data['ACIS057Flux'] = data['Fcsc_s'] + data['Fcsc_m'] + data['Fcsc_h']

    '''
    data.loc[data.ACIS052Flux <=10**(-19.),'ACIS052Flux']= 10**(-19.)
    data.loc[data.ACIS127Flux  <=10**(-21.),'ACIS127Flux']= 10**(-21.)
    data.loc[data.ACIS057Flux <=10**(-20.),'ACIS057Flux']= 10**(-20.)
    '''

    data['HR_ms'] = (data['Fcsc_m'] - data['Fcsc_s'])/data['ACIS052Flux']
    data['HR_hm'] = (data['Fcsc_h'] - data['Fcsc_m'])/data['ACIS127Flux']
    data['HR_hms'] = (data['Fcsc_h'] - data['Fcsc_m']-data['Fcsc_s'])/data['ACIS057Flux']

    return data

def mag2flux(data):
    '''
    from magnitude to flux in erg/cm^s/s
    '''

    for band, zero, wave in zip(MW_features, MW_zeros, MW_waves):
        data[band] = zero * 10**(-data[band]/2.5) * wave

    return data

def luminosity(data, distance, verb=False):
    '''
    define luminosities for ACIS057, Gmag, and Jmag based on distance
    '''
    if verb:
        print("Adding Luminosities based on " +distance)
    
    if distance == 'rgeo_lum' or distance == 'rpgeo_lum' or distance == 'plx_lum':
        data['ACIS057_lum'] = np.log10((data['ACIS057Flux']*4*np.pi*np.power(data[dist_features_dict[distance][0]], 2)).astype('float64'))
        data['Gmag_lum'] = np.log10((data['Gmag']*4*np.pi*np.power(data[dist_features_dict[distance][0]], 2)).astype('float64'))
        data['Jmag_lum'] = np.log10((data['Jmag']*4*np.pi*np.power(data[dist_features_dict[distance][0]], 2)).astype('float64'))

    #if distance == 'plx_lum':
        #data['ACIS057_lum'] = (data['ACIS057Flux']*4*np.pi*np.power(1/data[dist_features[0]], 2)).astype('float64')
        #data['Gmag_lum'] = (data['Gmag']*4*np.pi*np.power(1/data[dist_features[0]], 2)).astype('float64')
        #data['Jmag_lum'] = (data['Jmag']*4*np.pi*np.power(1/data[dist_features[0]], 2)).astype('float64')

    return data

def standardize_log(data, by='ACIS057Flux'):
    # standardizing data by dividing all flux features (except by feature-EP057Flux) by 'by'-broad band flux to mitigate the impact of their unknown distances

    cols = Flux_features.copy()
    cols.remove(by)
    for col in cols:
        data[col] = np.log10((data[col]/data[by]).astype(float))
    data[by] = np.log10(data[by].astype(float))
    
    return data

def postprocessing(data, 
                   Xcat='CSC', 
                   distance='nodist', 
                   add_cols=['Class','name']):
    '''
    description:
        postprocess the data to be fed into classifier

    input:
        data: the input DataFrame
        Xcat: 'CSC' or 'XMM' based
        distance: 
        add_cols: columns to be added besides the features used to be trained
        
    output: the DataFrame after post-processing

    '''
    
    apply_limit = True # apply the magnitude limit cuts if apply_limit=True, otherwise not

    # Create colors from magnitudes and apply magnitude limit cut if apply_limit=True
    data_colors = create_colors(data, apply_limit=apply_limit)

    # Create X-ray features defined in X_features
    data_Xfeatures = create_Xfeatures(data_colors)

    # Convert MW magnitudes to flux in erg/s/cm^2
    data_mw2flux = mag2flux(data_Xfeatures)
    
    if distance != 'nodist':
        data_mw2flux = luminosity(data_mw2flux, distance)
        
    standidize_by = 'ACIS057Flux' # all flux features are divided by broad band X-ray flux for standardization except for Fb    

    # Standardizing data by dividing flux features (except EP057Flux) by EP057Flux
    data_stand = standardize_log(data_mw2flux, by=standidize_by)

    if Xcat=='CSC':
        datmod = data_stand[CSC_all_features+dist_features_dict[distance]+add_cols]
    if Xcat=='XMM':
        datmod = data_stand[XMM_all_features+add_cols]

    return datmod

def oversampling(method, X_train, y_train):
    # oversampling training dataset to mitigate for the imbalanced TD
    # default = SMOTE

    X_train.replace(np.nan, exnum, inplace=True)
    X_res, y_res = method.fit_resample(X_train, y_train)
    res = X_res.values

    X_train[X_train == exnum] = np.nan
    X_train_min = np.nanmin(X_train, axis=0)

    for i in np.arange(len(res[:,0])):
        for j in np.arange(len(res[0,:])):
            if res[i,j] <= X_train_min[j]:
                res[i,j] = np.nan
                
    X_res[:] = res
    return X_res, y_res

def scaling(scaler, X_train, unscales,verb=False):
    # apply scaler on training set and other data
    # default = StandardScaler to remove the mean and scale to unit variance
    if verb:
        print("Run scaling......")
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    if verb:
        print("Train DS transformed shape: {}".format(X_train_scaled.shape))

    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

    scaleds = []
    for un_scale in unscales:
        scaled = scaler.transform(un_scale)
        scaled_df = pd.DataFrame(scaled, columns=X_train.columns)
        scaleds.append(scaled_df)

        if verb:
            print("Transformed shape: {}".format(un_scale.shape))

    return X_train_scaled_df, scaleds

def loo_prepare(i, df, red_switch, Xcat, distance, Uncer_flag, ran_feature, random_state_sample, random_state_smote, tbabs_ene, tbabs_cross):   
    
    df_test = df[df.name == df.name[i]]
    df_train = df[df.name != df.name[i]]

    field_ra = df_test['ra'].values
    field_dec = df_test['dec'].values
    
    df_test  = sample_data(df_test,Xcat,distance,Uncer_flag,random_state_sample,rep_num=False)
    df_train = sample_data(df_train,Xcat,distance,Uncer_flag,random_state_sample)

    if Xcat == 'XMM':
        df_test = convert2csc(df_test, method = 'simple', Gamma =2.)
        df_train = convert2csc(df_train, method = 'simple', Gamma =2.)
 

    '''
    df = sample_data(df,Xcat,distance,Uncer_flag,random_state_sample)

    df_test = df[df.name == df.name[i]]
    df_train = df[df.name != df.name[i]]
    '''

    if red_switch:

        # Extract reddening parameters from SFD dustmap & DL HI map
        ebv, nh = get_red_par(field_ra, field_dec)

        # Applying reddening to AGNs     
        data_red2csc = apply_red2csc(df_train.copy(), nh, tbabs_ene, tbabs_cross, 'AGN', self_unred=False, Gamma=2)
        df_train  = apply_red2mw(data_red2csc, ebv, 'AGN', self_unred=False)

    df_train = postprocessing(df_train, Xcat, distance, add_cols=['Class','name'])
    df_test = postprocessing(df_test, Xcat, distance, add_cols=['Class','name'])

    scaler_switch = False # for ML_model = RFmodel

    X_train, y_train = df_train.drop(['Class', 'name'], axis=1), df_train.Class
    X_test, y_test, test_name = df_test.drop(['Class', 'name'], axis=1), df_test.Class, df_test.name

    if scaler_switch==True:
        X_train, [X_test] = scaling(ML_scaler, X_train, [X_test])
        
    ML_oversampler = SMOTE(random_state=random_state_smote, k_neighbors=4, n_jobs=-1) 
        
    X_train, y_train = oversampling(ML_oversampler, X_train, y_train)
    
    X_train = X_train.fillna(-100)
    X_test  = X_test.fillna(-100)

    if ran_feature=='normal':
        X_train['ran_fea'] = np.random.randn(X_train.shape[0]) 
        X_test['ran_fea'] = np.random.randn(X_test.shape[0]) 
    elif ran_feature=='uniform':
        X_train['ran_fea'] = np.random.rand(X_train.shape[0]) 
        X_test['ran_fea'] = np.random.rand(X_test.shape[0]) 
    
    return [i, X_train, y_train, X_test, y_test, test_name]

def class_prepare(TD, field, red_switch, field_ra, field_dec, Xcat, distance, Uncer_flag, random_state_sample, random_state_smote, tbabs_ene, tbabs_cross):   
    
    TD = sample_data(TD,Xcat,distance,Uncer_flag,random_state_sample)
    field = sample_data(field,Xcat,distance,Uncer_flag,random_state_sample)

    if red_switch:

        # Extract reddening parameters from SFD dustmap & DL HI map
        ebv, nh = get_red_par(field_ra, field_dec)

        # Applying reddening to AGNs     
        TD_red2csc = apply_red2csc(TD.copy(), nh, tbabs_ene, tbabs_cross, 'AGN', self_unred=False, Gamma=2)
        TD  = apply_red2mw(TD_red2csc, ebv, 'AGN', self_unred=False)

    TD = postprocessing(TD, Xcat, distance, add_cols=['Class','name'])
    field = postprocessing(field, Xcat, distance, add_cols=['name'])

    scaler_switch = False # for ML_model = RFmodel

    X_train, y_train = TD.drop(['Class', 'name'], axis=1), TD.Class
    X_test, test_name = field.drop('name', axis=1), field.name

    if scaler_switch==True:
        X_train, [X_test] = scaling(ML_scaler, X_train, [X_test])
        
    ML_oversampler = SMOTE(random_state=random_state_smote, k_neighbors=4, n_jobs=-1) 
        
    X_train, y_train = oversampling(ML_oversampler, X_train, y_train)
    
    X_train = X_train.fillna(-100)
    X_test  = X_test.fillna(-100)
    
    return [X_train, y_train, X_test, test_name]

def get_classification_path(clf, X_test, sample_id=0, verb=True):
    '''
    processes X_test.iloc[sample_id] 
    '''

    feature_names = clf.feature_names_in_  
    classes = clf.classes_  

    out_path = []
    out_pred = []

    for est in clf.estimators_:

        n_nodes = est.tree_.node_count
        children_left = est.tree_.children_left
        children_right = est.tree_.children_right
        feature = est.tree_.feature
        threshold = est.tree_.threshold

        weighted_n_node_samples = est.tree_.weighted_n_node_samples

        node_indicator = est.decision_path(X_test.to_numpy())
        leaf_id = est.apply(X_test.to_numpy())

        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        if verb:
            print('Rules used to predict sample {id}:\n'.format(id=sample_id))
        
        out_path.append([]) 
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue
                
            out_path[-1].append([weighted_n_node_samples[node_id], feature_names[feature[node_id]]])    
                                
            if verb:
                # check if value of the split feature for sample 0 is below threshold
                if (X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
            
                print("node {node}, {samples} samples : {feature} = {value:.2f} "
                      "{inequality} {threshold:.2f}".format(
                          node=node_id,
                          samples=str(weighted_n_node_samples[node_id]).rstrip('0').rstrip('.'),
                          sample=sample_id,
                          feature=feature_names[feature[node_id]],
                          value=X_test.iloc[sample_id, feature[node_id]],
                          inequality=threshold_sign,
                          threshold=threshold[node_id]))

        pred = int(est.predict(X_test.to_numpy())[sample_id])        
        out_pred.append(classes[pred])
                    
    return out_path, out_pred

def loo_train_and_classify(arr):
    
    [i, X_train, y_train, X_test, y_test, test_name], opts = arr
        
    clf = RandomForestClassifier(**opts)

    clf.fit(X_train, y_train)

    classes = clf.classes_ 

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    imp = clf.feature_importances_

    df_test = pd.DataFrame(prob, columns=classes)
    df_test['true_Class'] = y_test.tolist()
    df_test['Class'] = pred
    df_test['Class_prob'] = prob.max(axis=1)
    df_test['name'] = test_name.tolist()

    df_imp = pd.DataFrame(columns = X_test.columns)
    df_imp.loc[len(df_imp)] = np.array(imp)

    #test_path, test_pred = get_classification_path(clf, X_test, sample_id=0, verb=False)

    return i, df_test, df_imp#, test_path, test_pred

def class_train_and_classify(arr):
    
    [X_train, y_train, X_test, test_name], opts = arr
        
    clf = RandomForestClassifier(**opts)

    clf.fit(X_train, y_train)

    classes = clf.classes_ 

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    imp = clf.feature_importances_

    df_test = pd.DataFrame(prob, columns=classes)
    df_test['Class'] = pred
    df_test['Class_prob'] = prob.max(axis=1)
    df_test['name'] = test_name.tolist()

    df_imp = pd.DataFrame(columns = X_test.columns)
    df_imp.loc[len(df_imp)] = np.array(imp)

    #test_path, test_pred = get_classification_path(clf, X_test, sample_id=0, verb=False)

    return df_test, df_imp#, test_path, test_pred

def class_train_model_and_classify(arr):
    
    [X_train, y_train, X_test, test_name], model = arr
        
    clf = model

    clf.fit(X_train, y_train)

    classes = clf.classes_ 

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    #imp = clf.feature_importances_

    df_test = pd.DataFrame(prob, columns=classes)
    df_test['Class'] = pred
    df_test['Class_prob'] = prob.max(axis=1)
    df_test['name'] = test_name.tolist()

    #df_imp = pd.DataFrame(columns = X_test.columns)
    #df_imp.loc[len(df_imp)] = np.array(imp)

    #test_path, test_pred = get_classification_path(clf, X_test, sample_id=0, verb=False)

    return df_test#, df_imp#, test_path, test_pred

def loo_save_res(res, dir_out):
    
    ii = []

    df_classes = []
    df_imps = []
    paths = []

    for r in res:
        i, df_test, df_imp = r #, test_path, test_pred = r

        ii.append(i)

        df_classes.append(df_test)
        df_imps.append(df_imp)
        #paths.append([test_path, test_pred])

    ii = np.argsort(ii)  

    df_classes = [df_classes[i] for i in ii]
    df_classes = pd.concat(df_classes).reset_index(drop=True)
    df_classes.to_csv(f'{dir_out}/classes.csv', index=False)

    df_imps = [df_imps[i] for i in ii]
    df_imps = pd.concat(df_imps).reset_index(drop=True)
    df_imps.to_csv(f'{dir_out}/imps.csv', index=False)

    #paths = [paths[i] for i in ii]
    #json.dump(paths, open(f'{dir_out}/paths.json', 'wt'))
    
    print(f'output files in {dir_out}:\nclasses.csv\nimps.csv\npaths.json')

def class_save_res(res, dir_out):
    
    ii = []

    df_classes = []
    df_imps = []
    paths = []

    for i, r in enumerate(res):
        df_test = r#, df_imp = r #, test_path, test_pred = r

        ii.append(i)

        df_classes.append(df_test)
        #df_imps.append(df_imp)
        #paths.append([test_path, test_pred])

    ii = np.argsort(ii)  

    df_classes = [df_classes[i] for i in ii]
    df_classes = pd.concat(df_classes).reset_index(drop=True)
    df_classes.to_csv(f'{dir_out}/classes.csv', index=False)
    
    '''
    df_imps = [df_imps[i] for i in ii]
    df_imps = pd.concat(df_imps).reset_index(drop=True)
    df_imps.to_csv(f'{dir_out}/imps.csv', index=False)

    paths = [paths[i] for i in ii]
    # json.dump(paths, open(f'{dir_out}/paths.json', 'wt'))
    
    print(f'output files in {dir_out}:\nclasses.csv\nimps.csv\npaths.json')
    '''

from bokeh.models import ColumnDataSource, FuncTickFormatter, Plot
from bokeh.plotting import figure
from bokeh.transform import dodge, linear_cmap
import colorcet as cc

def plot_confusion_matrix(df,classes,
                          title='Normalized confusion matrix (%)',
                          cm_type='recall',
                          normalize=True,                          
                          pallete=cc.fire[::-1],
                          fill_alpha=0.6,
                          width=600, 
                          height=600,
                          plot_zeroes=True
                         ):

    #classes = np.sort(df.true_Class.unique())

    if cm_type=='recall':
        xlabel, x_class='Predicted Class', 'Class'
        ylabel, y_class='True Class', 'true_Class'
    elif cm_type=='precision':
        xlabel, x_class='True Class', 'true_Class'
        ylabel, y_class='Predicted Class', 'Class'
    else:
        raise ValueError("Type must be recall or precision!")
    
    cm = confusion_matrix(df[y_class], df[x_class], labels=classes)
    
    if normalize:
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    _ = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):        
            f = format(np.round(cm[i, j]).astype(int), 'd')  
            if not plot_zeroes and f == '0': continue # f = ''
            _.append([classes[i], classes[j], f])        
    _ =  pd.DataFrame(dict(zip([ylabel, xlabel, 'counts'], np.transpose(_))))         
    source = ColumnDataSource(_)

    p = figure(width=width, 
               height=height, 
               title=title,
               x_range=classes, 
               match_aspect=True,
               # aspect_scale=2,
               y_range=classes[::-1], 
               toolbar_location=None, 
               # tools='hover'
              )

    p.rect(xlabel, 
           ylabel, 
           1, 
           1, 
           source=source, 
           fill_alpha=fill_alpha, 
           line_color=None,
           color=linear_cmap('counts', pallete, 0, 2 * cm.max())
          )

    text_props = {'source': source, 'text_align': 'center', 'text_baseline': 'middle'}
    x = dodge(xlabel, 0, range=p.x_range)
    r = p.text(x=x, y=ylabel, text='counts', **text_props)
    r.glyph.text_font_style='bold'

    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = '10pt'
    p.xaxis.major_label_orientation = np.pi/4

    p.yaxis.axis_label = ylabel 
    p.xaxis.axis_label = xlabel

    p.axis.axis_label_text_font_size = '18pt'
    p.axis.axis_label_text_font_style = 'normal'

    p.title.text_font_size = '18pt'
    p.title.text_font_style = 'normal'

    p.axis.major_label_standoff = 5
        
    class_abun = df[y_class].value_counts().to_dict()        
    y_labels = {_: f'{_}\n{class_abun[_]}' for _ in p.y_range.factors}    
    p.yaxis.formatter = FuncTickFormatter(code=f'''
            var labels = {y_labels}
            return labels[tick]
        ''') 
        
    return(p)

def plot_CM_withSTD(cm, tds, classes,
                    normalize=False,
                    type='recall',
                    title=None,
                    cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm, axis=1)
    class_with_num = [class_labels[classes[i]] +'\n'+ str(round(total[i])) for i in range(len(total))]
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm_copy = cm.copy()
        cm = cm.astype('float') / cm_copy.sum(axis=1)[:, np.newaxis]
        tds = tds.astype('float') / cm_copy.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    if type=='recall':
        xlabel='Predicted label'
        ylabel='True label'
    elif type=='precision':
        xlabel='True label'
        ylabel='Predicted label'
    else:
        raise ValueError("Type must be recall or precision!") 

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=[class_labels[clas] for clas in classes], yticklabels=class_with_num,
           title=title,
           xlabel=xlabel,
           ylabel=ylabel)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #print(thresh)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{:.2f} \n".format(cm[i,j])+r"$\pm$"+"{:.2f}".format(tds[i,j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",fontsize=10)
    fig.tight_layout()
    return fig

def confident_flag(df, method = 'sigma-mean', thres=2., class_cols=['AGN','CV','HM-STAR','LM-STAR','HMXB','LMXB','NS','YSO']):

    if 'conf_flag' not in df:
        df['conf_flag'] = 0

    if method == 'sigma-mean':
        df['conf_flag'] = df.apply(lambda row: row.conf_flag+1 if row.Class_prob - thres*row.Class_prob_e - sorted([row['P_'+clas]+thres*row['e_P_'+clas] for clas in class_cols])[-2] >0 else row.conf_flag, axis=1)

    if method == 'sigma-qt':
        df['conf_flag'] = df.apply(lambda row: row.conf_flag+2 if row.Class_prob_2sig_low - sorted(row[[c for c in df.columns if 'P_2sig_upp' in c]])[-2] > 0 else row.conf_flag, axis=1)

    if method == 'hard-cut':
        df.loc[df.Class_prob>thres, 'conf_flag'] = df.loc[df.Class_prob>thres, 'conf_flag']+4

    return df 

def mw_counterpart_flag(df, mw_cols=['Gmag','BPmag','RPmag','Jmag','Hmag','Kmag','W1mag_comb','W2mag_comb','W3mag_allwise']):

    df['mw_cp_flag'] = 0
    df = df.fillna(exnum)#df.replace(np.nan, exnum, inplace=True)

    for i, mw_col in enumerate(mw_cols):
        df['mw_cp_flag'] = df.apply(lambda row: row.mw_cp_flag+2**i if row[mw_col]!=exnum else row.mw_cp_flag, axis=1)

    df = df.replace(exnum, np.nan)

    return df


def find_confident(df, method='70', thres=2.):
    
    df_conf = pd.DataFrame()
    
    if method =='sigma-qt':
        
        for i, df_s in df.iterrows():
            #print(i, df_s) 
            prob1 = df_s['Class_prob_2sig_low']#.values[0]
            upp_prob_cols = [c for c in df.columns if 'P_2sig_upp' in c]
            upp_prob_cols.remove('P_2sig_upp_'+df_s['Class'])
            prob2 = max([df_s[upp_prob_cols[i]] for i in range(len(upp_prob_cols))])
            if prob1 > prob2:
                
                df_conf = df_conf.append(df_s)
    

    if method == '70':
        
        for source,i in zip(df.name.unique(), range(len(df.name.unique()))): 
            df_source = df[df.name==source]
            if df_source.loc[df['name']==source, 'Class_prob'].values>0.7:
                df_conf = df_conf.append(df_source)
    
    if method == 'sigma':
        
        for source,i in zip(df.name.unique(), range(len(df.name.unique()))): 
            df_s = df[df.name==source]
            
            prob1 = df_s['Class_prob'].values[0]-thres*df_s['Class_prob_e'].values[0]
            
            classes = [c.strip('e_P_') for c in df.columns if 'e_P' in c]

            prob_cols = ['P_' + c for c in classes]
            e_prob_cols = ['e_P_' + c for c in classes]
            
            prob_cols.remove('P_'+df_s['Class'].values)
            e_prob_cols.remove('e_P_'+df_s['Class'].values)
            
            #print(prob_cols)
            #print(e_prob_cols)
            
            #prob2 = max([df_s[prob_cols[i]].values[0] for i in range(len(prob_cols))])
            prob2 = max([df_s[prob_cols[i]].values[0]+thres*df_s[e_prob_cols[i]].values[0] for i in range(len(prob_cols))])
            
            if prob1 > prob2:
                
                df_conf = df_conf.append(df_s)
    
    if method == 'both':
        
         for source,i in zip(df.name.unique(), range(len(df.name.unique()))): 
            df_s = df[df.name==source]
            
            prob1 = df_s['Class_prob'].values[0]-thres*df_s['Class_prob_e'].values[0]
            
            prob_cols = ['P_AGN','P_NS','P_CV','P_HMXB','P_LMXB','P_LM-STAR','P_HM-STAR','P_YSO']
            e_prob_cols = ['e_P_AGN','e_P_NS','e_P_CV','e_P_HMXB','e_P_LMXB','e_P_LM-STAR','e_P_HM-STAR','e_P_YSO']
            
            prob_cols.remove('P_'+df_s['Class'].values)
            e_prob_cols.remove('e_P_'+df_s['Class'].values)
            
            #print(prob_cols)
            #print(e_prob_cols)
            
            #prob2 = max([df_s[prob_cols[i]].values[0] for i in range(len(prob_cols))])
            prob2 = max([df_s[prob_cols[i]].values[0]+thres*df_s[e_prob_cols[i]].values[0] for i in range(len(prob_cols))])
            
            if prob1 > prob2 and df_s['Class_prob'].values>0.7:
                
                df_conf = df_conf.append(df_s)
        
    if method == 'previous':
        
        for source,i in zip(df.name.unique(), range(len(df.name.unique()))): 
            df_s = df[df.name==source]
            
            prob1 = df_s['Class_prob'].values[0]-thres*df_s['Class_prob_e'].values[0]
            
            prob_cols = ['P_AGN','P_NS','P_CV','P_HMXB','P_LMXB','P_LM-STAR','P_HM-STAR','P_YSO']
            e_prob_cols = ['e_P_AGN','e_P_NS','e_P_CV','e_P_HMXB','e_P_LMXB','e_P_LM-STAR','e_P_HM-STAR','e_P_YSO']
            
            prob_cols.remove('P_'+df_s['Class'].values)
            e_prob_cols.remove('e_P_'+df_s['Class'].values)
            
            #print(prob_cols)
            #print(e_prob_cols)
            
            prob2 = max([df_s[prob_cols[i]].values[0] for i in range(8)])
            #print(prob2)
            
            if prob1 > prob2:
                
                df_conf = df_conf.append(df_s)
        
    
    return df_conf

def confidence(df_class, weighted=False, cut='sigma', sigma=2):

    df_class = df_class.copy()

    # get list of classes from probability columns
    classes = [c.strip('e_P_') for c in df_class.columns if 'e_P' in c and 'e_P_w' not in c]

    # define stellar compact object classes
    CO_classes = ['LMXB', 'HMXB', 'CV', 'NS']
    nonCO_classes = list(set(classes)-set(CO_classes))

    # define probability and error of stellar compact object 



    if weighted:
        df_class['P_CO'] = df_class[['P_w_' + c for c in CO_classes]].sum(1)
        df_class['e_P_CO'] = np.sqrt(np.sum(np.square(df_class[['e_P_w_' + c for c in CO_classes]]), 1))

    else:
        df_class['P_CO'] = df_class[['P_' + c for c in CO_classes]].sum(1)
        df_class['e_P_CO'] = np.sqrt(np.sum(np.square(df_class[['e_P_' + c for c in CO_classes]]), 1))

    # candidate CO if P_CO - sigma stdev greater than max of non CO classes' sigma * stdev upper probability limit
    df_class['Candidate_CO'] = (df_class['P_CO'] - sigma*df_class['e_P_CO'] >= (df_class.loc[:,['P_' + c for c in nonCO_classes]].values + sigma*df_class.loc[:,['e_P_' + c for c in nonCO_classes]].values).max(1))

    # df_class['Candidate_CO'] = df_class['P_CO']>=0.7

    if cut == "simple":
        df_class["Class"]=np.where(df_class["Class_prob"]>=0.70, df_class["Class"], "Unconfident Classification")

    if cut == "sigma":
        # select sources whose most confident classification is sigma stdev greater than all other classifications' P + sigma*e_P.

        classes = [c.strip('e_P_') for c in df_class.columns if 'e_P' in c and 'e_P_w' not in c]
        classes.remove('CO')
        ps = df_class.loc[:,['P_' + c for c in classes]]
        pes = df_class.loc[:,['e_P_' + c for c in classes]]

        # find second largest probability
        # second = ps.apply(lambda row: row.nlargest(n).values[-1], axis=1)
        # second_e = pes.apply(lambda row: row.nlargest(2).values[-1], axis=1)

        # p + sigma*e_p
        second = ps.values+sigma*pes.values
    
        # remove most confident classification from second
        idx = ps.values.argmax(1)[:,None]
        second = second[np.arange(ps.shape[1]) != idx].reshape(ps.shape[0],-1)

        if weighted:
            df_class["Class_w"]=np.where(df_class["Class_prob_w"]-sigma*df_class["Class_prob_e_w"]>=second.max(1), df_class["Class_w"], "Unconfident Classification")
        else:
            df_class["Class"]=np.where(df_class["Class_prob"]-sigma*df_class["Class_prob_e"]>=second.max(1), df_class["Class"], "Unconfident Classification")


    return df_class
    
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')    

def rename(data):
    
    data = data.rename(columns = {
         'flux_aper90_ave_b':'flux_aper_b', 'flux_aper90_ave_h':'flux_aper_h',
         'flux_aper90_ave_m':'flux_aper_m', 'flux_aper90_ave_s':'flux_aper_s',
        'var_inter_prob':'var_inter_prob_b','kp_prob_b_max':'var_intra_prob_b'
    })

def classification_probs(data):
    probs_median = []
    probs_std = []
    preds = []
    sources = []
    pred_p = []
    pred_e_p = []
    true_classes = []

    for source,i in zip(data.name.unique(), range(len(data.name.unique()))): 
        df_source = data[data.name==source]
        src_probs = np.array(df_source.iloc[:,:8])

        prob_median = np.median(src_probs, axis=0) / np.median(src_probs, axis=0).sum()

        # median of probability vectors no longer add to 1, normaize

        prob_std = (np.percentile(src_probs,84,axis=0) - np.percentile(src_probs,16,axis=0))/2.
        #print(prob_ave,'\n',prob_std)
        probs_median.append(prob_median)
        probs_std.append(prob_std)
        ind_pred =  np.argmax(prob_median)
        preds.append(data.columns[ind_pred])
        if 'true_Class' in data.columns:
            true_classes.append(df_source['true_Class'].values[0])
        sources.append(source)
        pred_p.append(prob_median[ind_pred])
        pred_e_p.append(prob_std[ind_pred])

    if 'true_Class' in data.columns:
        df_save = pd.DataFrame({'name': sources, 
                                'Class': preds,
                                'true_Class': true_classes,
                                'Class_prob': pred_p,
                                'Class_prob_e': pred_e_p}
                            )
    else:
        df_save = pd.DataFrame({'name': sources, 
                            'Class': preds,
                            'Class_prob': pred_p,
                            'Class_prob_e': pred_e_p}
                        )

    for i in range(8):
        df_save['P_'+data.columns[i]] = np.array(probs_median)[:,i]
        df_save['e_P_'+data.columns[i]] = np.array(probs_std)[:,i] 

    return df_save

def classification_probs_old(data):
    df_all = pd.DataFrame()
    probs_ave = []
    probs_std = []
    for src, i in zip(data.name.unique(), range(len(data))):
        
        df_src = data[data.name==src]

        prob_ave = np.mean(df_src.iloc[:, :9].values, axis=0)
        prob_std = np.std(df_src.iloc[:, :9].values,axis=0)
        probs_ave.append(prob_ave)
        probs_std.append(prob_std)
    #     print(prob_ave,prob_std)
        
        # column index of most probable class
        ind_pred =  np.argmax(prob_ave)
    #     print(ind_pred)
        preds = data.columns[ind_pred]
        
        pred_p = prob_ave[ind_pred]
        pred_e_p = prob_std[ind_pred]
        #print(src,preds,pred_p,pred_e_p)
        
        new_row = {'name': src,'Class': preds,'Class_prob': pred_p,'Class_prob_e': pred_e_p}
        df_all = df_all.append(new_row, ignore_index=True)
        
    #print(probs_ave)    

    for i in range(9):
        df_all['P_'+data.columns[i]] = np.array(probs_ave)[:,i]
        df_all['e_P_'+data.columns[i]] = np.array(probs_std)[:,i] 

    return (df_all)

def plot_classifier_matrix_withSTD_old(df, title='Classifier matrix', nocmap=False, cmap=plt.cm.Blues):

    '''
    Plot classification probability matrix with standard deviations

            Parameters:
                    df: dataframe with source classification, classification probabilities and stdevs for all classes 
            Returns:
                    fig: plt figure
    '''

    classes = [c.strip('e_P_') for c in df.columns if 'e_P' in c]

    probs = df.loc[:,['P_' + c for c in classes]].to_numpy()
    stds = df.loc[:,['e_P_' + c for c in classes]].to_numpy()
    names= [str(i) + '. ' + df['name'][i] for i in df.index.to_list()]

    length = df.shape[0]
    fig, ax = plt.subplots(figsize=(21, length*1.5+3))
    im = ax.imshow(probs, interpolation='nearest', cmap=cmap)
    if nocmap ==False:
        ax.figure.colorbar(im, ax=ax)
    probs = np.array(probs)
    # We want to show all ticks...
    ax.set(xticks=np.arange(probs.shape[1]),
           yticks=np.arange(probs.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=names,
           title=title,
           #ylabel='True label',
           xlabel='Class')
        
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
           
    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    thresh = probs.max() / 2.
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, "{:.2f} \n".format(probs[i,j])+r"$\pm$"+"{:.2f}".format(stds[i,j]),
                    ha="center", va="center",
                    color="white" if probs[i, j] > thresh else "black")

    fig.tight_layout()
    return fig

plt.rcParams.update({'font.size': 30})
params = {'legend.fontsize': 'large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)

def plot_classifier_matrix_withSTD(probs, stds, pred, yaxis, classes, normalize=False,
                           title=False, nocmap=False, 
                           cmap=plt.cm.Blues):
    textsize=20
    if not title:
        title = 'Classifier matrix'
    length = len(pred)
    fig, ax = plt.subplots(figsize=(21, length*1.5+3))
    im = ax.imshow(probs, interpolation='nearest', cmap=cmap)
    if nocmap ==False:
        ax.figure.colorbar(im, ax=ax)
    probs = np.array(probs)
    # We want to show all ticks...
    ax.set(xticks=np.arange(probs.shape[1]),
           yticks=np.arange(probs.shape[0]))#,
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=yaxis,
           #title=title)#,
           #ylabel='True label',
           #xlabel='Class')
        
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
           
    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    thresh = probs.max() / 2.
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            ax.text(j, i, "{:.2f} \n".format(probs[i,j])+r"$\pm$"+"{:.2f}".format(stds[i,j]),
                    ha="center", va="center",
                    color="white" if probs[i, j] > thresh else "black",fontsize=textsize)
    
    #ax.set_xticklabels(('0','1','2','3','4','5','6','7','8','9'),fontweight='bold',fontsize=texts)
    ax.set_title(title,fontsize=textsize*1.2)
    ax.set_xticklabels(classes,fontsize=textsize)
    ax.set_yticklabels(yaxis,fontsize=textsize)

    fig.tight_layout()
    return fig

figs1, figs2 = 10., 12. 
texts = 15 # text fontsize

def plot_Feature_Importance_withSTD(imp, std, features, fig_width, fig_height):
    #sbn.set_style("white")
    N = len(imp)
    
    ind = np.arange(N)  # the x locations for the groups
    
    width = 0.7#16./N *2.      # the width of the bars
    
    fig, ax = plt.subplots(figsize=(fig_width/80, fig_height/80))
    rects1 = ax.barh(ind, imp*100, width, xerr=std*100, ecolor='orange')
    #rects2 = ax.barh(ind, imp_noran*100, width, xerr=std_noran*100, alpha=0, ecolor='red')
    #rects2 = ax.bar(ind + width, lassoo*100, width, color='g')
    # add some text for labels, title and axes ticks
    ax.set_xlabel('Importance (in % usage)',fontweight='bold',fontsize=texts)
    ax.set_xticks(range(10))
    ax.set_xticklabels(('0','1','2','3','4','5','6','7','8','9'),fontweight='bold',fontsize=texts)
    ax.set_title('Feature Importance',fontweight='bold',fontsize=texts*1.2)
    ax.set_yticks(ind)
    ax.set_yticklabels(features,fontweight='bold',fontsize=texts/1.5)
    
    #ax.legend(rects1[0], ('Random Forest Regressor'))
    ax.set_xlim(0,9)
    ax.set_ylim(-0.5,N+0.5)

    
    #print('There are ',len(features), ' features')
    #print(features)
    #print(imp)
    for threshold, color in zip([1.], ['red']):#zip([0.8, 1., 1.5], ['orange','red','green']):
        
        thres = threshold/100.
        plt.axvline(threshold,color=color)
        
        
        index_feature_select = np.where(imp>thres)[0]
        features_selected = np.array(features)[index_feature_select]
        features_selected_imp = imp[index_feature_select]
        #print('There are ',len(features_selected), ' features selected with thres at', str(threshold))
        #print( features_selected, ' as Selected features')
        #print(features_selected_imp, ' as Selected features imps')
    #plt.show()
    #fig.tight_layout()
    return fig


