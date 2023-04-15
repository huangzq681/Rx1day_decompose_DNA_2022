import xarray as xr
import numpy as np
from scipy.stats import linregress
import glob

full_name = {
    'prec':'precipitation',
    'tas':'surface temperature',
    'local_surf_temp':'annual local surface temperature',
    'scaling':'full scaling',
    'scaling_thermo':'thermodynamic scaling',
    'scaling_dynamic':'dynamic scaling',
    'scaling_interaction':'interaction between thermodynamic and dynamic scaling'
}

var_name = {
    'prec':'prec_cond',
    'tas':'tas_cond',
    'local_surf_temp':'tas',
    'scaling':'scaling',
    'scaling_thermo':'scaling_thermo',
    'scaling_dynamic':'scaling_dynamic',
    'scaling_interaction':'scaling_interaction'
}

forcing_name = ['historical','hist-GHG','hist-aer','hist-nat','piControl','era5','jra55','ncep2']
frcs_src_run = {
    'historical':{'ACCESS-CM2':['r1i1p1f1'],
                  'ACCESS-ESM1-5':['r1i1p1f1'],
                  'AWI-ESM-1-1-LR':['r1i1p1f1'],
                  'CAMS-CSM1-0':['r2i1p1f1'],
                  'CESM2-FV2':['r1i1p1f1'],
                  'CESM2-WACCM':['r1i1p1f1'],
                  'CESM2':['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1'],
                  'CanESM5':['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1'],
                  'FGOALS-g3':['r1i1p1f1'],
                  'MIROC-ES2L':['r1i1p1f2'],
                  'MIROC6':['r1i1p1f1'],
                  'MPI-ESM-1-2-HAM':['r1i1p1f1'],
                  'MPI-ESM1-2-HR':['r1i1p1f1'],
                  'MPI-ESM1-2-LR':['r1i1p1f1'],
                  'MRI-ESM2-0':['r1i1p1f1'],
                  'NESM3':['r1i1p1f1']
                 },
    'hist-GHG':{'ACCESS-CM2':['r2i1p1f1'],
                'ACCESS-ESM1-5':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'CanESM5':['r1i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r10i1p1f1'], #
                'HadGEM3-GC31-LL':['r1i1p1f3','r2i1p1f3','r3i1p1f3','r4i1p1f3'],
                'MIROC6':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'MRI-ESM2-0':['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1'],
                'NorESM2-LM':['r1i1p1f1','r2i1p1f1','r3i1p1f1']},
    'hist-aer':{'ACCESS-CM2':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'ACCESS-ESM1-5':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'CanESM5':['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1'],
                'HadGEM3-GC31-LL':['r1i1p1f3','r2i1p1f3','r3i1p1f3','r4i1p1f3'],
                'MIROC6':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'MRI-ESM2-0':['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1'],
                'NorESM2-LM':['r1i1p1f1','r2i1p1f1','r3i1p1f1']},
    'hist-nat':{'ACCESS-CM2':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'ACCESS-ESM1-5':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'CanESM5':['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1'],
                'HadGEM3-GC31-LL':['r1i1p1f3','r2i1p1f3','r3i1p1f3','r4i1p1f3'],
                'MIROC6':['r1i1p1f1','r2i1p1f1','r3i1p1f1'],
                'MRI-ESM2-0':['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1'],
                'NorESM2-LM':['r1i1p1f1','r2i1p1f1','r3i1p1f1']},
    'piControl':{'AWI-ESM-1-1-LR':['r1i1p1f1'],
                 'CESM2-FV2':['r1i1p1f1'],
                 'CESM2-WACCM':['r1i1p1f1'],
                 'CESM2':['r1i1p1f1'],
                 'CMCC-CM2-SR5':['r1i1p1f1'],
                 'CMCC-ESM2':['r1i1p1f1'],
                 'CanESM5':['r1i1p1f1'],
                 'IITM-ESM':['r1i1p1f1'],
                 'MIROC6':['r1i1p1f1'],
                 'MPI-ESM-1-2-HAM':['r1i1p1f1'],
                 'MPI-ESM1-2-HR':['r1i1p1f1'],
                 'MPI-ESM1-2-LR':['r1i1p1f1'],
                 'MRI-ESM2-0':['r1i1p1f1'],
                 'NorESM2-LM':['r1i1p1f1'],
                 'NorESM2-MM':['r1i1p1f1'],
                 'TaiESM1':['r1i1p1f1']},
    'era5':{'era5':['reanalysis']},
    'jra55':{'jra55':['reanalysis']},
    '20CR':{'20CR':['reanalysis']},
    'merra2':{'merra2':['reanalysis']},
    'ncep2':{'ncep2':['reanalysis']},
    'hist-mme':{'hist-mme':['hist-mme']},
    'aer-mme':{'aer-mme':['aer-mme']},
    'GHG-mme':{'GHG-mme':['GHG-mme']},
    'nat-mme':{'nat-mme':['nat-mme']}
}

forcing_dir = {
    'historical':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/historical/',
    'hist-GHG':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/GHG/',
    'hist-aer':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/aer/',
    'hist-nat':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/nat/',
    'piControl':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/piCtrl/',
    'era5':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/era5/',
    'jra55':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/jra55/',
    '20CR':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/20CR/',
    'ncep2':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/ncep2/',
    'hist-mme':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/historical/',
    'GHG-mme':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/GHG/',
    'aer-mme':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/aer/',
    'nat-mme':'/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/plotScript/scaling_res_NW/nat/',
}

ensembles = {
    'historical':[s+'_historical_'+i for s,r in frcs_src_run['historical'].items() for i in r],
    'hist-GHG':[s+'_hist-GHG_'+i for s,r in frcs_src_run['hist-GHG'].items() for i in r],
    'hist-aer':[s+'_hist-aer_'+i for s,r in frcs_src_run['hist-aer'].items() for i in r],
    'hist-nat':[s+'_hist-nat_'+i for s,r in frcs_src_run['hist-nat'].items() for i in r],
    'piControl':[s+'_piControl_'+i for s,r in frcs_src_run['piControl'].items() for i in r]
}

target_griddes = xr.Dataset({'lat': (['lat'], np.linspace(start=-89.375,stop=89.375,num=144)),
                             'lon': (['lon'], np.linspace(start=0.9375,stop=359.0625,num=192)),})

target_griddes_2 = xr.Dataset({'lat': (['lat'], np.linspace(start=-90,stop=90,num=73)),
                             'lon': (['lon'], np.linspace(start=1.25,stop=358.75,num=144)),})

hadex_rx1day = xr.open_dataset('/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/HadEX3/HadEX3-0-4_rx1day_ann.nc')
hadex_rx1day = hadex_rx1day['Rx1day']
hadex_rx1day = hadex_rx1day.rename({'longitude':'lon','latitude':'lat'})
hadex_rx1day = hadex_rx1day.sel(time=slice('1960','2014'))

hadex_rx1day_not_null = ~hadex_rx1day.isnull()
hadex_rx1day_not_null_sum = hadex_rx1day_not_null.sum(axis=0,skipna=True)
hadex_rx1day_grid_sel = hadex_rx1day_not_null_sum >= (2014-1960+1) * 0.85
hadex_rx1day_grid_sel = hadex_rx1day_grid_sel.where(hadex_rx1day_grid_sel>0)

hadex_rx5day = xr.open_dataset('/Users/zeqinhuang/Documents/paper/prec_scaling_da/scaling_script/HadEX3/HadEX3_Rx5day_ANN.nc')
hadex_rx5day = hadex_rx5day['Rx5day']
hadex_rx5day = hadex_rx5day.rename({'longitude':'lon','latitude':'lat'})
hadex_rx5day = hadex_rx5day.sel(time=slice('1960','2014'))

hadex_rx5day_not_null = ~hadex_rx5day.isnull()
hadex_rx5day_not_null_sum = hadex_rx5day_not_null.sum(axis=0,skipna=True)
hadex_rx5day_grid_sel = hadex_rx1day_grid_sel

# domain range for subregions
domain_lonlat = {
    'EAS':{'lon_min':88,'lon_max':140,'lat_min':20,'lat_max':59}, # 52 * 39
    'EU':{'lon_min':360-10,'lon_max':38,'lat_min':36,'lat_max':72}, # 48 * 36
    'NA':{'lon_min':360-124,'lon_max':360-68,'lat_min':16,'lat_max':55}, # 56 * 42
    'AU':{'lon_min':114,'lon_max':154,'lat_min':-40,'lat_max':-10}, # 40 * 30
    'SSA':{'lon_min':360-78,'lon_max':360-42,'lat_min':-45,'lat_max':-15}, # 40 * 30
}

def sel_lonlat_range(dataarray,lon_min,lon_max,lat_min,lat_max):
    if lon_min > lon_max:
        mask_lon = ((dataarray.lon >= lon_min) & (dataarray.lon <= 360)) | ((dataarray.lon <= lon_max) & (dataarray.lon >= 0))
    else:
        mask_lon = (dataarray.lon >= lon_min) & (dataarray.lon <= lon_max)
    mask_lat = (dataarray.lat >= lat_min) & (dataarray.lat <= lat_max)
    dataarray = dataarray.where(mask_lon & mask_lat, drop=False)
    return dataarray

mask_global = hadex_rx1day_grid_sel
mask_EAS = sel_lonlat_range(mask_global,lon_min=domain_lonlat['EAS']['lon_min'],lon_max=domain_lonlat['EAS']['lon_max'],lat_min=domain_lonlat['EAS']['lat_min'],lat_max=domain_lonlat['EAS']['lat_max'])
mask_EU = sel_lonlat_range(mask_global,lon_min=domain_lonlat['EU']['lon_min'],lon_max=domain_lonlat['EU']['lon_max'],lat_min=domain_lonlat['EU']['lat_min'],lat_max=domain_lonlat['EU']['lat_max'])
mask_NA = sel_lonlat_range(mask_global,lon_min=domain_lonlat['NA']['lon_min'],lon_max=domain_lonlat['NA']['lon_max'],lat_min=domain_lonlat['NA']['lat_min'],lat_max=domain_lonlat['NA']['lat_max'])
mask_AU = sel_lonlat_range(mask_global,lon_min=domain_lonlat['AU']['lon_min'],lon_max=domain_lonlat['AU']['lon_max'],lat_min=domain_lonlat['AU']['lat_min'],lat_max=domain_lonlat['AU']['lat_max'])
mask_SSA = sel_lonlat_range(mask_global,lon_min=domain_lonlat['SSA']['lon_min'],lon_max=domain_lonlat['SSA']['lon_max'],lat_min=domain_lonlat['SSA']['lat_min'],lat_max=domain_lonlat['SSA']['lat_max'])

## functions
def _compute_slope(var):
    slp = linregress(range(len(var)),var).slope
    return slp

def _compute_slope_skipna(var):
    if (var!=0).all():
        slp = linregress(range(len(var)),var).slope
    elif (var==0).all():
        slp = linregress(range(len(var)),var).slope
    else:
        var = var[~np.isnan(var)]
        slp = linregress(range(len(var)),var).slope
    return slp

def _compute_sig(var):
    sig = linregress(range(len(var)),var).pvalue
    return sig

def trend_cal(data):
    slopes = xr.apply_ufunc(_compute_slope,
                            data,
                            vectorize=True,
                            dask='parallelized', 
                            input_core_dims=[['time']],
                            output_dtypes=[float],
                            )
    return slopes

def trend_cal_skipna(data):
    slopes = xr.apply_ufunc(_compute_slope_skipna,
                            data,
                            vectorize=True,
                            dask='parallelized', 
                            input_core_dims=[['time']],
                            output_dtypes=[float],
                            )
    return slopes

def Regridder(data):
    data = data.interp(lat=target_griddes['lat'],lon=target_griddes['lon'], method="nearest")
    return data

def Regridder2(data):
    data = data.interp(lat=target_griddes_2['lat'],lon=target_griddes_2['lon'], method="nearest")
    return data

class Rx1dayCond():
    def __init__(self,name,forcing,src_id,run,mask='Global',pi=None,pi_time=None,for_sn=False,for_satellite=False):
        self.name = name
        self.forcing  = forcing
        self.src_id = src_id
        self.run  = run
        self.mask = mask
        self.pi   = pi
        self.pi_time = pi_time
        self.for_sn = for_sn
        self.for_satellite = for_satellite
        self.target_griddes = target_griddes
        
        if self.name not in full_name.keys():
            raise Exception('ERROR, name must be a string object belongs to {}'.format(list(full_name.keys())))
        if self.forcing not in frcs_src_run.keys():
            raise Exception('ERROR, forcing must be a string object belongs to {}'.format(list(frcs_src_run.keys())))
        if self.src_id not in frcs_src_run[self.forcing].keys():
            raise Exception('ERROR, for forcing {f}, src_id must be an element of {s}'.format(f = self.forcing, s=frcs_src_run[self.forcing].keys()))
        if self.run not in frcs_src_run[self.forcing][self.src_id]:
            raise Exception('ERROR, for forcing {f}, src_id {i}, run must be an element of {r}'.format(f=self.forcing,i=self.src_id,r=frcs_src_run[self.forcing][self.src_id]))
        if self.mask not in ['None','EU','EAS','SSA','Global','AU','NA']:
            raise Exception('ERROR, mask must be a string object belongs to [Global, EAS, EU, NA, AU, SSA]')
        
        if self.forcing != 'piControl':
            self.pi_time = None
            self.file = glob.glob(forcing_dir[self.forcing] + self.name +'_day'+'*' + src_id + '_' + '*' + run + '*.nc')[0]
        else:
            pi_num = len(glob.glob(forcing_dir[self.forcing] + self.name +'_day'+ '*' + src_id + '*' + run + '*.nc'))
            ens    = [filepath[-12:-3] for filepath in glob.glob(forcing_dir[self.forcing] + self.name +'*' + src_id + '*' + run + '*.nc')]
            if self.pi == None and self.pi_time == None:
                raise Exception('ERROR, for piControl forcing, provide \'pi\' parameter for initiation, which should be a time period {a} or the order of {b}'.format(a=ens,b=[i for i in range(pi_num)]))
            elif self.pi != None and self.pi_time == None:
                self.file = glob.glob(forcing_dir[self.forcing] + self.name +'_day_' + src_id + '*' + run + '*.nc')[self.pi]
            elif self.pi == None and self.pi_time != None:
                self.file = glob.glob(forcing_dir[self.forcing] + self.name +'_day_' + src_id + '*' + run + '*' + self.pi_time + '.nc')[0]
            else:
                self.file = glob.glob(forcing_dir[self.forcing] + self.name +'_day_' + src_id + '*' + run + '*.nc')[self.pi]
        
        self.fullname = full_name[name]
        data0 = xr.open_dataset(self.file)
        if self.for_sn == False:
            self.data = data0.sel(time=slice('1960','2014'))
        elif self.for_sn == True:
            self.data = data0.sel(time=slice('1951','2014'))
        else:
            pass
        # if self.forcing == 'era5' and self.for_satellite == True:
        if self.for_satellite == True:
            self.data = data0.sel(time=slice('1979','2014'))

        if self.src_id == 'ACCESS-ESM1-5':
            self.data[var_name[self.name]][:,:,-1] = (self.data[var_name[self.name]][:,:,0]+self.data[var_name[self.name]][:,:,-2])/2
        else:
            pass

        self.dims = self.data.dims
        self.coords = self.data.coords

        self.origin_shape = self.data[var_name[self.name]].shape
        
        try:
            self.data = self.data.rename({'longitude':'lon','latitude':'lat'})
        except:
            pass
        try: 
            self.data = self.data.rename({'year':'time'})
        except:
            pass
        
        try:
            self.data['time'] = self.data['time'].dt.year
        except:
            pass
        
        self.area_weights = None
            
    def regrid(self,data=None):
        if data is None:
            pass
        else:
            self.data = data
        data_regridded = Regridder(self.data)
        
        if self.forcing == 'era5':
            if self.name == 'prec':
                data_regridded = data_regridded * 1000
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'jra55':
            if self.name == 'prec':
                data_regridded = data_regridded
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'hist-mme' or self.forcing == 'aer-mme' or self.forcing == 'GHG-mme' or self.forcing == 'nat-mme':
            data_regridded = data_regridded
        else:
            if self.name=='prec' or self.name=='scaling' or self.name=='scaling_thermo' or self.name=='scaling_dynamic' or self.name=='scaling_interaction':
                data_regridded = data_regridded * 86400
            else:
                pass
        
        if self.mask == 'None':
            data_regridded = data_regridded
        elif self.mask == 'Global':
            data_regridded = data_regridded * mask_global
        elif self.mask == 'EU':
            data_regridded = data_regridded * mask_EU
        elif self.mask == 'EAS':
            data_regridded = data_regridded * mask_EAS
        elif self.mask == 'NA':
            data_regridded = data_regridded * mask_NA
        elif self.mask == 'SSA':
            data_regridded = data_regridded * mask_SSA
        elif self.mask == 'AU':
            data_regridded = data_regridded * mask_AU

        else:
            raise('ERROR!')
        
        return data_regridded

    def regrid_2(self,data=None):
        if data is None:
            pass
        else:
            self.data = data
        data_regridded = Regridder2(self.data)
        
        if self.forcing == 'era5':
            if self.name == 'prec':
                data_regridded = data_regridded * 1000
            else:
                data_regridded = data_regridded * 86400

        if self.forcing == 'jra55':
            if self.name == 'prec':
                data_regridded = data_regridded
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'hist-mme' or self.forcing == 'aer-mme' or self.forcing == 'GHG-mme' or self.forcing == 'nat-mme':
            data_regridded = data_regridded
        else:
            if self.name=='prec' or self.name=='scaling' or self.name=='scaling_thermo' or self.name=='scaling_dynamic' or self.name=='scaling_interaction':
                data_regridded = data_regridded * 86400
            else:
                pass
        
        if self.mask == 'None':
            data_regridded = data_regridded
        else:
            raise('ERROR!')
        
        return data_regridded

    def multiyear_mean(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        data_regridded_avg = data_regridded[var_name[self.name]].mean(axis = 0)
        return data_regridded_avg

    def trend_cal(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        slopes = xr.apply_ufunc(_compute_slope,
                                data_regridded,
                                vectorize=True,
                                dask='parallelized', 
                                input_core_dims=[['time']],
                                output_dtypes=[float],
                                )
        return slopes
    
    def trend_cal2(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        sig = xr.apply_ufunc(_compute_sig,
                             data_regridded,
                             vectorize=True,
                             dask='parallelized', 
                             input_core_dims=[['time']],
                             output_dtypes=[float],
                             )
        return sig
           
def spatial_weighted_mean(data_array):
    area_weights = np.cos(np.deg2rad(data_array.lat))
    area_weights.name = 'weights'
    data_array_weighted = data_array.weighted(area_weights)
    data_array_weighted_mean = data_array_weighted.mean(('lon','lat'),skipna=True)
    return data_array_weighted_mean

class Rx3dayCond():
    def __init__(self,name,forcing,src_id,run,mask='Global',pi=None,pi_time=None,for_sn=False,for_satellite=False):
        self.name = name
        self.forcing  = forcing
        self.src_id = src_id
        self.run  = run
        self.mask = mask
        self.pi   = pi
        self.pi_time = pi_time
        self.for_sn = for_sn
        self.for_satellite = for_satellite
        self.target_griddes = target_griddes
        
        if self.name not in full_name.keys():
            raise Exception('ERROR, name must be a string object belongs to {}'.format(list(full_name.keys())))
        if self.forcing not in frcs_src_run.keys():
            raise Exception('ERROR, forcing must be a string object belongs to {}'.format(list(frcs_src_run.keys())))
        if self.src_id not in frcs_src_run[self.forcing].keys():
            raise Exception('ERROR, for forcing {f}, src_id must be an element of {s}'.format(f = self.forcing, s=frcs_src_run[self.forcing].keys()))
        if self.run not in frcs_src_run[self.forcing][self.src_id]:
            raise Exception('ERROR, for forcing {f}, src_id {i}, run must be an element of {r}'.format(f=self.forcing,i=self.src_id,r=frcs_src_run[self.forcing][self.src_id]))
        if self.mask not in ['None','EU','EAS','SSA','Global','AU','NA']:
            raise Exception('ERROR, mask must be a string object belongs to [Global, EAS, EU, NA, AU, SSA]')
        
        if self.forcing != 'piControl':
            self.pi_time = None
            self.file = glob.glob(forcing_dir[self.forcing] + 'Rx3day/' + self.name +'_day'+'*' + src_id + '_' + '*' + run + '*.nc')[0]
        else:
            pi_num = len(glob.glob(forcing_dir[self.forcing] + 'Rx3day/' + self.name +'_day'+ '*' + src_id + '*' + run + '*.nc'))
            ens    = [filepath[-12:-3] for filepath in glob.glob(forcing_dir[self.forcing] + 'Rx3day/' + self.name +'*' + src_id + '*' + run + '*.nc')]
            if self.pi == None and self.pi_time == None:
                raise Exception('ERROR, for piControl forcing, provide \'pi\' parameter for initiation, which should be a time period {a} or the order of {b}'.format(a=ens,b=[i for i in range(pi_num)]))
            elif self.pi != None and self.pi_time == None:
                self.file = glob.glob(forcing_dir[self.forcing] + 'Rx3day/' + self.name +'_day_' + src_id + '*' + run + '*.nc')[self.pi]
            elif self.pi == None and self.pi_time != None:
                self.file = glob.glob(forcing_dir[self.forcing] + 'Rx3day/' + self.name +'_day_' + src_id + '*' + run + '*' + self.pi_time + '*.nc')[0]
            else:
                self.file = glob.glob(forcing_dir[self.forcing] + 'Rx3day/' + self.name +'_day_' + src_id + '*' + run + '*.nc')[self.pi]

        self.fullname = full_name[name]
        data0 = xr.open_dataset(self.file)
        if self.for_sn == False:
            self.data = data0.sel(time=slice('1960','2014'))
        elif self.for_sn == True:
            self.data = data0.sel(time=slice('1951','2014'))
        else:
            pass
        if self.forcing == 'era5' and self.for_satellite == True:
            self.data = data0.sel(time=slice('1979','2014'))

        if self.src_id == 'ACCESS-ESM1-5':
            self.data[var_name[self.name]][:,:,-1] = (self.data[var_name[self.name]][:,:,0]+self.data[var_name[self.name]][:,:,-2])/2
        else:
            pass

        self.dims = self.data.dims
        self.coords = self.data.coords

        self.origin_shape = self.data[var_name[self.name]].shape
        
        try:
            self.data = self.data.rename({'longitude':'lon','latitude':'lat'})
        except:
            pass
        try: 
            self.data = self.data.rename({'year':'time'})
        except:
            pass
        
        try:
            self.data['time'] = self.data['time'].dt.year
        except:
            pass
        
        self.area_weights = None
            
    def regrid(self,data=None):
        if data is None:
            pass
        else:
            self.data = data
        data_regridded = Regridder(self.data)
        
        if self.forcing == 'era5':
            if self.name == 'prec':
                data_regridded = data_regridded * 1000
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'jra55':
            if self.name == 'prec':
                data_regridded = data_regridded
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'hist-mme' or self.forcing == 'aer-mme' or self.forcing == 'GHG-mme' or self.forcing == 'nat-mme':
            data_regridded = data_regridded
        else:
            if self.name=='prec' or self.name=='scaling' or self.name=='scaling_thermo' or self.name=='scaling_dynamic' or self.name=='scaling_interaction':
                data_regridded = data_regridded * 86400
            else:
                pass
        
        if self.mask == 'None':
            data_regridded = data_regridded
        elif self.mask == 'Global':
            data_regridded = data_regridded * mask_global
        elif self.mask == 'EU':
            data_regridded = data_regridded * mask_EU
        elif self.mask == 'EAS':
            data_regridded = data_regridded * mask_EAS
        elif self.mask == 'NA':
            data_regridded = data_regridded * mask_NA
        elif self.mask == 'SSA':
            data_regridded = data_regridded * mask_SSA
        elif self.mask == 'AU':
            data_regridded = data_regridded * mask_AU

        else:
            raise('ERROR!')
        
        return data_regridded

    def regrid_2(self,data=None):
        if data is None:
            pass
        else:
            self.data = data
        data_regridded = Regridder2(self.data)
        
        if self.forcing == 'era5':
            if self.name == 'prec':
                data_regridded = data_regridded * 1000
            else:
                data_regridded = data_regridded * 86400

        if self.forcing == 'jra55':
            if self.name == 'prec':
                data_regridded = data_regridded
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'hist-mme' or self.forcing == 'aer-mme' or self.forcing == 'GHG-mme' or self.forcing == 'nat-mme':
            data_regridded = data_regridded
        else:
            if self.name=='prec' or self.name=='scaling' or self.name=='scaling_thermo' or self.name=='scaling_dynamic' or self.name=='scaling_interaction':
                data_regridded = data_regridded * 86400
            else:
                pass
        
        if self.mask == 'None':
            data_regridded = data_regridded
        else:
            raise('ERROR!')
        
        return data_regridded

    def multiyear_mean(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        data_regridded_avg = data_regridded[var_name[self.name]].mean(axis = 0)
        return data_regridded_avg

    def trend_cal(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        slopes = xr.apply_ufunc(_compute_slope,
                                data_regridded,
                                vectorize=True,
                                dask='parallelized', 
                                input_core_dims=[['time']],
                                output_dtypes=[float],
                                )
        return slopes
    
    def trend_cal2(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        sig = xr.apply_ufunc(_compute_sig,
                             data_regridded,
                             vectorize=True,
                             dask='parallelized', 
                             input_core_dims=[['time']],
                             output_dtypes=[float],
                             )
        return sig

class Rx5dayCond():
    def __init__(self,name,forcing,src_id,run,mask='Global',pi=None,pi_time=None,for_sn=False,for_satellite=False):
        self.name = name
        self.forcing  = forcing
        self.src_id = src_id
        self.run  = run
        self.mask = mask
        self.pi   = pi
        self.pi_time = pi_time
        self.for_sn = for_sn
        self.for_satellite = for_satellite
        self.target_griddes = target_griddes
        
        if self.name not in full_name.keys():
            raise Exception('ERROR, name must be a string object belongs to {}'.format(list(full_name.keys())))
        if self.forcing not in frcs_src_run.keys():
            raise Exception('ERROR, forcing must be a string object belongs to {}'.format(list(frcs_src_run.keys())))
        if self.src_id not in frcs_src_run[self.forcing].keys():
            raise Exception('ERROR, for forcing {f}, src_id must be an element of {s}'.format(f = self.forcing, s=frcs_src_run[self.forcing].keys()))
        if self.run not in frcs_src_run[self.forcing][self.src_id]:
            raise Exception('ERROR, for forcing {f}, src_id {i}, run must be an element of {r}'.format(f=self.forcing,i=self.src_id,r=frcs_src_run[self.forcing][self.src_id]))
        if self.mask not in ['None','EU','EAS','SSA','Global','AU','NA']:
            raise Exception('ERROR, mask must be a string object belongs to [Global, EAS, EU, NA, AU, SSA]')
        
        if self.forcing != 'piControl':
            self.pi_time = None
            self.file = glob.glob(forcing_dir[self.forcing] + 'Rx5day/' + self.name +'_day'+'*' + src_id + '_' + '*' + run + '*.nc')[0]
        else:
            pi_num = len(glob.glob(forcing_dir[self.forcing] + 'Rx5day/' + self.name +'_day'+ '*' + src_id + '*' + run + '*.nc'))
            ens    = [filepath[-12:-3] for filepath in glob.glob(forcing_dir[self.forcing] + 'Rx5day/' + self.name +'*' + src_id + '*' + run + '*.nc')]
            if self.pi == None and self.pi_time == None:
                raise Exception('ERROR, for piControl forcing, provide \'pi\' parameter for initiation, which should be a time period {a} or the order of {b}'.format(a=ens,b=[i for i in range(pi_num)]))
            elif self.pi != None and self.pi_time == None:
                self.file = glob.glob(forcing_dir[self.forcing] + 'Rx5day/' + self.name +'_day_' + src_id + '*' + run + '*.nc')[self.pi]
            elif self.pi == None and self.pi_time != None:
                self.file = glob.glob(forcing_dir[self.forcing] + 'Rx5day/' + self.name +'_day_' + src_id + '*' + run + '*' + self.pi_time + '*.nc')[0]
            else:
                self.file = glob.glob(forcing_dir[self.forcing] + 'Rx5day/' + self.name +'_day_' + src_id + '*' + run + '*.nc')[self.pi]

        self.fullname = full_name[name]
        data0 = xr.open_dataset(self.file)
        if self.for_sn == False:
            self.data = data0.sel(time=slice('1960','2014'))
        elif self.for_sn == True:
            self.data = data0.sel(time=slice('1951','2014'))
        else:
            pass
        if self.forcing == 'era5' and self.for_satellite == True:
            self.data = data0.sel(time=slice('1979','2014'))

        if self.src_id == 'ACCESS-ESM1-5':
            self.data[var_name[self.name]][:,:,-1] = (self.data[var_name[self.name]][:,:,0]+self.data[var_name[self.name]][:,:,-2])/2
        else:
            pass

        self.dims = self.data.dims
        self.coords = self.data.coords

        self.origin_shape = self.data[var_name[self.name]].shape
        
        try:
            self.data = self.data.rename({'longitude':'lon','latitude':'lat'})
        except:
            pass
        try: 
            self.data = self.data.rename({'year':'time'})
        except:
            pass
        
        try:
            self.data['time'] = self.data['time'].dt.year
        except:
            pass
        
        self.area_weights = None
            
    def regrid(self,data=None):
        if data is None:
            pass
        else:
            self.data = data
        data_regridded = Regridder(self.data)
        
        if self.forcing == 'era5':
            if self.name == 'prec':
                data_regridded = data_regridded * 1000
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'jra55':
            if self.name == 'prec':
                data_regridded = data_regridded
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'hist-mme' or self.forcing == 'aer-mme' or self.forcing == 'GHG-mme' or self.forcing == 'nat-mme':
            data_regridded = data_regridded
        else:
            if self.name=='prec' or self.name=='scaling' or self.name=='scaling_thermo' or self.name=='scaling_dynamic' or self.name=='scaling_interaction':
                data_regridded = data_regridded * 86400
            else:
                pass
        
        if self.mask == 'None':
            data_regridded = data_regridded
        elif self.mask == 'Global':
            data_regridded = data_regridded * mask_global
        elif self.mask == 'EU':
            data_regridded = data_regridded * mask_EU
        elif self.mask == 'EAS':
            data_regridded = data_regridded * mask_EAS
        elif self.mask == 'NA':
            data_regridded = data_regridded * mask_NA
        elif self.mask == 'SSA':
            data_regridded = data_regridded * mask_SSA
        elif self.mask == 'AU':
            data_regridded = data_regridded * mask_AU

        else:
            raise('ERROR!')
        
        return data_regridded

    def regrid_2(self,data=None):
        if data is None:
            pass
        else:
            self.data = data
        data_regridded = Regridder2(self.data)
        
        if self.forcing == 'era5':
            if self.name == 'prec':
                data_regridded = data_regridded * 1000
            else:
                data_regridded = data_regridded * 86400

        if self.forcing == 'jra55':
            if self.name == 'prec':
                data_regridded = data_regridded
            else:
                data_regridded = data_regridded * 86400

        elif self.forcing == 'hist-mme' or self.forcing == 'aer-mme' or self.forcing == 'GHG-mme' or self.forcing == 'nat-mme':
            data_regridded = data_regridded
        else:
            if self.name=='prec' or self.name=='scaling' or self.name=='scaling_thermo' or self.name=='scaling_dynamic' or self.name=='scaling_interaction':
                data_regridded = data_regridded * 86400
            else:
                pass
        
        if self.mask == 'None':
            data_regridded = data_regridded
        else:
            raise('ERROR!')
        
        return data_regridded

    def multiyear_mean(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        data_regridded_avg = data_regridded[var_name[self.name]].mean(axis = 0)
        return data_regridded_avg

    def trend_cal(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        slopes = xr.apply_ufunc(_compute_slope,
                                data_regridded,
                                vectorize=True,
                                dask='parallelized', 
                                input_core_dims=[['time']],
                                output_dtypes=[float],
                                )
        return slopes
    
    def trend_cal2(self,data=None,data_regrid=None):
        if data is not None:
            data_regridded = self.regrid(data=data)
        if data is None and data_regrid is not None:
            data_regridded = data_regrid
        else:
            data_regridded = self.regrid()
        sig = xr.apply_ufunc(_compute_sig,
                             data_regridded,
                             vectorize=True,
                             dask='parallelized', 
                             input_core_dims=[['time']],
                             output_dtypes=[float],
                             )
        return sig