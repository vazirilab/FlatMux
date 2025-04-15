'''
Separate data belonging to different neurons.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy import ndimage
import json
from joblib import Parallel, delayed
import zarr
from numcodecs import Blosc, Zstd
import shutil


from analysis_utils import load_roi_sel, roi_sel_adjust, gain_calib_shotnoise, load_imag, rmv_roi_boundary, rmv_roi_px
from pmt_noise_model import PMTNoise

N_frame_avrg_pre_post = 25  # To estimate the base line of the pixel af a certain frame. I perform average over a time window of frames [current_frame-N_frame_avrg_pre_post, current_frame+N_frame_avrg_pre_post].
roi_val_cutoff = 1e-10  # epsilon.

if __name__ == '__main__':
    roi_out_dir = os.path.join('./Results/rois/')
    if os.path.exists(roi_out_dir):
        shutil.rmtree(roi_out_dir, ignore_errors=True)
    os.makedirs(roi_out_dir, exist_ok=True)
    
    fs = 751.864
    N_frame_type = 2  # Galvo forward and backward scanning..
    
    data_param = { \
            'data_path': './demo_data/demo_data.tif', \
            'roi_path': './Results/imag_avrg_mc_16bit_maxcontrast_cp_masks.png', \
            'roi_param': {}, \
            'roi_adjust_param': {'expand':1}, \
            'label': 'noisy', \
            'N_frame_avrg_pre_post': N_frame_avrg_pre_post, \
            'roi_val_cutoff': roi_val_cutoff, \
        }
    
    data = load_imag(data_param['data_path'])
    
    Nt, Ny, Nx = data.shape
    px_idx = np.arange(Nx*Ny).reshape((Ny, Nx))
    Nt = (Nt//N_frame_type) * N_frame_type
    brightness_val_low_remove = 0.05 # Discard those pixels that are too dim. They basically do not contribute to the signal.
    
    ## Calibration using shotnoise
    (calib_gain, calib_offset) = gain_calib_shotnoise(data[4:, 10:-10, 10:-10].astype(np.float32), sel_perc=99.99, plot_result=False)
    calib_offset += 100
    tmp = (data+calib_offset)/calib_gain
    image_average = np.average(tmp.astype(np.float32), axis=0)
    
    pmt_noise = PMTNoise()
    pmt_noise_lut_upper_bound = pmt_noise.lut_px_val_mean.max()
    
    frame_idx = np.arange(Nt, dtype=int)
    t = frame_idx / fs
    
    ## Log
    data_param['brightness_val_low_remove'] = brightness_val_low_remove
    data_param['calib_gain'] = calib_gain
    data_param['calib_offset'] = calib_offset
    
    json_string = json.dumps(data_param, indent=4)
    with open('./Results/sep_roi_data_param.json', 'w') as json_file:
        json_file.write(json_string)
    
    ## ROI
    print('ROIs')
    rois = load_roi_sel(data_param['roi_path'], param=data_param['roi_param'])
    rois[rois>roi_val_cutoff] = 1
    rois = rmv_roi_boundary(rois)
    for n_roi in range(len(rois)):
        rois[n_roi,image_average<brightness_val_low_remove] = 0
    
    np.savez_compressed('./Results/roi_px_weight_0.npz', roi_fp=rois)
    
    N_roi = rois.shape[0]
    
    for n_roi in range(N_roi):
        rois[n_roi] = roi_sel_adjust( \
            roi_val=rois[n_roi], imag=image_average, param=data_param['roi_adjust_param'])
        
    
    ########
    # Group pixels within a same ROI into a new file for faster access for the rest of the processings
    print('Group ROIs')
    f_grp_roi = []  # Raw
    ds_grp_roi = []
    f_grp_roi_rm = []  # Rolling mean
    ds_grp_roi_rm = []
    ds_grp_roi_fp = [] # footprint
    
    for n_roi in range(N_roi):
        roi = rois[n_roi]
        roi_sel = (roi>roi_val_cutoff)
        N_roi_px = np.sum(roi_sel)
        
        # Raw data
        fname_roi = os.path.join(roi_out_dir, data_param['label']+'_orig_%04d.zarr'%(n_roi))
        
        ds_grp_roi.append( \
                zarr.open(fname_roi, mode='w', shape=(Nt,N_roi_px), \
                chunks=(32, N_roi_px), dtype=np.float32, \
                synchronizer=zarr.ThreadSynchronizer(), \
                compressor = Zstd(level=1))
            )
            
        fname_roi = os.path.join(roi_out_dir, data_param['label']+'_rm_%04d.zarr'%(n_roi))
        
        # Temporal averaged data for baseline estimation
        ds_grp_roi_rm.append( \
                zarr.open(fname_roi, mode='w', shape=(Nt,N_roi_px), \
                chunks=(32, N_roi_px), dtype=np.float32, \
                synchronizer=zarr.ThreadSynchronizer(), \
                compressor = Zstd(level=1))
            )
        
        # Footprint
        fname_roi = os.path.join(roi_out_dir, data_param['label']+'_fp_%04d.zarr'%(n_roi))
        ds_grp_roi_fp.append( \
                zarr.open(fname_roi, mode='w', shape=(Ny,Nx), dtype=np.int32, \
                compressor = Zstd(level=1))
            )
    
    for n_roi in range(N_roi):
        tmp = np.zeros((Ny,Nx))
        roi = rois[n_roi]
        roi_sel = (roi>roi_val_cutoff)
        tmp[roi_sel] = 1
        
        ds_grp_roi_fp[n_roi][:] = tmp
        
    
    N_t_per_seg = 256
    N_full_seg = Nt // N_t_per_seg
    t0 = time.time()
    
    
    def group_roi(job_param):
        n_seg = job_param['n_seg']
        N_frame_extra_pre_post = int(N_frame_avrg_pre_post*N_frame_type)
        frame_beg = n_seg*N_t_per_seg
        frame_beg_rm = frame_beg-N_frame_extra_pre_post
        frame_beg_rm_clip = max(0, frame_beg_rm)
        frame_end = min(Nt, (n_seg+1)*N_t_per_seg)
        frame_end_rm = frame_end+N_frame_extra_pre_post
        frame_end_rm_clip = min(Nt, frame_end_rm)
        if frame_end <= frame_beg:
            return -1
        
        tmp = (data[frame_beg:frame_end]+calib_offset)/calib_gain
        # I need some extra frames before and after the current frames to do rolling mean
        tmp_rm = np.zeros((tmp.shape[0]+2*N_frame_extra_pre_post, tmp.shape[1], tmp.shape[2]))
        
        tmp_rm[N_frame_extra_pre_post:N_frame_extra_pre_post+tmp.shape[0]] = tmp
        tmp_pre = (data[frame_beg_rm_clip:frame_beg]+calib_offset)/calib_gain
        tmp_post = (data[frame_end:frame_end_rm_clip]+calib_offset)/calib_gain
        
        tmp_rm[N_frame_extra_pre_post-len(tmp_pre):N_frame_extra_pre_post] = tmp_pre
        tmp_rm[len(tmp_rm)-N_frame_extra_pre_post:len(tmp_rm)-N_frame_extra_pre_post+len(tmp_post)] = tmp_post
        
        for n_roi in range(N_roi):
            roi_sel = ds_grp_roi_fp[n_roi][:] == 1
            ds_grp_roi[n_roi][frame_beg:frame_end] = tmp[:,roi_sel]
            tmp_rm_roi = tmp_rm[:,roi_sel]
            tmp_rm_result = np.zeros(tmp_rm_roi.shape, dtype=np.float32)
            for nft in range(N_frame_type):
                tmp_rm_result[nft::N_frame_type] = ndimage.uniform_filter1d(tmp_rm_roi[nft::N_frame_type], size=2*N_frame_avrg_pre_post+1, axis=0, mode='mirror')
            ds_grp_roi_rm[n_roi][frame_beg:frame_end] = tmp_rm_result[N_frame_extra_pre_post:-N_frame_extra_pre_post]
            
        return 0
    
    
    jobs_grp_rois_param = [{'n_seg': n_seg} for n_seg in range(N_full_seg+1)]
    
    job_grp_rois_results = Parallel(n_jobs=8, backend='multiprocessing', verbose=5) \
        (delayed(group_roi)(jobs_grp_rois_param[n]) for n in range(len(jobs_grp_rois_param)))
    
    
    print('Roi dataset groupped. It took %.3f sec ' % (time.time()-t0))
    print('Fix the rolling mean of the begining and ending frames')
    # They just make the downstream code more stable. I will discard those frames affected by the rolling mean for spike inference.
    for n_roi in range(N_roi):
        for n_pp in range(N_frame_avrg_pre_post):
            ds_grp_roi_rm[n_roi][n_pp*N_frame_type:(n_pp+1)*N_frame_type] = ds_grp_roi_rm[n_roi][N_frame_avrg_pre_post*N_frame_type:(N_frame_avrg_pre_post+1)*N_frame_type]
            ds_grp_roi_rm[n_roi][Nt-(n_pp+1)*N_frame_type:Nt-n_pp*N_frame_type] = ds_grp_roi_rm[n_roi][Nt-(N_frame_avrg_pre_post+1)*N_frame_type:Nt-N_frame_avrg_pre_post*N_frame_type]
    
