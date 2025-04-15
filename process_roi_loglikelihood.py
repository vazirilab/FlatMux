import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import hdf5plugin
from scipy import ndimage
from scipy.optimize import minimize_scalar, curve_fit
import json
from joblib import Parallel, delayed
import zarr
import copy
import shutil

from skimage.feature import canny
from analysis_utils import load_imag, detect_events, blur_px_template

from pmt_noise_model import PMTNoise


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

pmt_noise = PMTNoise()

hdf5_compression_level = 3

px_val_lower_clip = 1e-3
init_with_nmf_background = False
peak_min_dist = 10 # Minimum distance (number of frames) between spikes
N_frame_bound_rmv = 100  # Those frames too close to the begining or the end of the recording do not have a good estimation of the baseline. They are removed.

## Matrix factorization
def f_px_update_likelihood(y, b, c, eta_bnd):
    ## y: observed value, 1d array
    ## b: baseline, 1d array
    ## c: time trace, 1d array
    ## eta: px_template
    ## eta_bnd: bounds for eta
    ## y = b (1 + eta*c)
    def f_neg_log_prob(eta):
        nlogp = -np.log(np.clip( pmt_noise.px_prob(x_mean=b*(1+eta*c), x=y), \
            1e-20, None))
        return np.sum(nlogp)
    
    res = minimize_scalar(f_neg_log_prob, bounds=eta_bnd, method='bounded')
    return res.x

def f_t_update_likelihood(y, b, eta):
    ## y = b (1 + eta*c)
    def f_neg_log_prob(c):
        nlogp = -np.log(np.clip( pmt_noise.px_prob(x_mean=b*(1+eta*c), x=y), \
            1e-20, None))
        # print(nlogp)
        return np.sum(nlogp)
    
    res = minimize_scalar(f_neg_log_prob, bounds=(-2,2), method='bounded')
    return res.x
    

if __name__ == '__main__':
    tmp_dir = os.path.join('./tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    with open('./Results/sep_roi_data_param.json') as json_file:
        data_sep_log = json.load(json_file)
        calib_gain = data_sep_log['calib_gain']
        calib_offset = data_sep_log['calib_offset']
        
    roi_process_log = {'sep_roi': data_sep_log}
    
    N_frame_type = 2
    fs = 751.864  # Bidirectional sampling rate
    N_job = 4
    dF_F_template_init = np.concatenate(([0.5], np.exp(-np.arange(5)/fs/2.3e-3)))
    dF_F_peak_expect = -0.4
    dF_F_peak_expect_bound = -0.8
    peak_ptr = 1 # In the temporal template, the peak of dF/F is at index 1.
    template_frame_idx = np.arange(len(dF_F_template_init)) - peak_ptr # Shift the index of the temporal template such that the peak dF/F is at index 0.

    N_frame_log_lh = len(dF_F_template_init)
    t_template_update = { \
            'log_llh_std_thres': 4, \
        }
    
    # log_llh_std_thres: detection threshold in the log-likelihood ratio test.
    # Minimum number of events to update the pixel template.
    # Update date: how does the new pixel tamplate mix with the old one?
    # peak_min_dist: Minimum distance (number of frames) between two spikes.
    # cutoff_brightness: Pixels with brightness lower than this value are discarded.
    # cutoff_val: set the discarded pixels to this value.
    # collapse_frametype: Do I need to consider forward and backward scanning separately?
    # solve_frame_pre and solve_frame_post: How many frames before and after the peak of the spike to solve for the pixel template.
    # solve_frame_idx_after and solve_frame_idx_before_finish: do not use the frames that are too close to the begining or the end of the recording.
    # blur_sigma and blur_kernel_radius: use a Gaussian filter to smothen a bit the pixel template (eta).
    px_template_update = { \
            'log_llh_std_thres': 2.75, \
            'log_llh_N_event_min': 30, \
            'update_rate': 0.25, \
            'peak_min_dist': peak_min_dist, \
            'cutoff_brightness': 0.05, \
            'cutoff_val': 0, \
            'collapse_frametype': False, \
            'init_outer_ring_size': 2, \
            'init_inner_ratio': 0.2, \
            'solve_frame_pre': 2,\
            'solve_frame_post': 6,\
            'solve_frame_idx_after': 200, \
            'solve_frame_idx_before_finish': 200, \
            'blur_sigma': 0.5, \
            'blur_kernel_radius': 1, \
            'subiter_N_t_norm': 25, \
        }
    
    event_detect_kernel_size = { \
            'mean':int(5*fs/N_frame_type), \
            'med':int(5*fs/N_frame_type), \
            'std':int(5*fs/N_frame_type), \
            'step':int(0.5*fs/N_frame_type) \
        }
    
    N_frame_event_det_boundary_avoid = 200

    N_iter = 1 # Global iterations: find potential spikes and relavant frames. Since the dataset is too small, it is set to 1.
    N_iter_sub = 2  # Sub iterations: NMF.
    
    data_label = 'noisy'
    
    roi_process_log['px_template_update'] = px_template_update
    roi_process_log['t_template_update'] = t_template_update
    roi_process_log['dF_F_template_init'] = dF_F_template_init
    roi_process_log['dF_F_peak_expect'] = dF_F_peak_expect
    roi_process_log['dF_F_peak_expect_bound'] = dF_F_peak_expect_bound
    roi_process_log['peak_ptr'] = peak_ptr
    roi_process_log['event_detect_kernel_size'] = event_detect_kernel_size
    roi_process_log['N_iter'] = N_iter
    roi_process_log['N_iter_sub'] = N_iter_sub
    roi_process_log['px_val_lower_clip'] = px_val_lower_clip
    roi_process_log['peak_min_dist'] = peak_min_dist
    roi_process_log['N_frame_bound_rmv'] = N_frame_bound_rmv
    

    data = load_imag('./demo_data/demo_data.tif')

    _, Ny, Nx = data.shape
    px_idx = np.arange(Nx*Ny).reshape((Ny, Nx))

    pmt_noise_lut_upper_bound = pmt_noise.lut_px_val_mean.max()
    roi_process_log['pmt_noise_lut_upper_bound'] = pmt_noise_lut_upper_bound

    
    ## ROI
    print('ROIs')
    if init_with_nmf_background:
        roi_px_weight = np.load('./Results/roi_px_weight_nmf.npz')
    else:
        roi_px_weight = np.load('./Results/roi_px_weight_0.npz')
    rois = roi_px_weight['roi_fp']
    N_roi = rois.shape[0]
    
    roi_data_dir = './Results/rois'
    ds_grp_roi = []
    ds_grp_roi_rm = []
    for n_roi in range(N_roi):
        fname = os.path.join(roi_data_dir, data_label+'_orig_%04d.zarr'%(n_roi))
        ds_grp_roi.append(zarr.open(fname, mode='r'))
        
        fname = os.path.join(roi_data_dir, data_label+'_rm_%04d.zarr'%(n_roi))
        ds_grp_roi_rm.append(zarr.open(fname, mode='r'))
        

    Nt = ds_grp_roi[0].shape[0]
    # Nt = int(fs*16/4)*4  # Only check a small part of the data for debugging
    
    frame_idx = np.arange(Nt, dtype=int)
    t = frame_idx / fs
    
    result = { \
            'image_average': np.stack([np.average(data[16+kk:16+1024+kk:N_frame_type], axis=0) for kk in range(N_frame_type)], axis=0), \
            't': t, \
            'rois': rois, \
            'rois_px_val': np.zeros((N_roi, Nt)), \
            'rois_px_val_rm': np.zeros((N_roi, Nt)), \
            'rois_log_likelihood_ratio_poisson': np.zeros((N_roi, Nt)), \
            'rois_log_likelihood_ratio_pmtmodel': np.zeros((N_roi, Nt)), \
        }
    result['image_average'] = (result['image_average']+calib_offset)/calib_gain
    
    px_remove_extra = result['image_average']<px_template_update['cutoff_brightness']

    ########
    ## 
    results = []
    def calc_roi_val(job_param):
        job_result = {}
        n_roi = job_param['n_roi']
        print('    ', n_roi)
        roi = job_param['roi']
        roi_sel = (roi==1)
        N_roi_px = np.sum(roi_sel)

        roi_data = ds_grp_roi[n_roi]
        roi_data_rm = ds_grp_roi_rm[n_roi]
        
        t_template = job_param['t_template']  # dF/F
        px_template = job_param['px_template'][:,roi_sel]  ## A template for the cells. Percentage. If all the photon detected are from the molecules on the cell membrane, then it is 1.
        px_template = np.expand_dims(px_template, axis=1)

        job_result['rois_log_likelihood_ratio_poisson'] = np.zeros((Nt))
        job_result['rois_log_likelihood_ratio_pmtmodel'] = np.zeros((Nt))
        job_result['rois_px_val'] = np.zeros((Nt))
        job_result['rois_px_val_rm'] = np.zeros((Nt))
        N_t_per_seg = 512
        N_ful_seg = Nt // N_t_per_seg


        for n_seg in range(N_ful_seg+1):
            # print('  >', n_seg, N_ful_seg)
            frame_beg = n_seg*N_t_per_seg
            frame_end = min(Nt, (n_seg+1)*N_t_per_seg)
            if frame_end <= frame_beg:
                break

            ## Average
            for n_ft in range(N_frame_type):
                roi_weight = px_template[n_ft]
                roi_weight = roi_weight / np.sum(roi_weight, axis=1, keepdims=True)
                job_result['rois_px_val'][frame_beg+n_ft:frame_end+n_ft:N_frame_type] = np.sum(roi_data[frame_beg+n_ft:frame_end+n_ft:N_frame_type]*roi_weight, axis=1)
                job_result['rois_px_val_rm'][frame_beg+n_ft:frame_end+n_ft:N_frame_type] = np.sum(roi_data_rm[frame_beg+n_ft:frame_end+n_ft:N_frame_type]*roi_weight, axis=1)

            ## log likelihood
            for n_shift in range(N_frame_log_lh):
                # print('    >>', n_shift)
                B_mean = np.clip(roi_data_rm[frame_beg+n_shift:frame_end+n_shift], px_val_lower_clip, None)
                S_mean = B_mean.copy()
                clip_sel = (B_mean==px_val_lower_clip)
                for n_ft in range(N_frame_type):
                    S_mean[n_ft::N_frame_type] *= \
                        (1 + t_template[n_shift] * px_template[(frame_beg+n_shift+n_ft)%N_frame_type])

                N_frame_curr_seg = len(B_mean)
                tmp = roi_data[frame_beg+n_shift:frame_beg+n_shift+N_frame_curr_seg]*np.log(np.clip(S_mean/B_mean, 1e-3,None)) - (S_mean-B_mean)
                tmp[clip_sel] = 0
                job_result['rois_log_likelihood_ratio_poisson'][frame_beg:frame_beg+N_frame_curr_seg] += np.sum(tmp, axis=1)
                # Use the PMT noise model
                S_mean = np.clip(S_mean, px_val_lower_clip, None)
                tmp = pmt_noise.px_log_likelyhood_ratio(B_mean, S_mean, roi_data[frame_beg+n_shift:frame_beg+n_shift+N_frame_curr_seg])
                tmp[clip_sel] = 0
                job_result['rois_log_likelihood_ratio_pmtmodel'][frame_beg:frame_beg+N_frame_curr_seg] += np.sum(tmp, axis=1)

        return job_result
    
    
    
    job_params = []
    for n_iter in range(N_iter):
        print('n_iter', n_iter)
        
        if n_iter > 0:
            job_params_prev = job_params
        job_params = []
        
        # print('Px update')
        for n_roi in range(N_roi):
            # print('    ', n_roi)
            job_params.append({ \
                        'n_roi': n_roi, \
                        'roi': rois[n_roi], \
                        'px_template': np.zeros((N_frame_type, Ny, Nx)), \
                        't_template': dF_F_template_init.copy(), \
                    })
            
        if n_iter == 0:
            for n_roi in range(N_roi):
                for n_ft in range(N_frame_type):
                    roi_curr = rois[n_roi].copy()
                    roi_edge = canny(roi_curr).astype(np.float32)
                    roi_edge_expand = ndimage.uniform_filter(roi_edge, size=2*px_template_update['init_outer_ring_size'])
                    job_params[n_roi]['px_template'][n_ft] = dF_F_peak_expect * rois[n_roi] * px_template_update['init_inner_ratio']
                    job_params[n_roi]['px_template'][n_ft][roi_edge_expand>0] = rois[n_roi][roi_edge_expand>0] * dF_F_peak_expect
                    job_params[n_roi]['px_template'][n_ft,px_remove_extra[n_ft]] = px_template_update['cutoff_val']*rois[n_roi][px_remove_extra[n_ft]]
                    
        else:
            ## Find events
            event_idx_prev_px_update = []
            event_idx_prev_t_update = []
            
            for n_roi in range(N_roi):
                px_sel = rois[n_roi]>0
                N_roi_px = np.sum(px_sel)
                event_idx_prev = detect_events( results[-1]['rois_log_likelihood_ratio_pmtmodel'][n_roi], \
                    height=px_template_update['log_llh_std_thres'], distance=peak_min_dist, N_frame_bound_rmv=N_frame_bound_rmv, \
                    detect_method='med_std', kernel_size=event_detect_kernel_size)
                
                if 'log_llh_N_event_max' in px_template_update.keys():
                    if len(event_idx_prev) > px_template_update['log_llh_N_event_max']:
                        event_idx_prev_val = results[-1]['rois_log_likelihood_ratio_pmtmodel'][n_roi][event_idx_prev]
                        event_idx_prev_val_srt = np.sort(event_idx_prev_val)[::-1]
                        event_idx_prev_cutoff_val = event_idx_prev_val_srt[px_template_update['log_llh_N_event_max']-1]
                        event_idx_prev = event_idx_prev[event_idx_prev_val>=event_idx_prev_cutoff_val]
                            
                if 'log_llh_N_event_min' in px_template_update.keys():
                    thres_tmp = px_template_update['log_llh_std_thres'] + 0
                    while len(event_idx_prev) < px_template_update['log_llh_N_event_min']:
                        thres_tmp -= 0.5
                        event_idx_prev = detect_events( results[-1]['rois_log_likelihood_ratio_pmtmodel'][n_roi], \
                            height=thres_tmp, distance=peak_min_dist, N_frame_bound_rmv=N_frame_bound_rmv, \
                            detect_method='med_std', kernel_size=event_detect_kernel_size)
                            
                        if len(event_idx_prev) > px_template_update['log_llh_N_event_min']:
                            event_idx_prev_val = results[-1]['rois_log_likelihood_ratio_pmtmodel'][n_roi][event_idx_prev]
                            event_idx_prev_val_srt = np.sort(event_idx_prev_val)[::-1]
                            event_idx_prev_cutoff_val = event_idx_prev_val_srt[px_template_update['log_llh_N_event_min']-1]
                            event_idx_prev = event_idx_prev[event_idx_prev_val>=event_idx_prev_cutoff_val]
                            
                event_idx_prev_px_update.append(event_idx_prev)
                
            for n_roi in range(N_roi):
                event_idx_prev = detect_events( results[-1]['rois_log_likelihood_ratio_pmtmodel'][n_roi], \
                    height=t_template_update['log_llh_std_thres'], distance=peak_min_dist, N_frame_bound_rmv=N_frame_bound_rmv, \
                    detect_method='med_std', kernel_size=event_detect_kernel_size)
                event_idx_prev_t_update.append(event_idx_prev)
                
                job_params[n_roi]['t_template'] = job_params_prev[n_roi]['t_template'].copy()
            
            
            px_template_curr_iter = np.zeros((N_roi, N_frame_type, Ny, Nx), dtype=np.float64)
            for n_roi in range(N_roi):
                px_template_curr_iter[n_roi] = job_params_prev[n_roi]['px_template']
            
            update_frame_sel = np.zeros((N_roi, N_frame_type, Nt), dtype=bool)
            update_frame_sel_idx = [[] for kk in range(N_roi)]
            update_frame_idx_norm = np.zeros((N_roi, px_template_update['subiter_N_t_norm']), dtype=np.int64)
            frame_idx = np.arange(Nt)
            for n_roi in range(N_roi):
                tmp = np.zeros(Nt, dtype=bool)
                for n_ev in range(len(event_idx_prev_px_update[n_roi])):
                    if (event_idx_prev_px_update[n_roi][n_ev] > px_template_update['solve_frame_idx_after']) & \
                            (event_idx_prev_px_update[n_roi][n_ev] < Nt-px_template_update['solve_frame_idx_before_finish']):
                        tmp[event_idx_prev_px_update[n_roi][n_ev]-px_template_update['solve_frame_pre']:\
                            event_idx_prev_px_update[n_roi][n_ev]+px_template_update['solve_frame_post']+1] = True
                for n_ft in range(N_frame_type):
                    update_frame_sel[n_roi, n_ft] = tmp & (frame_idx%N_frame_type==n_ft)
                    update_frame_sel_idx[n_roi].append(frame_idx[update_frame_sel[n_roi, n_ft]])
                
                tmp = event_idx_prev_px_update[n_roi][(event_idx_prev_px_update[n_roi]>px_template_update['solve_frame_idx_after']) \
                        & (event_idx_prev_px_update[n_roi]<Nt-px_template_update['solve_frame_idx_before_finish'])]
                llhr_peak = results[-1]['rois_log_likelihood_ratio_pmtmodel'][n_roi][tmp]
                llhr_peak_order = np.argsort(llhr_peak)
                update_frame_idx_norm[n_roi] = tmp[llhr_peak_order[-px_template_update['subiter_N_t_norm']:]]
            update_frame_idx_norm += peak_ptr
            
            # Matrix factorization
            for n_iter_sub in range(N_iter_sub):
                ## solve t
                def job_solve_t(param):
                    n_roi = param['n_roi']
                    n_ft = param['n_ft']
                    update_frame_sel_curr = param['update_frame_sel']
                    px_template_flat_curr = param['px_template_flat']
                    frame_idx_curr_sel = frame_idx[update_frame_sel_curr]
                    N_frame_curr = len(frame_idx_curr_sel)
                    ff = np.zeros(N_frame_curr, dtype=np.float64)
                    for n_t in range(N_frame_curr):
                        ff[n_t] = f_t_update_likelihood(\
                            y=ds_grp_roi[n_roi][frame_idx_curr_sel[n_t]], \
                            b=ds_grp_roi_rm[n_roi][frame_idx_curr_sel[n_t]], \
                            eta=px_template_flat_curr)
                    return {'n_roi': n_roi, 'n_ft': n_ft, 'f': ff}
                
                params_solve_t = []
                for n_roi in range(N_roi):
                    for n_ft in range(N_frame_type):
                        roi_sel = (rois[n_roi] > 0)
                        params_solve_t.append({ \
                                'n_roi': n_roi, \
                                'n_ft': n_ft, \
                                'update_frame_sel': update_frame_sel[n_roi, n_ft], \
                                'px_template_flat': px_template_curr_iter[n_roi, n_ft][roi_sel]
                            })
                        
                outputs_solve_t = Parallel(n_jobs=N_job, backend='multiprocessing', verbose=0) \
                    (delayed(job_solve_t)(params_solve_t[n]) for n in range(len(params_solve_t)))
                
                update_ff = []
                for n_roi in range(N_roi):
                    update_ff.append([])
                    for n_ft in range(N_frame_type):
                        ff = outputs_solve_t[n_roi*N_frame_type+n_ft]['f'].copy()
                        update_ff[n_roi].append(ff)
                    
                    ## Normalize it, to match the dF/F (ff ~ 1 at the spike)
                    ff_val_events = []
                    for n_ev in range(px_template_update['subiter_N_t_norm']):
                        frame_idx_curr = update_frame_idx_norm[n_roi][n_ev]
                        ft_curr = frame_idx_curr % N_frame_type
                        idx_in_ff = np.where(update_frame_sel_idx[n_roi][ft_curr]==frame_idx_curr)
                        ff_val_events.append(update_ff[n_roi][ft_curr][idx_in_ff])
                    
                    ff_val_events_avrg = np.average(ff_val_events)
                    for n_ft in range(N_frame_type):
                        update_ff[n_roi][n_ft] = update_ff[n_roi][n_ft] / ff_val_events_avrg
                    
                
                ## Solve pixel template
                def job_solve_px(param):
                    n_roi = param['n_roi']
                    n_ft = param['n_ft']
                    n_px = param['n_px']
                    ff_px = param['ff']
                    update_frame_sel_curr = param['update_frame_sel']
                    frame_idx_curr_sel = frame_idx[update_frame_sel_curr]
                    N_frame_curr = len(frame_idx_curr_sel)
                    y_px = ds_grp_roi[n_roi][frame_idx_curr_sel,n_px]
                    y0_px = ds_grp_roi_rm[n_roi][frame_idx_curr_sel,n_px]
                    
                    if dF_F_peak_expect_bound < 0:
                        eta = f_px_update_likelihood(y_px, y0_px, ff_px, \
                            [dF_F_peak_expect_bound, -1e-3])
                    else:
                        eta = f_px_update_likelihood(y_px, y0_px, ff_px, \
                            [1e-3, dF_F_peak_expect_bound])
                    return eta
                
                params_solve_px = []
                for n_roi in range(N_roi):
                    roi_sel = (rois[n_roi] > 0)
                    N_px_curr = len(rois[n_roi][roi_sel])
                    for n_ft in range(N_frame_type):
                        for n_px in range(N_px_curr):
                            params_solve_px.append({ \
                                    'n_roi': n_roi, \
                                    'n_ft': n_ft, \
                                    'n_px': n_px, \
                                    'ff': update_ff[n_roi][n_ft], \
                                    'update_frame_sel': update_frame_sel[n_roi, n_ft], \
                                })
                            
                outputs_solve_px = Parallel(n_jobs=N_job, backend='multiprocessing', verbose=0) \
                    (delayed(job_solve_px)(params_solve_px[n]) for n in range(len(params_solve_px)))
                
                px_template_prev_iter = px_template_curr_iter.copy()
                px_template_curr_iter = np.zeros((N_roi, N_frame_type, Ny, Nx), dtype=np.float64)
                n_output = 0
                for n_roi in range(N_roi):
                    for n_ft in range(N_frame_type):
                        roi_sel = (rois[n_roi] > 0)
                        N_px_curr = len(rois[n_roi][roi_sel])
                        px_template_curr_tmp = np.zeros(N_px_curr, dtype=np.float64)
                        for n_px in range(N_px_curr):
                            px_template_curr_tmp[n_px] = outputs_solve_px[n_output]
                            n_output += 1
                        
                        
                        if dF_F_peak_expect > 0:
                            px_template_curr_tmp = np.clip(px_template_curr_tmp, 1e-3, dF_F_peak_expect_bound)
                        else:
                            px_template_curr_tmp = np.clip(px_template_curr_tmp, dF_F_peak_expect_bound, -1e-3)
                        
                        px_template_curr_tmp_2d = np.zeros((Ny, Nx), dtype=np.float64)
                        px_template_curr_tmp_2d[roi_sel] = px_template_curr_tmp
                        px_template_curr_iter[n_roi, n_ft] = blur_px_template(rois[n_roi], px_template_curr_tmp_2d, \
                            blur_sigma=px_template_update['blur_sigma'], blur_kernel_radius=px_template_update['blur_kernel_radius'])
                
                
                px_template_curr_iter = px_template_update['update_rate'] * px_template_curr_iter \
                    + (1-px_template_update['update_rate']) * px_template_prev_iter
            
            for n_roi in range(N_roi):
                job_params[n_roi]['px_template'] = px_template_curr_iter[n_roi]
            
                if px_template_update['collapse_frametype']:
                    if N_frame_type == 2:
                        job_params[n_roi]['px_template'][0] = np.average(job_params[n_roi]['px_template'], axis=0)
                        job_params[n_roi]['px_template'][1] = job_params[n_roi]['px_template'][0]
                    if N_frame_type == 4:
                        job_params[n_roi]['px_template'][0] = (job_params[n_roi]['px_template'][0]+job_params[n_roi]['px_template'][3])/2
                        job_params[n_roi]['px_template'][1] = (job_params[n_roi]['px_template'][1]+job_params[n_roi]['px_template'][2])/2
                        job_params[n_roi]['px_template'][2] = job_params[n_roi]['px_template'][1]
                    job_params[n_roi]['px_template'][3] = job_params[n_roi]['px_template'][0]
                    
                for n_ft in range(N_frame_type):
                    job_params[n_roi]['px_template'][n_ft,px_remove_extra[n_ft]] = px_template_update['cutoff_val']*rois[n_roi][px_remove_extra[n_ft]]
                
            
        print('calc')
        output = Parallel(n_jobs=N_job, backend='multiprocessing', verbose=5) \
            (delayed(calc_roi_val)(job_params[n]) for n in range(len(job_params)))
        
        ## Gether results and save
        results.append(copy.deepcopy(result))
        for n_roi in range(N_roi):
            for key in output[n_roi].keys():
                results[-1][key][n_roi] = output[n_roi][key]
        
        fname_result_pre = './Results/roi_traces_0'
        fname_result = fname_result_pre + '.h5'
        if n_iter == 0:
            if os.path.exists(fname_result):
                os.remove(fname_result)
            
            json_string = json.dumps(roi_process_log, indent=4, cls=NumpyEncoder)
            with open(fname_result_pre+'.json', 'w') as json_file:
                json_file.write(json_string)
    
            
        with h5py.File(fname_result, 'a') as f_result:
            grp = f_result.create_group('%d'%n_iter)
            grp.attrs['N_roi'] = N_roi
            for key in results[-1].keys():
                grp.create_dataset(key, data=results[-1][key], **hdf5plugin.Zstd(clevel=hdf5_compression_level))
            
            init_px_template = np.zeros((N_roi, N_frame_type, Ny, Nx), dtype=np.float32)
            for n_roi in range(N_roi):
                init_px_template[n_roi] = job_params[n_roi]['px_template']
                if n_iter > 0:
                    grp.create_dataset('init_event_idx_px_update_%06d'%(n_roi), data=event_idx_prev_px_update[n_roi], **hdf5plugin.Zstd(clevel=hdf5_compression_level))
                    grp.create_dataset('init_event_idx_t_update_%06d'%(n_roi), data=event_idx_prev_t_update[n_roi], **hdf5plugin.Zstd(clevel=hdf5_compression_level))
            grp.create_dataset('init_px_template', data=init_px_template, **hdf5plugin.Zstd(clevel=hdf5_compression_level))
            grp.create_dataset('init_dF_F_t_template', data=job_params[0]['t_template'], **hdf5plugin.Zstd(clevel=hdf5_compression_level))
            # grp.create_dataset('rois', data=rois)
            
        
        
