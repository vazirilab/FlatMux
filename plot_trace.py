'''
This generates most of the figure panels, but the firing sequence analysis and the correlation analysis are excluded.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib as mpl
import json
import os
from scipy import ndimage
from scipy.optimize import curve_fit
import tifffile
import h5py
import hdf5plugin
import colorsys
import matplotlib.ticker as ticker

from analysis_utils import detect_events, load_imag


inch = 25.4 # to mm
pt = 1/2.8346
lw_default = 0.2/pt

font_size_default = 8
mpl.rcParams['lines.linewidth'] = lw_default
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['font.size'] = font_size_default

x_label_pad_default = 0.2/pt
px_val_lower_clip = 1e-3
def downsample(x, N):
    if N > 1:
        if len(x)%N != 0:
            x = x[:(len(x)//N)*N]
        x1 = np.reshape(x, (len(x)//N,N))
        return np.average(x1, axis=1)
    else:
        return x

def f_gaussian_with_offset(x, x0, xc, A, offset):
    return offset + A*np.exp(-(x-x0)**2/xc**2)

def f_exp_with_offset(x, xc, A, offset):
    return offset + A*np.exp(x/xc)

def avrg_event_trace(dF_F, event_idx, N_frame_pre_post, N_frame_pre_post_avoid):
    event_trace = np.zeros(2*N_frame_pre_post+1, dtype=float)
    event_count = 0
    
    idx = np.arange(len(dF_F[0]))
    for n in range(len(dF_F)):
        if len(event_idx[n]) == 0:
            continue
        for m in range(len(event_idx[n])):
            if (event_idx[n][m] < N_frame_pre_post) | (event_idx[n][m] + N_frame_pre_post+1 > idx[-1]):
                continue
            idx_distance = np.abs(event_idx[n] - event_idx[n][m])
            idx_distance = idx_distance[idx_distance>0]
            if len(idx_distance) > 0:
                if np.min(idx_distance) > N_frame_pre_post_avoid:
                    event_count += 1
                    event_trace += dF_F[n][event_idx[n][m]-N_frame_pre_post:event_idx[n][m]+N_frame_pre_post+1]
            else:
                event_count += 1
                event_trace += dF_F[n][event_idx[n][m]-N_frame_pre_post:event_idx[n][m]+N_frame_pre_post+1]
    if event_count > 0:
        event_trace /= event_count
    return (event_trace, event_count)


if __name__ == '__main__':
    N_frame_type = 2
    rng = np.random.default_rng(0)
    
    folder_figs = './figs'
    os.makedirs(folder_figs, exist_ok=True)
    
    N_layer = 1  # Only 1 layer in this recording
    layer_label = ('', '')
    event_detect_idx_offset = 1
    N_frame_llhr_shift = -1
    
    fs = 751.864
    
    detect_llhr_std_threshold = 3.2
    detect_llhr_type_std = False
    
    
    # %% Load data
    f_result =  h5py.File('./Results/roi_traces_0.h5', 'r')
    N_results = len(f_result.keys())
    print(N_results)
    data = f_result['%d'%(N_results-1)]
    
    t = data['t'][:]
    Nt = len(t)
    F = data['rois_px_val'][:]
    roi_loglikehood_pmt_model = data['rois_log_likelihood_ratio_pmtmodel'][:]
    rois = data['rois'][:]
    rois_px_template = np.abs(data['init_px_template'])
    N_roi = rois.shape[0]
    N_cell_find_event = N_roi
    
    data_after_mc = load_imag('./demo_data/demo_data.tif')
    image_average = np.average(data_after_mc[16:16+2048], axis=0)
    
    Ny = image_average.shape[0]
    Nx = image_average.shape[1]
    x = np.arange(Nx) * 1.4  # 1.4 um per pixel
    y = np.arange(Ny) * 1.4
    X, Y = np.meshgrid(x, y)
    
    F_roll_mean = np.zeros(F.shape, dtype=np.float32)
    for n_ft in range(N_frame_type):
        F_roll_mean[:,n_ft::N_frame_type] = ndimage.uniform_filter1d(F[:,n_ft::N_frame_type], size=int(0.2*fs/2/2)*2+1, axis=1, mode='mirror')
    dF = F - F_roll_mean
    dF_std = ndimage.uniform_filter1d(dF**2, size=int(4*fs), axis=1, mode='mirror')**0.5
    dF_F = dF / F_roll_mean
    dF_std = dF / dF_std
    
    roi_layer = np.zeros(N_roi, dtype=int)-1
    for n_roi in range(N_roi):
        roi_sel = (rois[n_roi]>1e-3)
        roi_center_X = np.average(X[roi_sel])
        roi_layer[n_roi] = int(roi_center_X / (X.max()/N_layer))
    N_roi_layer = [np.sum(roi_layer==kk) for kk in range(N_layer)]

    # %% Resonance scanner correction
    tff = 0.7 # Temporal fill factor
    Nx_per_col = Nx // 2
    x_orig_per_col = np.arange(Nx_per_col)
    x_orig_per_col = x_orig_per_col - x_orig_per_col.mean()
    x_orig_per_col = x_orig_per_col / x_orig_per_col.max() # Normalize to (-1,1)
    x_corrected_per_col = np.sin(x_orig_per_col * np.pi/2 * tff)
    x_corrected_per_col = x_corrected_per_col / x_corrected_per_col.max()
    x_corrected_per_col -= x_corrected_per_col.min()
    x_corrected = np.concatenate((x_corrected_per_col, x_corrected_per_col+x_corrected_per_col[-1]+x_corrected_per_col[1]))
    x_corrected = x_corrected / x_corrected.max() * x.max()
    X_corrected, Y_corrected = np.meshgrid(x_corrected, y)
    
    # %% Event detection
    print('Event detection')
    def event_detection(thres):
        event_idx = []
        y_thres = []
        for n_roi in range(N_roi):
            y = roi_loglikehood_pmt_model[n_roi]
            detect_result = detect_events(y, height=thres, distance=5, \
                    N_frame_bound_rmv=100, detect_method='med_std', \
                    kernel_size={'mean': int(round(16*fs)), 'med': int(round(16*fs)), \
                        'std': int(round(16*fs)), 'step': int(round(0.5*fs))}, ret_peak_only=False)
            event_idx.append(detect_result['peaks'])
            y_thres.append(detect_result['y_thres'])
        
        return (event_idx, y_thres)
    
    event_idx, y_thres = event_detection(thres=detect_llhr_std_threshold)
    f_event_detected = h5py.File('./Results/event_detected.h5', 'w')
    for n_roi in range(len(event_idx)):
        roi_label_curr = '%d_%d' % (n_roi, roi_layer[n_roi])
        ds = f_event_detected.create_dataset(roi_label_curr, data=event_idx[n_roi]/fs)
    f_event_detected.close()
    
    # %% Example traces
    trace_roi_sel = 0
    
    N_ds = 2
    event_frame_offset = 1
    dF_F_plt_range = (-0.6, 0.5)
    llhr_plt_range = (-30, 32)
    dF_F_event_marker = -0.45
    llhr_event_marker = 10
    
    fig_trace, ax_trace = plt.subplots(2, 1, sharex=True)
    ax_trace[0].plot(t, dF_F[trace_roi_sel])
    ax_trace[1].plot(t, roi_loglikehood_pmt_model[trace_roi_sel])
    ax_trace[1].set_xlabel('Time (s)')
    ax_trace[0].set_ylabel('$\Delta F/F$')
    ax_trace[1].set_ylabel('Log likelihood ratio')

    # Plot the events
    ax_trace[0].plot(event_idx[trace_roi_sel]/fs, np.ones(len(event_idx[trace_roi_sel]))*dF_F_event_marker, ls='', marker='o', color='C1')
    ax_trace[1].plot(event_idx[trace_roi_sel]/fs, np.ones(len(event_idx[trace_roi_sel]))*llhr_event_marker, ls='', marker='o', color='C1')

    # Stimulus
    frame_idx_stim = np.loadtxt('./demo_data/demo_data_stim.txt')
    for n_stim in range(len(frame_idx_stim)):
        ax_trace[0].axvline(frame_idx_stim[n_stim]/fs, color='r', ls='--')
        ax_trace[1].axvline(frame_idx_stim[n_stim]/fs, color='r', ls='--')
        

    # %% Plot the FOV
    fig_fov, ax_fov = plt.subplots(1, 1)
    ax_fov.pcolormesh(X, Y, image_average, cmap='gray', shading='auto')
    ax_fov.set_aspect('equal')
    ax_fov.set_xlabel('X ($\mu$m)')
    ax_fov.set_ylabel('Y ($\mu$m)')
    ax_fov.set_title('Field of view')
    # Mark the selected roi
    X_roi = np.average(X[rois[trace_roi_sel]>1e-3])
    Y_roi = np.average(Y[rois[trace_roi_sel]>1e-3])
    ax_fov.plot(X_roi, Y_roi, ls='', marker='+', color='r')