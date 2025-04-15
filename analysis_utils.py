'''
Some libraries to process the data.
Jingkun Guo
'''
import numpy as np
import h5py
import copy
import os
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, uniform_filter, uniform_filter1d, median_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import tifffile
from PIL import Image as PIL_Img
from sklearn.mixture import GaussianMixture
import jax
jax.default_device = jax.devices('cpu')[0]
import jax.numpy as jnp
from jax import grad, jit, vmap
import zarr
import cv2
from numba import njit
import copy


# How x and y are indexed in the scanimage files
si_img_idx_rs = 1  # Resonance scanner
si_img_idx_nrg = 0  # (Non-resonance) galvo

hdf5_compression_level = 4


# %% Motion correction
def apply_shift_iteration(img, shift, border_nan:bool=False, border_type=cv2.BORDER_REFLECT):
    # Adapted from CAIMAN, motion_correction.py
    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.nanmin(img), np.nanmax(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i),
                                 flags=cv2.INTER_NEAREST, borderMode=border_type), min_, max_)
    # The flags was cv2.INTER_CUBIC. Other options: INTER_LINEAR, INTER_NEAREST
    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shift)).astype(int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shift)).astype(int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h-1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w-1, np.newaxis]

    return img


def save_tiff(fname, data, gamma=1, \
    save_normal=True, save_max_contrast_uint16=True, \
    save_max_contrast_uint8=True):
    fname_inside = copy.deepcopy(fname)
    if fname[-4:] == '.tif':
        fname_inside = fname[:-4]
    elif fname[-5:] == '.tiff':
        fname_inside = fname[:-5]
    else:
        fname_inside = fname
    data_max_contrast = (data-data.min())/(data.max()-data.min())
    if save_normal:
        tifffile.imwrite(fname_inside+'.tif', data.astype(np.uint16))
    if save_max_contrast_uint16:
        tifffile.imwrite(fname_inside+'_16bit_maxcontrast.tif', \
                (data_max_contrast * (2**16-1)).astype(np.uint16) \
            )
    if save_max_contrast_uint8:
        tifffile.imwrite(fname_inside+'_8bit_maxcontrast.tif', \
                (data_max_contrast * (2**8-1)).astype(np.uint8)
            )
    if gamma != 1:
        data_max_contrast_gamma = data_max_contrast**gamma
        tifffile.imwrite(fname_inside+'gamma%.1f_16bit_maxcontrast.tif'%(gamma), \
                (data_max_contrast_gamma * (2**16-1)).astype(np.uint16) \
            )




def load_roi_sel(fpath, param={}):
    if os.path.isdir(fpath):
        return load_roi_sel_folder(fpath, param)
    if 'cp_mask' in fpath:
        return load_roi_sel_cellpose(fpath, param)
    

def load_roi_sel_cellpose(fpath, param):
    with PIL_Img.open(fpath) as img:
            img_arr = np.asarray(img)
    N_roi = int(img_arr.max()) - 1
    area_sel_list = np.zeros((N_roi, img_arr.shape[0], img_arr.shape[1]))
    for n in range(N_roi):
        area_sel_list[n] += (img_arr == (n+1)).astype(float)
    area_sel_list /= np.sum(area_sel_list, axis=(1,2), keepdims=True)
    return area_sel_list

def load_roi_sel_folder(folder_name, param):
    if 'fname_prefix' in param.keys():
        fname_prefix = param['fname_prefix']
    else:
        fname_prefix = ''
    area_sel_list = []
    area_sel_fname_list = []
    for root, dirs, files in os.walk(folder_name):
        for fname in files:
            if (len(fname_prefix)>0) & (fname[:len(fname_prefix)]!=fname_prefix):
                continue
            if fname[-4:] == '.png':
                area_sel_fname_list += [fname]
                with PIL_Img.open(os.path.join(root, fname)) as img:
                    img_arr = np.asarray(img)
                area_sel_list += [img_arr[:,:,3].astype(np.float32)/np.sum(img_arr[:,:,3])]
            if fname[-4:] == '.tif':
                area_sel_fname_list += [fname]
                with tifffile.TiffFile(os.path.join(root, fname)) as f_tif:
                    img_arr = f_tif.asarray().astype(np.float32)
                    img_arr /= np.sum(img_arr)
                    area_sel_list += [img_arr]
    # print(area_sel_fname_list)
    sorted_args = np.argsort(area_sel_fname_list)
    area_sel_list = np.array(area_sel_list)
    area_sel_list = area_sel_list[sorted_args]
    
    area_sel_fname_list = np.array(area_sel_fname_list)
    print(area_sel_fname_list[sorted_args])
    print(sorted_args)
    
    return np.stack(area_sel_list, axis=0)

def roi_sel_adjust(roi_val, imag, param):
    roi_val = roi_val.copy()
    roi_sel_orig = (roi_val>1e-8)
    if 'brightness_ratio' in param.keys():
        sel_brightness_cut = imag[roi_val].max()*param['brightness_ratio']
        roi_val[imag<sel_brightness_cut] = 0
    if 'expand' in param.keys():
        roi_val = uniform_filter(roi_val, size=param['expand'], mode='constant', cval=0.)
        roi_val[roi_val>1e-8] = 1
        roi_val /= np.sum(roi_val)
        
    elif ('sklearn' in param.keys()):
        if param['sklearn']:
            imag_flat = imag.reshape(imag.size)
            roi_val_flat = roi_val.reshape(roi_val.size)
            px_idx_arr = np.arange(imag_flat.size)
            px_sel = px_idx_arr[roi_val_flat>1e-8]
            px_brightness = imag_flat[px_sel]
            gm_label = GaussianMixture(n_components=2, random_state=0).fit_predict(px_brightness.reshape((px_brightness.size,1)))
            avrg_brightness_0 = np.average(px_brightness[gm_label==0])
            avrg_brightness_1 = np.average(px_brightness[gm_label==1])
            if avrg_brightness_0<avrg_brightness_1:
                label_sel = 1
            else:
                label_sel = 0
            px_sel_new = px_sel[gm_label==label_sel]
            roi_val_new_flat = np.zeros(imag_flat.size)
            roi_val_new_flat[px_sel_new] = 1
            roi_val_new_flat /= np.sum(roi_val_new_flat)
            if 'mix_old_ratio' in param.keys():
                roi_val_new_flat = roi_val_new_flat*(1-param['mix_old_ratio']) + roi_val_flat*param['mix_old_ratio']
                roi_val = roi_val_new_flat.reshape(imag.shape).copy()
            # return roi_val_new_flat.reshape(imag.shape)
    elif ('optim_shotnoise_model' in param.keys()):
        if param['optim_shotnoise_model']:
            ## The image is normalized to (average) photon number
            # I select the brightest pixel to fix the scaling of the roi
            imag_roi_sel = imag[roi_sel_orig]
            idx_mat = np.arange(imag.size).reshape(imag.shape)
            idx_mat_roi_sel = idx_mat[roi_sel_orig]
            bg_val = imag_roi_sel.min()#*0.5
            a = imag_roi_sel - bg_val  # cell value, probably
            v = imag_roi_sel.copy()  # total variance expected from shotnoise
            
            idx_a_max = np.argmax(a)
            a_max = a[idx_a_max]
            v_a_max = v[idx_a_max]
            idx_arr = np.arange(len(a))
            a_excl_max = a[idx_arr!=idx_a_max]
            v_excl_max = v[idx_arr!=idx_a_max]
            
            s0 = np.ones(len(imag_roi_sel)-1, dtype=float)
            def nsr_est(s): # Inverse of SNR. Minimize it
                # return jnp.sqrt(jnp.dot(s**2, v))/jnp.dot(s, a)
                return jnp.sqrt(jnp.dot(s**2, v_excl_max)+v_a_max)/(jnp.dot(s, a_excl_max)+a_max)
            nsr_est_jit = jit(nsr_est)
            res = minimize(nsr_est_jit, s0, method='BFGS', jac=grad(nsr_est_jit), \
                tol=1e-6, options={'disp':True})
            roi_val_new = np.ones(len(a))
            roi_val_new[idx_arr!=idx_a_max] = res.x
            
            roi_val = np.zeros(roi_val.shape, dtype=float)
            roi_val[roi_sel_orig] = roi_val_new
            roi_val[roi_sel_orig] /= np.sum(roi_val)
    
    if 'blur' in param.keys():
        # print(param['blur'])
        for nb in range(len(param['blur'])):
            if param['blur'][nb] > 0:
                roi_val = gaussian_filter(roi_val, param['blur'][nb])
            roi_val[~roi_sel_orig] = 0
            roi_val /= np.sum(roi_val)
    return roi_val
    
def rmv_roi_boundary(rois_orig, col=2, N_px=[0, 1]):
    N_roi_orig = len(rois_orig)
    rois = []
    cutoff = 1e-10
    N_rs_1c = rois_orig.shape[2] // col
    for n in range(N_roi_orig):
        if N_px[0] > 0:
            if np.sum(rois_orig[n,0:N_px[0]]) > cutoff:
                continue
            if np.sum(rois_orig[n,-N_px[0]:]) > cutoff:
                continue
        if N_px[1] > 0:
            if (np.sum(rois_orig[n,:,:N_px[1]]) > cutoff) | (np.sum(rois_orig[n,:,-N_px[1]:]) > cutoff):
                continue
            if col == 2:
                if (np.sum(rois_orig[n,:,N_rs_1c:N_rs_1c+N_px[1]]) > cutoff) | (np.sum(rois_orig[n,:,N_rs_1c-N_px[1]:N_rs_1c]) > cutoff):
                    continue
        rois.append(rois_orig[n,:,:])
    rois = np.stack(rois, axis=0)
    return rois

def rmv_roi_px(roi_orig, px_blacklist):
    N_pxbl = len(px_blacklist)
    rois = roi_orig.copy()
    for n_pxbl in range(len(px_blacklist)):
        rois[:,px_blacklist[n_pxbl][0], px_blacklist[n_pxbl][1]] = 0
    return rois

def load_imag(fname, mmap_mode='r', hdf5_load2mem=False):
    if fname[-5:] == '.hdf5':
        # os.environ['HDF5_USE_FILE_LOCKING'] = 'TRUE'
        f_data = h5py.File(fname, 'r', rdcc_nbytes=1024**3)
        if 'mov' in f_data.keys():
            ds = f_data['mov']
        elif 'data' in f_data.keys():
            ds = f_data['data']
        if hdf5_load2mem:
            data = ds[:]
            f_data.close()
            return data
        else:
            return (ds, f_data)
    elif fname[-4:] == '.tif':
        data = tifffile.TiffFile(fname).asarray().astype(np.float32)
    elif fname[-5:] == '.mmap':
        ## See caiman
        ## path.py, decode_mmap_filename_dict
        ret = {}
        _, fn = os.path.split(fname)
        fn_base, _ = os.path.splitext(fn)
        fpart = fn_base.split('_')[1:] # First part will (probably) reference the datasets
        for field in ['d1', 'd2', 'd3', 'order', 'frames']:
            # look for the last index of fpart and look at the next index for the value, saving into ret
            for i in range(len(fpart) - 1, -1, -1): # Step backwards through the list; defensive programming
                if field == fpart[i]:
                    if field == 'order': # a string
                        ret[field] = fpart[i + 1] # Assume no filenames will be constructed to end with a key and not a value
                    else: # numeric
                        ret[field] = int(fpart[i + 1]) # Assume no filenames will be constructed to end with a key and not a value
        if fpart[-1] != '':
            ret['T'] = int(fpart[-1])
        if 'T' in ret and 'frames' in ret and ret['T'] != ret['frames']:
            print(f"D: The value of 'T' {ret['T']} differs from 'frames' {ret['frames']}")
        if 'T' not in ret and 'frames' in ret:
            ret['T'] = ret['frames']
        
        def prepare_shape(shape_in):
            return tuple(map(lambda x: np.uint64(x), shape_in))
        Yr = np.memmap(fname, mode=mmap_mode, \
            shape=prepare_shape((ret['d1']*ret['d2']*ret['d3'], ret['frames'])), \
            dtype=np.float32, order=ret['order'])
        ## My convention: the 1st axis is typically the time
        data = np.reshape(Yr.T, \
            [ret['frames'], ret['d1'], ret['d2']], order='F')
    elif '.zarr' in fname:
        data = zarr.open(fname, mode='r')
    return data


def gain_calib_shotnoise(data, sel_perc=95, plot_result=False):
    data_avrg = np.average(data, axis=0)
    data_var = np.var(data, axis=0)
    ratio = (data_var-data_var.min())/(data_avrg-data_avrg.min()+1e-10)
    data_sel = (ratio < np.percentile(ratio,sel_perc))
    
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(data_avrg[data_sel].flatten(), data_var[data_sel].flatten(), deg=1).convert()
    
    gain=p.coef[1]
    offset = p.coef[0]/p.coef[1]
    
    if plot_result:
        fig_gain_N_pht = plt.figure()
        ax_gain_N_pht = fig_gain_N_pht.add_subplot(1,1,1)
        ax_gain_N_pht.plot(data_avrg[data_sel].flatten(), data_var[data_sel].flatten(), color='C0', ls='', marker='.', mec='C0', mfc=(0,0,0,0))
        ax_gain_N_pht.plot(data_avrg[~ data_sel].flatten(), data_var[~ data_sel].flatten(), color='C1', ls='', marker='.', mec='C1', mfc=(0,0,0,0))
        xy1 = (0,p.coef[0])
        ax_gain_N_pht.axline(xy1=xy1, slope=p.coef[1], color='C2', ls='--')
        ax_gain_N_pht.set_xlabel('Average')
        ax_gain_N_pht.set_ylabel('Variance')
        
    return (gain, offset)
    

## NMF
## J. Friedric et al, Multi-scale approaches for high-speed imaging and analysis of large neural populations
def nmf_HALS_temporal(Y, A, C, N_iter):
    N_roi = A.shape[1]-1  # The last is the background
    U = A.transpose() @ Y
    V = A.transpose() @ A
    for n_iter in range(N_iter):
        C_update = np.zeros(C.shape, dtype=np.float32)
        for n_roi in range(N_roi):
            C_update[n_roi] = (U[n_roi] - V[n_roi:n_roi+1,:]@C) / V[n_roi,n_roi]
            C[n_roi] += C_update[n_roi]
        # C = C + C_update
        C = np.clip(C, 0, None)
    return C

def nmf_HALS_spatial(Y, A, C, N_iter):
    N_roi = A.shape[1]-1  # The last is the background
    U = C @ Y.transpose()
    V = C @ C.transpose()
    for n_iter in range(N_iter):
        A_update = np.zeros(A.shape, dtype=np.float32)
        for n_roi in range(N_roi):
            A_update[:,n_roi] = (U[n_roi] - V[n_roi:n_roi+1]@A.transpose())/V[n_roi,n_roi]
            A[:,n_roi] += A_update[:,n_roi]
        # print(n_iter, np.abs(A_update[:,:-1]).max(), np.abs(A[:,:-1]).max())
        
        # A = A + A_update
        A = np.clip(A, 0, None)
    return A

def normalize_AC(A, C):
    norm = np.sum(np.average(C[:-1,:], axis=1, keepdims=True))
    A[:,:-1] *= norm
    C[:,-1:] /= norm
        

def roi_nmf(Y, N_iter=5):
    ## J. Friedric et al, Multi-scale approaches for high-speed imaging and analysis of large neural populations
    ## Y = AC + B F
    ## A: Footprint. N_px*1
    ## C: Temporal. 1*N_t
    ## B: Background footprint. N_px*1.
    ## F: Background activities = ones((1, Nt))
    ## Y: N_px*N_t. Note that it is the transpose of the general convention I have
    Y_mean = np.average(Y, axis=1)
    N_px, N_t = Y.shape
    A = np.average(Y, axis=1, keepdims=True)
    C = np.average(Y, axis=0, keepdims=True)
    norm = np.mean(A, axis=0, keepdims=True)
    A /= norm
    C *= norm
    # Include the background
    B = np.zeros((N_px,1), dtype=np.float32)
    A = np.concatenate((A, B), axis=1)
    F = np.ones((1, N_t), dtype=np.float32)
    C = np.concatenate((C, F), axis=0)
    A_init = A.copy()
    C_init = C.copy()
    for n_iter in range(N_iter):
        C = nmf_HALS_temporal(Y, A, C, 5)
        A = nmf_HALS_spatial(Y, A, C, 5)
        normalize_AC(A, C)
        AC_sig_avrg = np.average(A[:,:-1] @ C[:-1,:], axis=1)
        sel = Y_mean<AC_sig_avrg
        idx = np.arange(N_px)[sel]
        for ns in range(len(idx)):
            A[idx[ns],:-1] *= (Y_mean[idx[ns]]/AC_sig_avrg[idx[ns]])
        A[:,-1] = np.clip(Y_mean - AC_sig_avrg, 0, None)
        
        
    nmf_HALS_temporal(Y, A, C, 5)
    
    return { \
            'A': A[:,:-1], \
            'C': C[:,:-1], \
            'B': A[:,-1], \
            'Y_mean': Y_mean, \
        }

def downsample_median_interp(y, med_size, step_size=None):
    # print(med_size, step_size)
    if step_size is None:
        step_size = med_size
    Nt = len(y)
    med_ds = np.zeros(Nt//step_size)
    for n_ds in range(len(med_ds)):
        center_idx = n_ds*step_size + step_size//2
        med_ds[n_ds] = np.nanmedian(y[max(center_idx-med_size//2,0):center_idx-med_size//2+med_size])
    x = np.arange(Nt)
    x_ds = np.arange(len(med_ds)) * step_size + step_size/2
    y_med = np.interp(x, xp=x_ds, fp=med_ds)
    return y_med


def detect_events(y, height, distance, N_frame_bound_rmv, detect_method='simple', kernel_size=None, fix_val=0, ret_peak_only=True):
    kernel_size = copy.deepcopy(kernel_size)
    if detect_method == 'simple':
        y_det = y.copy()
        y_thres = np.zeros(len(y_det)) + height
    elif detect_method == 'mean_std':
        y_rm = uniform_filter1d(y, size=kernel_size['mean'], mode='mirror')
        dy = y - y_rm
        y_std = uniform_filter1d(dy**2, size=kernel_size['std'], mode='mirror')**0.5
        y_det = dy/y_std
        y_thres = height*y_std+y_rm
    elif detect_method == 'med_offset':
        if ('step' in kernel_size) == False:
            kernel_size['step'] = kernel_size['med']//4
        y_med = downsample_median_interp(y, kernel_size['med'], kernel_size['step'])
        y_det = y - y_med
        y_thres = height+y_med
    elif detect_method == 'mean_offset':
        y_rm = uniform_filter1d(y, size=kernel_size['mean'], mode='mirror')
        y_det = y - y_rm
        y_thres = height + y_rm
    elif detect_method == 'med_std':
        if ('step' in kernel_size) == False:
            kernel_size['step'] = kernel_size['med']//4
        y_rm = uniform_filter1d(y, size=kernel_size['mean'], mode='mirror')
        y_med = downsample_median_interp(y, kernel_size['med'], kernel_size['step'])
        dy = y - y_med
        y_std = uniform_filter1d(dy**2, size=kernel_size['std'], mode='mirror')**0.5
        dy_med = y - y_med
        y_det = dy_med/y_std
        y_thres = height*y_std + y_med
    elif detect_method == 'med_med':
        if ('step' in kernel_size) == False:
            kernel_size['step'] = kernel_size['med']//4
        y_rm = uniform_filter1d(y, size=kernel_size['mean'], mode='mirror')
        y_med = downsample_median_interp(y, kernel_size['med'], kernel_size['step'])
        dy = y - y_med
        dy_0 = dy.copy()
        dy[dy<0] = np.nan
        dy_med = downsample_median_interp(dy, kernel_size['med'], kernel_size['step'])
        y_det = dy_0/dy_med
        y_thres = height*dy_med + y_med
    elif detect_method == 'fixed_std':
        y_rm = uniform_filter1d(y, size=kernel_size['mean'], mode='mirror')
        dy = y - fix_val
        y_std = uniform_filter1d(dy**2, size=kernel_size['std'], mode='mirror')**0.5
        y_det = dy/y_std
        y_thres = height*y_std+fix_val
    
    peaks, _ = find_peaks(y_det, height=height, distance=distance)
    peaks = peaks[peaks>N_frame_bound_rmv]
    peaks = peaks[peaks<len(y)-N_frame_bound_rmv]
    if ret_peak_only:
        return peaks
    return {'peaks': peaks, 'y_thres': y_thres, 'y_det': y_det}

def average_trace(y, event_idx, min_distance, N_frame_pre_post, frame_offset):
    event_idx_srt = np.sort(event_idx)
    avrg = np.zeros(N_frame_pre_post*2+1, dtype=np.float64)
    n_event = 0
    for n_ev in range(len(event_idx)):
        if event_idx[n_ev] < N_frame_pre_post+frame_offset:
            continue
        if event_idx[n_ev] > len(y) - ( N_frame_pre_post+frame_offset):
            continue
        if n_ev >= 1:
            if event_idx[n_ev] - event_idx[n_ev-1] < min_distance:
                continue
        if n_ev < len(event_idx)-1:
            if event_idx[n_ev+1] - event_idx[n_ev] < min_distance:
                continue
        avrg += y[event_idx[n_ev]-N_frame_pre_post+frame_offset:event_idx[n_ev]+N_frame_pre_post+frame_offset+1]
        n_event += 1
    return avrg / n_event

@njit
def rs_map_px_to_realspace(x, fill_factor_temporal=0.7):
    ## x is centered
    ## Works for single colume
    x_max = x.max()
    return 2*x_max/(np.pi*fill_factor_temporal) * np.sin(np.pi/2*fill_factor_temporal*x/x_max)

@njit
def rs_map_realspace_to_px(X, N_px, fill_factor_temporal=0.7):
    ## x is centered
    ## Works for single colume
    ## x_max: It is the pixel number / 2
    x_max = N_px/2
    return 2*x_max/(np.pi*fill_factor_temporal) * np.arcsin(np.pi/2*fill_factor_temporal*X/x_max)

@njit
def expand_px_template(roi, px_template, expand_size):
    px_template_expanded = px_template.copy()
    roi_expanded = roi.copy()
    for niter in range(expand_size):
        for n0 in range(roi.shape[0]):
            for n1 in range(roi.shape[1]):
                if px_template_expanded[n0,n1] == 0:
                    tmp = 0
                    n_neighb = 0
                    if n0+1 < roi.shape[0]:
                        if roi_expanded[n0+1,n1] > 0:
                            tmp += px_template[n0+1,n1]
                            n_neighb += 1
                    if n1+1 < roi.shape[1]:
                        if roi_expanded[n0,n1+1] > 0:
                            tmp += px_template[n0,n1+1]
                            n_neighb += 1
                    if n0-1 > 0:
                        if roi_expanded[n0-1,n1] > 0:
                            tmp += px_template[n0-1,n1]
                            n_neighb += 1
                    if n1-1 > 0:
                        if roi_expanded[n0,n1-1] > 0:
                            tmp += px_template[n0,n1-1]
                            n_neighb += 1
                            
                    if n_neighb > 0:
                        px_template_expanded[n0,n1] = tmp / n_neighb
                        roi_expanded[n0,n1] = 1
                        
    return px_template_expanded
                    
        
def blur_px_template(roi, px_template, blur_sigma, blur_kernel_radius):
    if blur_sigma <= 0:
        return px_template.copy()
    px_template_out = expand_px_template(roi, px_template, blur_kernel_radius+2)
    px_template_out = gaussian_filter(px_template_out, blur_sigma, radius=blur_kernel_radius)
    px_template_out[roi==0] = 0
    return px_template_out

