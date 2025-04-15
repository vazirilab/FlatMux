import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import os
import json
from scipy import interpolate
from scipy import signal, ndimage
from analysis_utils import gain_calib_shotnoise, load_imag
from scipy.stats import poisson as poissondist

class PMTNoise:
    def __init__(self, \
            fpath_noisemodel_lut='./PMT_noise/Results/P_LUT.npz', \
        ):
    # def __init__(self, \
    #         fpath_noisemodel_lut='/data1/Jingkun/code/noisemodel/LmuxPMT/PMT_0.88_2/Results/P_LUT.npz', \
    #     ):
        self.lut_data = np.load(fpath_noisemodel_lut, allow_pickle=True)
        self.lut_px_val = self.lut_data['px_val']
        self.lut_px_val_mean =self.lut_data['px_val_mean']
        self.lut_p = self.lut_data['P']
        self.lut_px_val_coeff = self.lut_data['px_val_coeff']
        self.N_lut_px_val = self.lut_px_val.shape[1]
        # Construct the interpolator
        lut_px_val_mean = (np.expand_dims(self.lut_px_val_mean, axis=1) * np.ones((1, self.lut_p.shape[1])))
        lut_px_val_mean_flat = lut_px_val_mean.flatten()
        lut_px_val_flat = self.lut_px_val.flatten()
        lut_p_flat = self.lut_p.flatten()
        # lut_px_val_mean_flat = np.append(lut_px_val_mean_flat, 1e-3)
        # lut_px_val_flat = np.append(lut_px_val_flat, 10)
        # lut_p_flat = np.append(lut_p_flat, 0)
        # self.lut_interp = interpolate.CloughTocher2DInterpolator(list(zip(lut_px_val_mean_flat, lut_px_val_flat)), lut_p_flat, maxiter=1000)
        # self.lut_interp = interpolate.LinearNDInterpolator(list(zip(lut_px_val_mean_flat, lut_px_val_flat)), lut_p_flat, fill_value=0)
        
        lut_px_val_x = np.arange(self.N_lut_px_val)
        self.lut_interp = interpolate.RegularGridInterpolator((self.lut_px_val_mean, lut_px_val_x), self.lut_p, \
            bounds_error=False, fill_value=0, method='linear')
    
    
    def lut_px_val_transform(self, x_mean, x):
        return (x-self.lut_px_val_coeff[0])*(self.N_lut_px_val-1) / \
            (self.lut_px_val_coeff[1]-self.lut_px_val_coeff[0] \
            +self.lut_px_val_coeff[2]*x_mean**0.5 \
            +self.lut_px_val_coeff[3]*x_mean)
        
    
    def px_prob(self, x_mean, x):
        x_mean_bc = np.broadcast_to(x_mean, x.shape)
        y = self.lut_interp( \
                list(zip(x_mean_bc.flatten(), self.lut_px_val_transform(x_mean_bc, x).flatten())) \
            ).reshape(x.shape)
        # y = self.lut_interp(x_mean_bc, x)
        # y = poissondist.pmf(np.round(x), np.broadcast_to(x_mean, x.shape))
        return y
    
    def check_px_hist(self, data, N_px=4, px_bright_ratio_bound=(0.5, 1.0), rng_seed=0, px_void=[]):
        ## Check whether the noise model applies to the data
        rng = np.random.default_rng(rng_seed)
        Nt, Ny, Nx = data.shape
        px_mean = np.average(data, axis=0)
        image_average = px_mean
        px_idx = np.arange(Nx*Ny).reshape((Ny, Nx))
        x = np.arange(Nx)
        y = np.arange(Ny)
        X, Y = np.meshgrid(x,y)
        
        ## Select pixels to do the evaluation
        px_mean_flat = px_mean.flatten()
        px_idx_flat = px_idx.flatten()
        
        sel = (px_mean_flat>=px_bright_ratio_bound[0]*px_mean_flat.max()) \
            & (px_mean_flat<=px_bright_ratio_bound[1]*px_mean_flat.max())
        for n_void in range(len(px_void)):
            sel[px_void[0],px_void[1]] == False
        
        px_idx_sel_range = px_idx_flat[sel]
        px_idx_sel = rng.choice(px_idx_sel_range, size=N_px)
        
        px_val = np.zeros((N_px, Nt), dtype=float)
        for n_px in range(N_px):
            px_coord = np.unravel_index(px_idx_sel[n_px], (Ny, Nx))
            px_val[n_px] = data[:,px_coord[0],px_coord[1]]
        px_val_mean = np.average(px_val, axis=1)
        
        fig_px_sel, ax_px_sel = plt.subplots(1,1)
        ax_px_sel.pcolormesh(X, Y, image_average, shading='nearest', \
            cmap='gray', vmin=0, vmax=image_average.max())
        for n_px in range(N_px):
            px_coord = np.unravel_index(px_idx_sel[n_px], (Ny, Nx))
            ax_px_sel.plot([px_coord[1]], [px_coord[0]], ls='', marker='o', mec='C0', mfc=(0,0,0,0))
        for n_void in range(len(px_void)):
            ax_px_sel.plot([px_void[n_void][1]], [px_void[n_void][0]], ls='', marker='x', color='r')
        
        ## Get the probability density distribution of the selected pixels and interpolate from the LUT
        bin_edge = np.linspace(start=px_val.min(), stop=px_val.max(), num=201)
        bin_edge_center = (bin_edge[1:]+bin_edge[:-1])/2
        bin_edge_center_reconstruct = np.linspace(start=px_val.min(), stop=px_val.max(), num=1000)
        fig_px, ax_px = plt.subplots(N_px, 1, sharex=True)
        for n_px in range(N_px):
            px_val_hist_tmp, px_val_bin_edges_tmp = np.histogram(px_val[n_px], bins=bin_edge, density=True)
            ax_px[n_px].stairs(px_val_hist_tmp, px_val_bin_edges_tmp)
            px_val_hist_reconstruct = self.px_prob(px_val_mean[n_px], bin_edge_center_reconstruct)
            ax_px[n_px].plot(bin_edge_center_reconstruct, px_val_hist_reconstruct, ls='--', color='C1')
            ax_px[n_px].set_yscale('log')
            ax_px[n_px].set_ylim((1e-5*px_val_hist_tmp.max(), 1.5*px_val_hist_tmp.max()))
            ax_px[n_px].axvline(px_val_mean[n_px], ls=':', color='g')
        ax_px[-1].set_xlabel('Pixel value')
    
    def px_log_likelyhood_ratio(self, x_mean_0, x_mean_1, x):
        p_0 = self.px_prob(x_mean_0, x)
        p_1 = self.px_prob(x_mean_1, x)
        p_0 = np.clip(p_0, 1e-6, None)
        p_1 = np.clip(p_1, 1e-6, None)
        return np.log(np.clip(p_1/p_0, 1e-100, None))
    
#     def simulate_px(self, px_mean, N=int(1e4), rng_seed=0, N_discretize=2**8, x=None):
#         rng = np.random.default_rng(rng_seed)
#         x_random = np.zeros(N)
#         if x is None:
#             x = np.linspace(start=self.lut_px_val.min(), stop=2+px_mean[n]+px_mean[n]**0.5*4, num=N_discretize)
#         p = self.px_prob(px_mean, x)
#         p /= np.sum(p)
#         x_random[n,:] = rng.choice(x, size=N, p=p)
#         return x_random
#     
    
