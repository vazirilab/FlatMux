'''
Jingkun GUo
'''

import numpy as np
import matplotlib.pyplot as plt
import os

import tifffile

from analysis_utils import save_tiff, load_imag

if __name__ == '__main__':
    data_orig = load_imag('./demo_data/demo_data.tif')
    os.makedirs('./Results', exist_ok=True)
    
    image_shape = (data_orig.shape[1], data_orig.shape[2])
    
    image_average_fw = np.average(data_orig[::2], axis=0)
    image_average_bw = np.average(data_orig[1::2], axis=0)
    
    x = np.arange(image_average_fw.shape[1])
    y = np.arange(image_average_fw.shape[0])
    X, Y = np.meshgrid(x,y)
    
    fig_img = plt.figure()
    ax_img_fw = fig_img.add_subplot(121)
    ax_img_bw = fig_img.add_subplot(122, sharex=ax_img_fw, sharey=ax_img_fw)
    ax_img_fw.pcolormesh(X, Y, image_average_fw, shading='nearest', \
        cmap='gray', vmin=image_average_fw.min(), vmax=image_average_fw.max())
    ax_img_fw.set_aspect('equal','box')
    
    ax_img_bw.pcolormesh(X, Y, image_average_bw, shading='nearest', \
        cmap='gray', vmin=image_average_bw.min(), vmax=image_average_bw.max())
    ax_img_bw.set_aspect('equal','box')
    
    save_tiff('./Results/imag_avrg_mc.tif', (image_average_bw+image_average_fw)/2, gamma=0.5)
    
    
