# FlatMux - Log-likelihood based spike inference pipeline

## System requirements

* Software and packages:
  
  * Python 3.10.12
  * CaImAn 1.9.15
  * numpy 1.24.4
  * matplotlib 3.7.2
  * scipy 1.11.1
  * matplotlib 3.7.2
  * jax 0.4.20
  * h5py 3.9.0
  * tifffile 2023.7.18
  * Cellpose 2.2.2
  * zarr 2.16.0
  * joblib 1.3.2
  * scikit-learn 1.3.0
  * py-opencv 4.7.0
  * numba 0.57.1

* Hardware and OS:
  
  * Tested on a workstation running Ubuntu 22.04 with two Intel Xeon Gold 6136 CPUs (12 cores each), 256 GB RAM, 4 TB NVMe flash disk, three Nvidia Titan V GPUs with 12 GB GPU-RAM each. To run the code with the demo data, a laptop should be possible. No non-standard 

## Installation

  Extract the code in a separate folder. Typical installation time: < 10 seconds.

## Usage with demo data

The demo data contains part of the recording (motion corrected) presented in Fig. 2. Each step takes a few seconds to run.

1. Run plot_avrg_image.py to generate an average image in the Results folder.

2. Use Cellpose to do segmentation. Use imag_avrg_mc_16bit_maxcontrast.tif as input. In Cellpose, the cell diameter is set to 9 (pixels), and the select "cyto2" as the model. Manually add or remove neurons if needed. Save the mask as png. Cellpose will create file "imag_avrg_mc_16bit_maxcontrast_cp_masks.png" in the "Results" folder.

3. Run "process_sep_roi.py".

4. Run "process_roi_loglikelihood.py". It will calculate the log likelihood ratio. Iteration in the NMF process is not performed in the code due to the short time span which does not contain enough spikes.

5. To plot the trace, edit "trace_roi_sel" in "plot_trace.py" to select the neuron to be plot. Run plot_trace.py which will show the selected neuron, $\Delta F/F$, the log-likelihood ratio of the selected neuron, the detected spikes (orange dots), and the time delivering the whisker stimulus (red dashed lines).
