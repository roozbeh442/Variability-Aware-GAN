# Variability-Aware-GAN
This repository contains the matlab code for  Variability Aware GAN
This Repository contains Matlab code for VA-GAN architecture. The architecture is available with two different loss functions. The MinMax loss and Wasserstein loss.  The code is   commented and  self explanatory. The Matlab version required to run the code is Matlab 2021a or later.
To run the code, binary realizations need to be loaded to a folder with the extension .tif this is because the network accepts single channel images only. The address to that folder must be assigned to the variable “datasetFolder” at the beginning of the code.
Finally copy all folder contents to the current folder location in Matlab. For the VA-GAN(LMinMax) run the file “GAN_training.m”. For VA-GAN(LWasserstein) run the file “WGANgp.m”
