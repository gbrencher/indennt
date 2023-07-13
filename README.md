# indennt
Interferogram denoising neural network

Atmospheric noise frequently obscures real displacement signals in InSAR interferograms. This convolutional neural network has been trained to remove atmospheric noise from interferograms over mountainous environments, preserving small, defined displacement signals associated with landslides, rock glaciers, and other slope processes. It can be easily applied to correct your own interferograms. 

For more information, check out [this preprint](https://www.techrxiv.org/articles/preprint/Removing_Atmospheric_Noise_from_InSAR_Interferograms_in_Mountainous_Regions_with_a_Convolutional_Neural_Network/22626748)

## Installation
Download and install Miniconda
Set up Mamba
```
$ conda install mamba -n base -c conda-forge
```
Clone the repo and set up the environment
```
$ git clone https://github.com/gbrencher/indennt.git
$ cd ./indennt
$ mamba env create -f environment.yml
$ conda activate indennt
```
## Usage
While the CNN can correct unwrapped interferograms from any source, it was trained on [HyP3](https://hyp3-docs.asf.alaska.edu/guides/insar_product_guide/) interferograms with 40 m spatial resolution. The example notebook demonstrates functions to correct HyP3 interferograms.

```
from indennt.models import UNet, torch
from indennt.core import correct_single_igram, correct_hyp3_dir
import matplotlib.pyplot as plt

#load model
model = UNet()
model.load_state_dict(torch.load('weights/noisemodel1.4_174epochs'))
model.eval();

# correct a single hyp3 interferogram, return as xarray ds
igram_path = '/mnt/d/indennt/hyp3_app/AT137/2020/S1BB_20200808T011058_20201007T011100_VVP060_INT40_G_ueF_70CB'
ds = correct_single_igram(igram_path, model)

# correct multiple hyp3 interferograms
#hyp3_path = '/mnt/d/indennt/hyp3_app/AT137/2017' # dir containing hyp3 outputs
#correct_hyp3_dir(hyp3_path, model, skip_exist=True)
```
![plot](./images/example_correction.png)

There's a lot more to be done, but hopefully this is enough to get started. 

## Contact 
Please don't hesitate to reach out with questions or ideas! I'll do my best to get back to you. 
George (Quinn) Brencher: gbrench@uw.edu