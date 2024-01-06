# Industrial_det
Please read the codes and relative papers to understand basic methods in deep learning for image classification.
Install the python environment as following instructions, modify parameters in configure
if you need. If you have been familiar with the architecture of the process, go on and try other way.
# Installation

Install in your server of PC, there are a few differences between windows and linux op-syt.
You need to modify the root path in ./configs/config.py before experiment.

use Anaconda or Miniconda to 
```
conda create --name Industrial_det python=3.10
conda activate Industrial_det
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
sh run.sh
```