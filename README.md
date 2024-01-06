# Industrial_det
Please read the codes and relative papers to understand basic methods in deep learning for image classification.
Install the python environment as following instructions, modify parameters in configure
if you need. If you have been familiar with the architecture of the process, go on and try other way.
# Installation

Install in your server of PC, there are a few differences between windows and linux op-syt.
You need to modify the root path in [./configs/config.py](https://github.com/jerryfeng2003/Industrial_det/blob/main/configs/config.py) before experiment.
The baseline pretrianed model swin_v2_b can be downloaded through [pytorch.org](https://download.pytorch.org/models/swin_v2_b-781e5279.pth), remember to correct the name.

use Anaconda or Miniconda to manage your packages.
```
conda create --name Industrial_det python=3.10
conda activate Industrial_det
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
sh run.sh
```