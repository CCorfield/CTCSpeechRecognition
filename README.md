# CTCSpeechRecognition -- Voxforge Branch

WARNING, this branch is very experimental and uses cudnn v5 which has not got stable R5 bindings (is a [WIP](https://github.com/soumith/cudnn.torch/tree/R5)).

First you must have cudnn v5 which can be downloaded and installed [here](https://developer.nvidia.com/cudnn). BEWARE cudnn v5 requires new torch bindings so other scripts depending on cudnn v4 will fail.

At the current time (09/04/16) I have a fork of the cudnn bindings that you can install instead of the standard cudnn bindings using these commands:
```
luarocks remove cudnn
luarocks install https://raw.githubusercontent.com/SeanNaren/cudnn.torch/R5/cudnn-scm-1.rockspec
```

Work in progress. Implementation of the [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc) using torch7. Feeds spectrogram data into a neural network using the Torch7 library, training itself with the CTC activation function.

Current implementation runs on CUDA 7.5, but can be using anything above 7.0.

## Installation

To install torch7 follow the guide [here](http://torch.ch/docs/getting-started.html).

You must have CUDA 7.0 or above (build supported by warp-ctc). To install CUDA 7.5:

Download the .run file of your platform [here](https://developer.nvidia.com/cuda-downloads).

Example for Ubuntu 14.04:
```
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
```
Then install using below commands.
```
sudo chmod +x ./cuda_7.5.18_linux.run
sudo ./cuda_7.5.18_linux.run
```
When prompted with this message I suggest pressing no. Highly recommended to install the latest driver from the nvidia website of your GPU,
but the driver is compatible if you select yes.
```
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 352.xx? ((y)es/(n)o/(q)uit):
```
Finally modify the .bashrc located at ~/.bashrc, including these lines at the end:
```
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
```
Restart the terminal for changes to take effect.

For CUDA implementation (make sure to install these via luarocks first before installing the warp-ctc library):
```
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
```

To install the Baidu warp-ctc library:
```
luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec
```
Other dependencies can be installed via luarocks:

[Audio Library for Torch](https://github.com/soumith/lua---audio): Audio Library for Torch</a>:

Linux (Ubuntu):
```
sudo apt-get install libfftw3-dev
sudo apt-get install sox libsox-dev libsox-fmt-all
```

```
luarocks install https://raw.githubusercontent.com/soumith/lua---audio/master/audio-0.1-0.rockspec
```

[Optim](https://github.com/torch/optim): numeric optimization package for Torch</a>:
```
luarocks install optim
```

[rnn](https://github.com/Element-Research/rnn): Recurrent Neural Network library for Torch7's nn</a>:
```
luarocks install rnn
```

[lua---nx](https://github.com/clementfarabet/lua---nnx): An extension to Torch7's nn package</a>:
```
luarocks install nnx
```

[xlua](https://github.com/torch/xlua): A set of useful extensions to Lua</a>:
```
luarocks install xlua
```

It is also suggested to update the following libraries:
```
luarocks install torch
luarocks install nn
luarocks install dpnn
```
**If using the voxforge branch do not install cudnn again like below (just install how described at the top of README).**
For cudnn you need to create an account, follow install instructions [here](https://developer.nvidia.com/cudnn).

Once you have completed the above installation, Install lua bindings for [cudnn](https://github.com/soumith/cudnn.torch):
```
luarocks install cudnn
```

Main method located at AN4CTCTrain.lua.

Training data is currently the [AN4 Audio database](http://www.speech.cs.cmu.edu/databases/an4/). 

Within the AN4CTCTrain we specify the file path to the an4 dataset which can be downloaded [here](http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz).

Extract the folder and give the filepath in the AN4CTCTrain.lua script.

**Note that the data has to be converted into wav format, a bash script ConvertAN4ToWav.sh has been included to help with this (place this into the an4 directory then run).**
