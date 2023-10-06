# [Provable Instance Specific Robustness via Linear Constraints (CROP)](https://openreview.net/forum?id=aVbG8bM1wg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HvQZJd9b3pgmjr5K6QUu_M9rtRA-Pvau)

### Requirements for CROP (if using SplineCam, look at the end of this README):
```
torch
```


### Using CROP with your own nn.Sequential model:

```ruby
from crop import *

# 1) Set layer.do_not_constrain = True to any Linear/normalization layer that should not be constrained
# For example, if constraining only the last layer, then all layers but the last need to have .do_not_constrain = True before casting.
# For example: model._modules['0']. do_not_constrain = True

# 2) Cast the model to a ConstrainedSequential
# If you are constraining the last layer, pass constrain_last = True in the cast() function.
model = ConstrainedSequential.cast(model)

# 3) Make your R x V x D array of constraint regions and vertices.
constraints = torch.Tensor([...]).to(device).to(dtype) # device and dtype should match those of your model

# 4) Fine-tune (for just a few epochs) or train (from scratch) your model on the constraints

# 5) Uncast your model if desired / if using SplineCam:
if is_constrained(model):
    model = ConstrainedSequential.uncast(model, constraints)
```


Experiment configurations for [crop_robustness_example.ipynb](crop_robustness_example.ipynb):


Figure 1 uses dataset="moons", imbalanced=True
* Figure 1, top: No additional hyperparameters
* Figure 1, bottom: constrained=True, fine_tune_constraints=True, constraint_eps=0.3

Figure 3 uses dataset="blobs", imbalanced=True
* Figure 2, top left and right: No additional hyperparameters
* Figure 2, bottom left: constrained=True, constrain_all_layers=True, fine_tune_constraints=True, constraint_eps=0.3
* Figure 2, bottom right: constrained=True, fine_tune_constraints=True, constraint_eps=0.6


### SplineCam setup

Here is how we setup SplineCam on Colab, but see the [SplineCam repository](https://github.com/AhmedImtiazPrio/splinecam/tree/main).

If using Conda, another option is to use the [SplineCam environment .yml file](https://github.com/AhmedImtiazPrio/splinecam/blob/main/environment.yml).

```
# Taken from https://colab.research.google.com/github/count0/colab-gt/blob/master/colab-gt.ipynb
# If this doesn't work, here's a potential workaround: https://stackoverflow.com/questions/69404659/troubles-with-graph-tool-installations-in-google-colab
!echo "deb http://downloads.skewed.de/apt jammy main" >> /etc/apt/sources.list
!apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25
!apt-get update
!apt-get install python3-graph-tool python3-matplotlib python3-cairo

# Colab uses a Python install that deviates from the system's! Bad collab! We need some workarounds.
!apt purge python3-cairo
!apt install libcairo2-dev pkg-config python3-dev
!pip install --force-reinstall pycairo
!pip install zstandard


#!pip install --upgrade gdown
!git clone https://github.com/AhmedImtiazPrio/splinecam.git

!pip install networkx
!pip install python-igraph>=0.10
!pip install tqdm
!pip install livelossplot

!pip uninstall torch torchvision -y
!pip install --pre torch==1.12+cu116 torchvision -f https://download.pytorch.org/whl/torch_stable.html
```
