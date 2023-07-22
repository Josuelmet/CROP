# [Provable Instance Specific Robustness via Linear Constraints](https://openreview.net/forum?id=aVbG8bM1wg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HvQZJd9b3pgmjr5K6QUu_M9rtRA-Pvau)

Requirements:
```
torch
```

Using CROP with your own model:

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
    model = ConstrainedSequential.uncast(model)
```


Experiment configurations for [crop_robustness_example.ipynb](crop_robustness_example.ipynb):


Figure 1 uses dataset="moons", imbalanced=True
* Figure 1, top: No additional hyperparameters
* Figure 1, bottom: constrained=True, fine_tune_constraints=True, constraint_eps=0.3

Figure 3 uses dataset="blobs", imbalanced=True
* Figure 2, top left and right: No additional hyperparameters
* Figure 2, bottom left: constrained=True, constrain_all_layers=True, fine_tune_constraints=True, constraint_eps=0.3
* Figure 2, bottom right: constrained=True, fine_tune_constraints=True, constraint_eps=0.6
