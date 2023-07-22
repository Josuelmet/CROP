# [Provable Instance Specific Robustness via Linear Constraints](https://openreview.net/forum?id=aVbG8bM1wg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HvQZJd9b3pgmjr5K6QUu_M9rtRA-Pvau)

To use CROP on your own model: 

TODO: describe casting + uncasting; fine-tuning

TODO: mention do_not_constrain

TODO: mention requirements or lack thereof


Experiment configurations for [crop_robustness_example.ipynb](crop_robustness_example.ipynb):


Figure 1 uses dataset="moons", imbalanced=True
* Figure 1, top: No additional hyperparameters
* Figure 1, bottom: constrained=True, fine_tune_constraints=True, constraint_eps=0.3

Figure 3 uses dataset="blobs", imbalanced=True
* Figure 2, top left and right: No additional hyperparameters
* Figure 2, bottom left: constrained=True, constrain_all_layers=True, fine_tune_constraints=True, constraint_eps=0.3
* Figure 2, bottom right: constrained=True, fine_tune_constraints=True, constraint_eps=0.6
