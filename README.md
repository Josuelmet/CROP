# [Provable Instance Specific Robustness via Linear Constraints](https://openreview.net/forum?id=aVbG8bM1wg)

Experiment configurations for [police_robustness.ipynb](police_robustness.ipynb):


Figure 1 uses dataset="moons", imbalanced=True
* Figure 1, top: No additional hyperparameters
* Figure 1, bottom: constrained=True, fine_tune_constraints=True, constraint_eps=0.3

Figure 3 uses dataset="blobs", imbalanced=True
* Figure 2, top left and right: No additional hyperparameters
* Figure 2, bottom left: constrained=True, constrain_all_layers=True, fine_tune_constraints=True, constraint_eps=0.3
* Figure 2, bottom right: constrained=True, fine_tune_constraints=True, constraint_eps=0.6
