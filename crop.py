import torch
from torch import Tensor
from torch import nn


def sign(tensor):
    return tensor.sign() + (tensor == 0)


def enforce_constraint_linear(
    self, h: Tensor, R: int, V: int, force_linearity=True
) -> Tensor:
    """
    Perform a forward pass on the given `h` argument which contains both the mini-batch and
    the regions `R` with vertices `V` onto which the DNN is constrainted to stay affine.
    Args:
        h (Tensor): vertices and activations of the linear layer before nonlinearity,
                    the first `R` * `V` rows contain the indices
        R (int): number of regions
        V (int): number of vertices describing each region
        force_linearity (bool): whether linearity needs to be enforced in every region
    Returns:
        Tensor: the forwarded inputs and vertices
    """

    # Given h, the pre-activation for everyone (data + constraints).
    # Shape is thus (N + R * V, K) with K the output dim (since W is R^D:-> R^K)
    h_c = torch.clone(h[-R * V:])
    h_c_signs = sign(h_c).reshape((R, V) + h_c.shape[1:])

    with torch.no_grad():
        # Select which units/neurons actually need intervention;
        # i.e., which neurons do not have signs that agree within each of the R constraint regions.
        # conflict_dims is a length-K boolean vector.
        conflict_dims = (h_c_signs.sum(1).abs() != V).any(0)

        # Calculate the overall majority sign of each unit/neuron in conflict_dims.
        # Let K_conflict be the number of True elements in conflict_dims. Then,
        # desired_signs is a (K_conflict,) shaped vector with values in {-1, 1}.
        desired_signs = sign(h_c_signs.sum((0,1)))[conflict_dims]

        if not force_linearity:
            # Calculate each region's majority sign for each neuron.
            # regionwise_majority has shape (R, K_conflict)
            regionwise_majority = sign(h_c_signs.sum(1))[:, conflict_dims]

            # Reshape the conflicted part of h_c to (R, V, K_conflict), then
            # multiply by 0 all neurons that do not agree with the regionwise majority.
            h_c[:, conflict_dims] = (
                h_c.reshape_as(h_c_signs)[:, :, conflict_dims] * (regionwise_majority == desired_signs).unsqueeze(1)
            ).reshape_as(h_c[:, conflict_dims])


    # Look by how much do we have to shift each hyper-plane
    # so that all constraints have the majority sign
    extra_bias = (h_c[:, conflict_dims] * desired_signs).amin(0).clamp(max=0) * desired_signs * (1 + 1e-3)
    h[:, conflict_dims] -= extra_bias
    self.last_extra_bias = extra_bias
    self.last_conflict_dims = conflict_dims
    return h





def is_supported(layer):
    supported = [
        # nn.modules.pooling._AvgPoolNd,
        # nn.modules.conv._ConvNd,
        # nn.modules.conv._ConvTransposeNd,
        nn.modules.batchnorm._BatchNorm,
        nn.Linear
    ]
    return any([isinstance(layer, layer_type) for layer_type in supported])

def is_constrained(layer):
    return 'constrained' in layer.__class__.__name__.lower()



@classmethod
def cast(cls, module: nn.Module):
    assert isinstance(module, nn.Module)
    module.prev_class = module.__class__
    module.__class__ = cls
    assert isinstance(module, cls)
    return module

def forward(self, input, R, V, force_linearity=True):
    return enforce_constraint_linear(self, self.prev_class.forward(self, input), R, V, force_linearity)


# Make a Constrained layer via casting.
def constrain_layer(layer: nn.Module, C=None, N_c=None):
    if is_constrained(layer):
        return layer

    new_class = type(f'Constrained{layer.__class__.__name__}', (layer.__class__,), {
                        "cast": cast,
                        "forward": forward
                    })
    return new_class.cast(layer)



class ConstrainedSequential(nn.Sequential):
    @classmethod
    def cast(cls, model: nn.Sequential, constrain_last=False, main=True):
        assert isinstance(model, nn.Module)

        model.prev_class = type(model)
        model.__class__ = cls
        model.main = main

        modules = [m for m in model if is_supported(m) or isinstance(m, nn.Sequential)]
        for m in modules[:-1]:
            if is_supported(m):
                if 'do_not_constrain' in vars(m):
                    if m.do_not_constrain:
                        continue
                m = constrain_layer(m)
            elif isinstance(m, nn.Sequential):
                m = ConstrainedSequential.cast(m, constrain_last=True, main=False)

        if constrain_last and is_supported(m):
            modules[-1] = constrain_layer(modules[-1])
        elif isinstance(modules[-1], nn.Sequential):
            modules[-1] = ConstrainedSequential.cast(modules[-1], constrain_last=constrain_last, main=False)

        assert isinstance(model, ConstrainedSequential)
        return model

    @classmethod
    def uncast(cls, model):

        # TODO: Consider using super().eval() and merging this function with eval().

        # TODO: Fix uncasting to work with convolutional layers. This may require changing the forward enforcement function

        if not is_constrained(model):
            return model

        assert isinstance(model, cls)
        model.eval()

        for m in model._modules.values():
            if not is_constrained(m):
                continue

            with torch.no_grad():
                m.bias[m.last_conflict_dims] -= m.last_extra_bias
            m.__class__ = m.prev_class

        model.__class__ = model.prev_class
        assert isinstance(model, model.prev_class)
        return model



    def forward(self, input, constraints, force_linearity=True):

        # TODO: Consider adding "if training" to this function so that eval() works optimally

        # TODO: Need to make sure this code works with VGG w/ Avg-pooling

        R, V = constraints.shape[:2]

        if self.main:
            with torch.no_grad():
                input = torch.cat([input, constraints.detach().reshape((-1,) + constraints.shape[2:])], 0)

        for module in self:
            if is_constrained(module):
                if isinstance(module, nn.Sequential):
                    input = module(input, constraints, force_linearity)
                else:
                    input = module(input, R=R, V=V, force_linearity=force_linearity)
            else:
                input = module(input)

        if self.main:
            return input[:-R* V]
        else:
            return input
