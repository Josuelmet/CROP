import torch
from torch import nn
from torch import Tensor

EPS = 1e-2

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
    extra_bias = (h_c[:, conflict_dims] * desired_signs).amin(0).clamp(max=0) * desired_signs * (1 + EPS)
    h[:, conflict_dims] -= extra_bias
    self.last_extra_bias = extra_bias
    self.last_conflict_dims = conflict_dims
    return h




# NOTE: This method has not yet been tested for Conv1D or Conv3D

def enforce_constraint_conv(
    self, h: Tensor, R: int, V: int, force_linearity=True
) -> Tensor:

    # Get the number of channels
    C = h.shape[1]

    # Isolate the preactivations belonging to constraint region vertices
    h_c = torch.clone(h[-R * V:])
    # Make the first dimension the channel dimension
    h_c = h_c.transpose(0, 1)
    # Flatten all but the first two dimensions
    h_c_flat = h_c.reshape((C, R, -1))


    with torch.no_grad():
        # The bias applied to each channel's output is a scalar.
        # Therefore, we need to take the aggregate sum of *all* entry signs from
        # *all* vertices of *eac h* constraint region for *each* channel.
        # We do this via sign(h_c_flat).sum(2), which returns a (C x R) matrix.
        # Taking the sign() of that matrix indicates the majority sign of each
        # channel of each constraint region.
        regionwise_majority = sign(sign(h_c_flat).sum(2))

        # regionwise_majority is now a (C x R) binary tensor with values {-1, 1}.
        # We compute desired_signs (a length-C vector) via majority vote among regions
        desired_signs = sign(regionwise_majority.sum(1))

        # If we don't need to force linearity in all regions,
        # then we only need to look at points that are part of the majority-sign coalition.
        # All other points will be multiplied by zero.
        if not force_linearity:
            h_c_flat *= (regionwise_majority == desired_signs[:, None]).unsqueeze(2)


    # Calculate extra bias
    extra_bias = (h_c_flat * desired_signs[:, None, None]).amin((1,2)).clamp(max=0) * desired_signs * (1 + EPS)

    # Add and store extra bias
    self.last_extra_bias = extra_bias
    # Reshape extra_bias to be compatible with the shape of h
    return h - extra_bias.reshape((1, C,) + tuple(torch.ones(h.ndim - 2).to(int)))





def is_supported(layer):
    supported = [
        # nn.modules.pooling._AvgPoolNd,
        nn.modules.conv._ConvNd,
        nn.modules.conv._ConvTransposeNd,
        nn.modules.batchnorm._BatchNorm,
        nn.Linear
    ]
    return any([isinstance(layer, layer_type) for layer_type in supported])

def is_constrained(layer):
    return 'constrained' in layer.__class__.__name__.lower()

def is_conv(layer):
    return isinstance(layer, nn.modules.conv._ConvNd) or isinstance(layer, nn.modules.conv._ConvTransposeNd)



@classmethod
def cast(cls, module: nn.Module):
    assert isinstance(module, nn.Module)
    module.prev_class = module.__class__
    module.__class__ = cls
    assert isinstance(module, cls)
    return module

def forward_linear(self, input, R, V, force_linearity=True):
    return enforce_constraint_linear(self, self.prev_class.forward(self, input), R, V, force_linearity)

def forward_conv(self, input, R, V, force_linearity=True):
    return enforce_constraint_conv(self, self.prev_class.forward(self, input), R, V, force_linearity)


# Make a Constrained layer via casting.
def constrain_layer(layer: nn.Module, C=None, N_c=None):
    assert is_supported(layer)
    if is_constrained(layer):
        return layer

    new_class = type(f'Constrained{layer.__class__.__name__}', (layer.__class__,), {
                        "cast": cast,
                        "forward": forward_conv if is_conv(layer) else forward_linear,
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
            if 'do_not_constrain' in vars(m):
                if m.do_not_constrain:
                    continue
            if is_supported(m):
                if m.bias is None:
                    raise ValueError(f"Layer {m}.bias is None. Please ensure that all layers have biases before casting.")
                m = constrain_layer(m)
            elif isinstance(m, nn.Sequential):
                m = ConstrainedSequential.cast(m, constrain_last=True, main=False)

        if constrain_last and is_supported(modules[-1]):
            modules[-1] = constrain_layer(modules[-1])
        elif isinstance(modules[-1], nn.Sequential):
            modules[-1] = ConstrainedSequential.cast(modules[-1], constrain_last=constrain_last, main=False)

        assert isinstance(model, ConstrainedSequential)
        return model

    @classmethod
    def uncast(cls, model, constraints=None, main=True):

        if not is_constrained(model):
            return model

        assert isinstance(model, cls)
        model.eval()

        # If constraints are given, pass the constraints one lass time through the model
        # to make sure that the extra biases are correct.
        if constraints is not None and main:
            model(torch.Tensor([]).to(constraints.dtype).to(constraints.device), constraints)
            # Alternate: it doesn't matter what our "official input" is, so we could just choose the first constraint point.
            #model(constraints[0][0:1], constraints)

        for i, m in enumerate(model):
            if not is_constrained(m):
                continue
            if isinstance(m, nn.Sequential):
                model[i] = ConstrainedSequential.uncast(m, main=False)
                continue
            with torch.no_grad():
                if is_conv(m):
                    model[i].bias -= m.last_extra_bias
                else:
                    if any(m.last_conflict_dims):
                        model[i].bias[m.last_conflict_dims] -= m.last_extra_bias
            model[i].__class__ = m.prev_class


        model.__class__ = model.prev_class
        assert isinstance(model, model.prev_class)
        return model



    def forward(self, input, constraints, force_linearity=True):

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




@torch.no_grad()
def check_layerwise_signs(model,constraints):
    v = constraints
    layerwise_flags = []
    layerwise_unmatched_act = []
    layerwise_act = []

    for m in model.modules():
        if isinstance(m, nn.Sequential):
            continue

        v = m(v, R=1, V=len(constraints)) if is_constrained(m) else m(v)
        layerwise_act.append(v)

        # If this layer was never constrained:
        if 'last_extra_bias' not in vars(m):
            continue

        v_sign = v > 0
        match = v_sign  != v_sign[0,...]
        layerwise_unmatched_act.append(torch.sum(torch.abs(v[match])))
        layerwise_flags.append(torch.all(v_sign  == v_sign[0,...]))

    return layerwise_flags, layerwise_unmatched_act, layerwise_act
