from abc import abstractmethod
import torch
from torch import nn
import torch.nn.functional as F

import crop.sequential as crop

from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.resnet import ResNet
from torchvision.models.densenet import DenseNet




# Base (abstract) class
class ConstrainedNonSequential(nn.Module):
    # This class is an abstract classs.
    # It should not be directly casted to / instantiated.
    # It should instead be used as a parent class / interface.


    @abstractmethod
    def get_constrainable_layer(self):
        # Return the (last) constrainable layer.
        pass


    @abstractmethod
    def pre_constraint_forward(self, input):
        # Return the output of the typical forward() function
        # right before the (final) constrainable layer.
        pass


    @classmethod
    def cast(cls, model):
        assert isinstance(model, nn.Module)

        if crop.is_constrained(model):
            return model

        # Convert the model to a "constrained" class
        model.prev_class = model.__class__
        model.__class__ = cls
        assert isinstance(model, cls)

        model.init_constrained_layer()
        m = model.constrained_layer

        if isinstance(m, nn.Sequential):
            # Constrain the final Sequential layer,
            # which in ViTs just has a single Linear layer.
           m = crop.ConstrainedSequential.cast(m, constrain_last=True, main=False)
        elif isinstance(m, nn.Linear):
            # Constrain the final Linear layer
            m = crop.constrain_layer(m)
        else:
            raise ValueError(f"The constrainable layer {m} is neither Sequential nor Linear")

        assert crop.is_constrained(model)
        return model


    @classmethod
    def uncast(cls, model, constraints=None):

        if not crop.is_constrained(model):
            return model

        model.eval()

        # If constraints are given, pass the constraints one lass time through the model
        # to make sure that the extra biases are correct.
        if constraints is not None:
            model(torch.Tensor([]).to(constraints.dtype).to(constraints.device), constraints)

        # Identify the constrained layer
        m = model.constrained_layer

        # Uncast the constrained layer
        with torch.no_grad():
            if isinstance(m, nn.Sequential):
                # no need to pass in constraints since we already did the forward pass earlier
                m = crop.ConstrainedSequential.uncast(m)
            else:
                # This is the same code as uncasting a Linear layer.
                m.bias -= m.last_extra_bias
                m.__class__ = m.prev_class

        # "Unconstrain" the model itself, and make sure unconstraining changed the class.
        model.__class__ = model.prev_class
        assert isinstance(model, model.prev_class)
        assert not crop.is_constrained(model)
        return model


    def forward(self, input, constraints, force_linearity=True):

        R, V = constraints.shape[:2]

        with torch.no_grad():
            input = torch.cat([input, constraints.detach().reshape((-1,) + constraints.shape[2:])], 0)

        x = self.pre_constraint_forward(input)

        m = self.constrained_layer

        if isinstance(m, nn.Sequential):
            x = m(input=x, constraints=x[-R*V:].reshape((R,V,-1)), force_linearity=force_linearity)
        else:
            x = m(input=x, R=R, V=V, force_linearity=force_linearity)

        return x[:-R*V]









# ViT class
class ConstrainedVisionTransformer(ConstrainedNonSequential, VisionTransformer):

    def init_constrained_layer(self):
        self.constrained_layer = self.heads

    def pre_constraint_forward(self, input):
        # This part is taken straight from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py ,
        # except now we return the input of the heads

        # Reshape and permute the input tensor
        x = self._process_input(input)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x









# ResNet class
class ConstrainedResNet(ConstrainedNonSequential, ResNet):

    def init_constrained_layer(self):
        self.constrained_layer = self.fc

    def pre_constraint_forward(self, input):
        # This part is taken straight from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py ,
        # except now we return the input of the fc

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x









class ConstrainedDenseNet(ConstrainedNonSequential, DenseNet):

    def init_constrained_layer(self):
        self.constrained_layer = self.classifier

    def pre_constraint_forward(self, input):
        # This part is taken straight from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py ,
        # except now we return the input of the classifier

        features = self.features(input)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        return out
