import theano.tensor as T

from pylearn2.utils import wraps
from pylearn2.models.mlp import Layer
from pylearn2.models.mlp import PretrainedLayer

class PretrainedLayerWeight(PretrainedLayer):

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.layer_content.get_weights()
        return coeff * T.sqr(W).sum()