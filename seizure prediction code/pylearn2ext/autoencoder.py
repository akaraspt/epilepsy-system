import theano.sparse
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from theano.tensor.shared_randomstreams import RandomStreams

class WeightDecay(DefaultDataSpecsMixin, Cost):

    def __init__(self, coeff):
        self.coeff = coeff

    def expr(self, model, data, ** kwargs):
        # penalty on weights
        params = model.get_params()
        W = params[2]
        L1_weights = theano.tensor.sqr(W).sum()
        cost = self.coeff * L1_weights
        cost.name = 'L2_weight_decay'
        return cost
