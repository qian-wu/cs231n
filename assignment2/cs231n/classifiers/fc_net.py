from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = np.random.normal(0.0, weight_scale, size = (input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = np.random.normal(0.0, weight_scale, size = (hidden_dim, num_classes))
        b2 = np.zeros(num_classes)

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        h1_out, h1_cache = affine_forward(X, W1, b1)
        h1_relu, relu_cache = relu_forward(h1_out)
        h2_out, h2_cache = affine_forward(h1_relu, W2, b2)
        scores = h2_out
        out, dout = softmax_loss(scores, y)
        
        # print(loss)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss = out + 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        # dout = dout + self.reg * (np.sum(W1) + np.sum(W2))
        dh2, dw2, db2 = affine_backward(dout, h2_cache)
        dw2 = dw2 + self.reg * W2
        drelu = relu_backward(dh2, relu_cache)
        dh1, dw1, db1 = affine_backward(drelu, h1_cache)
        dw1 = dw1 + self.reg * W1

        grads['W2'] = dw2
        grads['b2'] = db2
        grads['W1'] = dw1
        grads['b1'] = db1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        num_hiddens = len(hidden_dims)
        if num_hiddens == 0 :
            self.params['W1'] = np.random.normal(0.0, weight_scale, size = (input_dim, num_classes)).astype(self.dtype)
            self.params['b1'] = np.zeros(num_classes).astype(self.dtype)
        if num_hiddens > 0 :
            max_layer = 1
            self.params['W1'] = np.random.normal(0.0, weight_scale, size = (input_dim, hidden_dims[0])).astype(self.dtype)
            self.params['b1'] = np.zeros(hidden_dims[0]).astype(self.dtype)
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
                self.params['gamma1'] = np.ones(hidden_dims[0])
                self.params['beta1'] = np.zeros(hidden_dims[0])
            if num_hiddens > 1 :
                for idx in range(1, len(hidden_dims)) :
                    w_name = 'W' + str(idx + 1)
                    b_name = 'b' + str(idx + 1)
                    self.params[w_name] = np.random.normal(0.0, weight_scale, 
                                                            size = (hidden_dims[idx - 1], hidden_dims[idx])).astype(self.dtype)
                    self.params[b_name] = np.zeros(hidden_dims[idx]).astype(self.dtype)

                    if self.normalization == "batchnorm" or self.normalization == "layernorm":
                        self.params['gamma' + str(idx + 1)] = np.ones(hidden_dims[idx])
                        self.params['beta' + str(idx + 1)] = np.zeros(hidden_dims[idx])

                    max_layer += 1
            self.params['W' + str(max_layer + 1)] = np.random.normal(0.0, weight_scale, 
                                                                        size = (hidden_dims[max_layer - 1], num_classes)).astype(self.dtype)
            self.params['b' + str(max_layer + 1)] = np.zeros(num_classes).astype(self.dtype)

            # print([(key, value.shape) for key, value in self.params.items()])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # print([(key, value.shape) for key, value in self.params.items()])
        cache_list = []
        W_sum = 0.0
        if self.num_layers == 1 :
            W1 = self.params['W1']
            b1 = self.params['b1']
            W_sum = np.sum(W1 * W1)
            scores, cache = affine_forward(X, W1, b1)
        if self.num_layers > 1 :
            scores = X
            for idx in range(1, self.num_layers) :
                W_name, b_name = 'W' + str(idx), 'b' + str(idx) 
                W_tmp = self.params[W_name]
                b_tmp = self.params[b_name]
                # print('get %s %s shape %s %s' % (W_name, b_name, W_tmp.shape, b_tmp.shape))
                af_score, af_cache = affine_forward(scores, W_tmp, b_tmp)
                # add batchnorm
                if self.normalization == "batchnorm":
                    gamma_name, beta_name = 'gamma' + str(idx), 'b' + str(idx)
                    bn_score, bn_cache = batchnorm_forward(af_score, self.params[gamma_name], 
                                                            self.params[beta_name], self.bn_params[idx - 1])
                    relu_score, relu_cache = relu_forward(bn_score)
                    cache_list.append((af_score, af_cache, bn_score, bn_cache, relu_score, relu_cache))
                if self.normalization == "layernorm":
                    gamma_name, beta_name = 'gamma' + str(idx), 'b' + str(idx)
                    ln_score, ln_cache = layernorm_forward(af_score, self.params[gamma_name], 
                                                            self.params[beta_name], self.bn_params[idx - 1])
                    relu_score, relu_cache = relu_forward(ln_score)
                    cache_list.append((af_score, af_cache, ln_score, ln_cache, relu_score, relu_cache))
                else :
                    relu_score, relu_cache = relu_forward(af_score)
                    cache_list.append((af_score, af_cache, relu_score, relu_cache))
                W_sum += np.sum(W_tmp * W_tmp)
                scores = relu_score

            W_end, b_end = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
            scores, cache_end = affine_forward(scores, W_end, b_end)
            W_sum += np.sum(W_end * W_end)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = scores, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out, dout = softmax_loss(scores, y)
        loss = out + 0.5 * self.reg * W_sum

        dout, dw_end, db_end = affine_backward(dout, cache_end)
        dw_end += self.reg * self.params['W' + str(self.num_layers)]
        grads['W' + str(self.num_layers)] = dw_end
        grads['b' + str(self.num_layers)] = db_end
        # print('W%d %s b%d %s' % (self.num_layers, dw_end.shape, self.num_layers, db_end.shape))

        if self.num_layers > 1 :
            tmp_score = dout
            for idx in reversed(range(1, self.num_layers)):

                if self.normalization == "batchnorm":
                    af_score, af_cache, bn_score, bn_cache, relu_score, relu_cache = cache_list.pop()
                elif self.normalization == "layernorm":
                    af_score, af_cache, ln_score, ln_cache, relu_score, relu_cache = cache_list.pop()
                else:
                    af_score, af_cache, relu_score, relu_cache = cache_list.pop()

                tmp = relu_backward(tmp_score, relu_cache)

                if self.normalization == "batchnorm":
                    tmp, dgamma, dbeta = batchnorm_backward_alt(tmp, bn_cache)
                elif self.normalization == "layernorm":
                    tmp, dgamma, dbeta = layernorm_backward(tmp, ln_cache)

                tmp_score, dw, db = affine_backward(tmp, af_cache)
                dw += self.reg * self.params['W' + str(idx)]
                grads['W' + str(idx)] = dw
                grads['b' + str(idx)] = db

                if self.normalization == "batchnorm" or self.normalization == "layernorm":
                    grads['gamma' + str(idx)] = dgamma
                    grads['beta' + str(idx)] = dbeta
                # print('dW%d %s db%d %s' % (idx, dw.shape, idx, db.shape))

            # if self.normalization == "batchnorm":
            #     for idx in reversed(range(1, self.num_layers)):
            #         af_score, af_cache, bn_score, bn_cache, relu_score, relu_cache = cache_list.pop()
            #         # print(af_score.shape, relu_score.shape)
            #         drelu = relu_backward(tmp_score, relu_cache)
            #         dbn, dgamma, dbeta = batchnorm_backward_alt(drelu, bn_cache)
            #         tmp_score, dw, db = affine_backward(dbn, af_cache)
            #         dw += self.reg * self.params['W' + str(idx)]
            #         grads['W' + str(idx)] = dw
            #         grads['b' + str(idx)] = db
            #         grads['gamma' + str(idx)] = dgamma
            #         grads['beta' + str(idx)] = dbeta
            #         # print('dW%d %s db%d %s' % (idx, dw.shape, idx, db.shape))
            # elif self.normalization == "layernorm":
            #     for idx in reversed(range(1, self.num_layers)):
            #         af_score, af_cache, ln_score, ln_cache, relu_score, relu_cache = cache_list.pop()
            #         # print(af_score.shape, relu_score.shape)
            #         drelu = relu_backward(tmp_score, relu_cache)
            #         dln, dgamma, dbeta = layernorm_backward(drelu, ln_cache)
            #         tmp_score, dw, db = affine_backward(dln, af_cache)
            #         dw += self.reg * self.params['W' + str(idx)]
            #         grads['W' + str(idx)] = dw
            #         grads['b' + str(idx)] = db
            #         grads['gamma' + str(idx)] = dgamma
            #         grads['beta' + str(idx)] = dbeta
            # else :
            #     for idx in reversed(range(1, self.num_layers)):
            #         af_score, af_cache, relu_score, relu_cache = cache_list.pop()
            #         # print(af_score.shape, relu_score.shape)
            #         drelu = relu_backward(tmp_score, relu_cache)
            #         tmp_score, dw, db = affine_backward(drelu, af_cache)
            #         dw += self.reg * self.params['W' + str(idx)]
            #         grads['W' + str(idx)] = dw
            #         grads['b' + str(idx)] = db
            #         # print('dW%d %s db%d %s' % (idx, dw.shape, idx, db.shape))


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
