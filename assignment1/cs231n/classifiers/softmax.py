from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_dim = X.shape[1]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        f = X[i].dot(W)
        f -= np.max(f)
        score = np.exp(f) / np.sum(np.exp(f))
        fi = score[y[i]]
        # if fi != 0:
        loss += -1 * np.log(fi)
        for j in range(num_class):
            if j != y[i]:
                dW[:, j] += score[j] * X[i]
            else:
                dW[:, j] += (score[j] - 1) * X[i]
        # print()
    loss = np.sum(loss) / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def softmax(W, x):
    f = np.exp(x.dot(W))
    f -= np.max(f)
    return f / np.sum(f)


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    f = X.dot(W)
    f -= np.max(f, axis = 1).reshape(num_train, -1)
    e_y = np.exp(f)
    sum_e_y = np.sum(e_y, axis = 1)
    loss = e_y[np.arange(num_train), y] / sum_e_y
    loss = -1 * np.log(loss)
    loss = np.sum(loss) / num_train + 0.5 * reg * np.sum(W * W)

    zero_mask = np.zeros((num_train, num_class))
    S = e_y / (sum_e_y.reshape(num_train, -1))
    zero_mask[np.arange(num_train), y] = 1
    dW = X.T.dot(S - zero_mask) / num_train + reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
