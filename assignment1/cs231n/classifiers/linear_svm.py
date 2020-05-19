from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    num_features = W.shape[0]
    x_row_sum = np.zeros(num_features)
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if j == y[i]:
                continue
            
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
        

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW /= num_train

    dW += reg  * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_dim = W.shape[0]
    num_train = X.reshape(-1, num_dim).shape[0]

    scores = X.dot(W)
    # print(X.shape, W.shape, y.shape, scores.shape, num_train)
    y_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    
    margin = scores - y_scores + 1
    margin[list(range(num_train)), y] = 0
    mask = margin > 0
    margin_mask = margin[mask]

    loss = np.sum(margin_mask) / num_train + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    trans_matrix = np.zeros(margin.shape)
    trans_matrix[margin > 0]  = 1

    neg_count = np.sum(trans_matrix, axis = 1)
    trans_matrix[np.arange(num_train), y] = -1 * neg_count

    dW = X.T.dot(trans_matrix) / num_train + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

if __name__ == '__main__':
    W = np.array([[1, 2], 
                    [2, 3], 
                    [3, 4]])
    X = np.array([[1, 2, 3], [3, 4, 5], [1, 3, 4]])
    y = np.array([0, 1, 0])
    score = X.dot(W)
    print(score)
    y_socre = score[[0, 1, 2], y]
    print(y_socre)
    margin = score - y_socre.reshape(3, 1) + 1
    score[[0, 1, 2], y] = 0
    print(margin)
    trans = np.zeros(margin.shape)
    trans[margin > 0] = 1
    neg_count = np.sum(trans, axis = 1)
    trans[np.arange(3), y] = -1 * neg_count
    print(trans)
    print(neg_count)

    
    

