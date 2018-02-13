import numpy as np
from random import shuffle

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D=W.shape[0]
    C=W.shape[1]
    N=X.shape[0]
    for i in range(N):
        y_pred=np.dot(X[i],W)
        y_pred=y_pred-np.max(y_pred)
        softmax=np.exp(y_pred[y[i]])/np.sum(np.exp(y_pred))
        loss=loss-np.log(softmax)
        for j in range(C):
            softmax=np.exp(y_pred[j])/np.sum(np.exp(y_pred))
            dW[:,j]=dW[:,j]+(softmax-1*(y[i]==j))*X[i]
    loss=loss/N+reg/2*np.sum(np.dot(np.transpose(W),W))
    dW=dW/N+reg*W


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D=W.shape[0]
    C=W.shape[1]
    N=X.shape[0]
    y_pred=np.dot(X,W)
    y_pred=y_pred-np.max(y_pred, axis=1, keepdims=True)
    softmax=np.exp(y_pred)/np.sum(np.exp(y_pred),axis=1, keepdims=True)
    loss=np.sum(-np.log(softmax[np.arange(N),y]))
    indicator=np.zeros_like(softmax)
    indicator[np.arange(N),y]=1
    dW=np.dot(np.transpose(X),(softmax-indicator))
    loss=loss/N+reg/2*np.sum(np.dot(np.transpose(W),W))
    dW=dW/N+reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

