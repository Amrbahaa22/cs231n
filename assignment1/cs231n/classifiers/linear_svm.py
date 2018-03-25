import numpy as np
from random import shuffle

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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
      	# add loss
        loss += margin
      	# update the gradient
        dW[:,j] += X[i]
        dW[:, y[i]] += -X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  delta=1
    # read correct scores into a column array of height N
	
  correct_Score_Class = scores[range(num_train), y]
  correct_Score_Class = np.reshape(correct_Score_Class,(num_train, -1))
    # subtract correct scores from score matrix and add margin
  scores += delta - correct_Score_Class
    # make sure correct scores themselves don't contribute to loss function
  scores[range(num_train), y] = 0
    # construct loss function
  loss /= np.sum(np.maximum(scores, 0))
  loss += 0.5*reg * np.sum(W **2)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  Pos_indices = np.zeros(scores.shape)
  #find the count of the class that has positve indices(i compress the code without appropriate it to additional variable)
  #margin
  Pos_indices[scores > 0] = 1
 
  incorrect_class=np.sum(Pos_indices,axis=1)
  dW=X.T.dot(Pos_indices)
  Pos_indices[range(num_train), y] = incorrect_class* (-1)
  dW += X.T.dot(Pos_indices)
  #calculate the mean of th weight
  dW /= num_train
  dW += 2 * reg * W

  #########################################################W####################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
