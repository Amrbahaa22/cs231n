# Starter codes from Assignment 1 of Stanford code is used.
import random
import time
import numpy as np
from read import load_CIFAR10
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Load the raw CIFAR-10 data.
# download the CIFAR10 data
cifar10_dir = 'cifar10data'
print 'Loading CIFAR10 data...'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print ''
print 'Checking the shape of data.'
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape
print ''
print 50*'#'

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
print 'Visualizing some examples from the dataset'
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 5
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.savefig('samples.png') 
print 'Saving figure into samples.png'
#plt.show()


# Subsample the data for more efficient code execution in this exercise
print ''
print 50*'#'
print 'Subsampling and reshaping the image data into rows.'
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print 'Saving class distribution plot into distrubution.png.'
plt.figure()
tick_label = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.bar(range(10), np.histogram(y_train)[0], tick_label=tick_label)
plt.title('Class distribution in training set')
plt.bar(range(10), np.histogram(y_test)[0], tick_label=tick_label)
plt.title('Classes distribution in subsampled data')
plt.xlabel('classes')
plt.ylabel('Number of samples')
plt.legend(['Training set', 'Test set'])
plt.savefig('distribution.png')


# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print 'Checking the shape of sampled data.'
print 'Subsample Training data shape: ', X_train.shape
print 'Subsample Training labels shape: ', y_train.shape
print 'Subsample Test data shape: ', X_test.shape
print 'Subsample Test labels shape: ', y_test.shape
print ''

print 50*'#'
from kNearestNeighbor import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
print 'Creating a kNN classifier instance'
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


# Test how fast three implementations are and if we implement them correct
print 'Comparing the different implementations speed'
tic = time.time()
dists_two_loops = classifier.compute_distances_two_loops(X_test)
toc = time.time()
print 'Two loops version took {:.2f} seconds'.format(toc - tic)

tic = time.time()
dists_one_loop = classifier.compute_distances_one_loop(X_test)
toc = time.time()
print 'One loop version took {:.2f} seconds'.format(toc - tic)

tic = time.time()
dists_no_loops = classifier.compute_distances_no_loops(X_test)
toc = time.time()
print 'No loops version took {:.2f} seconds'.format(toc - tic)

dists = dists_no_loops

print 'Distance matrix shape: ', dists.shape

print ''
print 50*'#'
# Testing if the distances matrix implementations are correct
difference = np.linalg.norm(dists - dists_one_loop, ord='fro')
print 'Difference of no loops and one loop implementations was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'


difference = np.linalg.norm(dists - dists_two_loops, ord='fro')
print 'Difference of no loops and two loop implementations was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'



# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
print 'Visualizing the aucludian distance matrix'
plt.figure()
plt.imshow(dists, interpolation='none')
plt.title('Eucludian distance matrix visualization')
plt.xlabel('Train images')
plt.ylabel('Test images')
print 'Saving figure into distancematrix.png'
plt.savefig('distancematrix.png')
#plt.show()


# Now implement the function predict_labels and run the code below:
# The function is implemented in kNearestNeghbor.py
# We use k = 1 (which is Nearest Neighbor).
print ''
print 50*'#'
print 'Predicting Test labels using k=1 Nearest Neighbors.'
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

print 'Predicting Test labels using k=5 Nearest Neighbors.'
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)


# Cross validation
print ''
print 50*'#'
print '5 folds cross validation...'


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# Split the trining set into 5 folds
X_train_folds = np.split(X_train, num_folds, axis=0)
# Split the training set labels accordingly
y_train_folds = np.split(y_train, num_folds, axis=0)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################

tic = time.time()

for k in k_choices:
    # iterate through folds
    for fold in range(num_folds):
        
        # select validation set 
        X_val = X_train_folds[fold]
        # select one of the folds as validation set and its corresponding labels
        y_val = y_train_folds[fold]
        
        # list of remaining folds and their labels
        X_remaining_folds = X_train_folds[:fold] + X_train_folds[fold+1:]
        y_remaining_folds = y_train_folds[:fold] + y_train_folds[fold+1:]
        
        # concatenate remaining folds to form the training set and coressponding labels
        X_train_cross = np.concatenate(X_remaining_folds)
        y_train_cross = np.concatenate(y_remaining_folds)
        
        # create a kNN classifier instance
        clf = KNearestNeighbor()
        clf.train(X_train_cross, y_train_cross)
        
        # compute the distances matrix
        dists = clf.compute_distances_no_loops(X_val)
        
        # We use k = 1 (which is Nearest Neighbor).
        y_val_pred = clf.predict_labels(dists, k=k)

        # Compute the fraction of correctly predicted examples
        num_correct = np.sum(y_val_pred == y_val)
        
        # calculate the accuracy for this fold
        accuracy = float(num_correct) / X_val.shape[0]
        
        # add the accuracy of the validation fold to dictionary
        k_to_accuracies.setdefault(k, []).append(accuracy)
    
toc = time.time()

print "Cross validation took {:.2f} seconds".format(toc-tic)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
	print '\nAccuracies of each 5 folds with k={}'.format(k)
	for accuracy in k_to_accuracies[k]:
		print 'accuracy = %f' % (accuracy)



# plot the raw observations
plt.figure()
print ''
print 'Plotting Cross-validation accuracy against k'
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
print 'Saving figure into accuracy_vs_k.png'
plt.savefig('accuracy_vs_k.png')


# Find the best k based on cross validation
# find the k with max accuracy
k_to_accuracy = {}
for k in k_to_accuracies:
	k_to_accuracy[k] = np.mean(k_to_accuracies[k])

import operator
best_k = int(max(k_to_accuracy.iteritems(), key=operator.itemgetter(1))[0])
print 'Accuracy of validation set is maximum with k = {}.'.format(best_k)



# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
num_test = X_test.shape[0]
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f with k=%d' % (num_correct, num_test, accuracy, best_k)

