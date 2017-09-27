# Utility routines for the MNIST data set.

from sklearn import metrics
from sklearn.externals import joblib
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#--------------------------------------------------------------------------------------------------------
def compute_metrics (classifier, X_test, y_test, classes):
    """
    This function computes and prints various performance measures for a classifier.
    """
    # Use the classifier to make predictions for the test set.
    y_pred = classifier.predict(X_test)

    print('Classes:', classes, '\n')

    # Compute the confusion matrix.
    cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
    print('Confusion matrix, without normalization')
    print(cm, '\n')

    # Normalize the confusion matrix by row (i.e by the number of samples in each class).
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=3, linewidth=132)
    print('Normalized confusion matrix')
    print(cm_normalized, '\n')

    # The confusion matrix as percentages.
    cm_percentage = 100 * cm_normalized
    print('Confusion matrix as percentages')
    print(np.array2string(cm_percentage, formatter={'float_kind':lambda x: "%6.2f" % x}), '\n')
    
    # Precision, recall, and f-score.
    print(metrics.classification_report(y_test, y_pred))

    return cm
#--------------------------------------------------------------------------------------------------------
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix (cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.ion()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#--------------------------------------------------------------------------------------------------------
def read_mnist (file):
    """
    This function reads the MNIST .npy files and returns the feature vectors and their associated
    class labels, and a list of the class labels.
    """
    Z = np.load(file)
    m, n = Z.shape
    X = np.float64(Z[:, 0:n-1])
    y = Z[:, n-1]
    classes = np.unique(y)
    return X, y, classes
#--------------------------------------------------------------------------------------------------------
def save_classifier(clf, filename):
    """
    save_classifier(clf, filename) saves the classifier object clf to the file named filename
    using joblib.dump().
    """
    joblib.dump(clf, filename)
#--------------------------------------------------------------------------------------------------------
def plot_image (X, n):
    """
    plot_image(X, n) displays the image n from the array X.
    """
    plt.figure()
    d = np.reshape(X[n,:], (28, 28)) # Row n.
    plt.imshow(d, cmap="Greys_r")
    title = 'Image ' + str(n)
    plt.title(title)
    plt.show()
