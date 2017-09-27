#Mnist tuning and classifying script with helper functions


# IMPORTS #
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from mnist_utils import (compute_metrics, plot_confusion_matrix, read_mnist, 
                         save_classifier)

# FUNCTIONS #
def load_data():
    ##Inputs are None.
    ##Outputs original X's, scaled X's (0,1), y, and classes.
    print('reading data...', end='')
    X,y,classes = read_mnist('mnist_public.npy') #Imports unscaled data.
    print('done!')
    min_max_scaler = MinMaxScaler() #Scaling function
    X_minmax = min_max_scaler.fit_transform(X) #Scales data to (0,1).
    return X, X_minmax, y, classes

def org_data(X_minmax,y,testfrac,tunefrac):
    ##Inputs are scaled X data, y, portion saved for testing, and percent 
    ##desired for tuning.
    ##Outputs are Xtrain and Xtest sets with their corresponding ys. 
    Xdf = pd.DataFrame(np.transpose(X_minmax)) #Moves to a dataframe.
    Xdf_test = Xdf.sample(frac=testfrac, random_state=0, axis=1) #Sets aside test portion.
    Xdf_train = Xdf.drop(Xdf_test.index.values, axis=1) #Removes test portion from training portion.
    Xdf_train_tune = Xdf_train.sample(frac=tunefrac, random_state=0, axis=1) #Uses only tunefrac fraction of total data.
    Xdf_test_tune = Xdf_test.sample(frac=tunefrac, random_state=0, axis=1) #Uses only tunefrac fraction of total data.
    Xtraindf = Xdf_train_tune
    Xtestdf = Xdf_test_tune
    Xtrain = np.transpose(Xtraindf.as_matrix()) #Returns data to numpy array.
    Xtest = np.transpose(Xtestdf.as_matrix()) #Returns data to numpy array.
    ytrain = y[Xtraindf.columns] #Finds training ys.
    ytest = y[Xtestdf.columns] #Finds testing ys.
    return Xtrain, ytrain, Xtest, ytest


def lgrtune(X_minmax, y, classes, testfrac, tunefrac, cparams):
    Xtrain, ytrain, Xtest, ytest = org_data(X_minmax, y, testfrac, tunefrac)
    ##Inputs are data needed to train and hyperparameter values to try.
    ##Outputs report of max misclassification, % of mistakes by class, time
    ##elapsed, confustion matrixes, and summary strings for each hyperparameter
    mistakes = [] #Preallocate
    report = [] #Preallocate
    sums = [] #Preallocate
    elapsed = [] #Preallocate
    cmlgrs = [] #Preallocate
    for it in range(len(cparams)): #Loop for hyperparameter values.
        t = time.time() #Start timer.
        lgr = LogisticRegression(C=cparams[it], random_state=0) #Classifier
        lgr.fit(Xtrain, ytrain) #Classifier training.
        cmlgr = compute_metrics(lgr, Xtest, ytest, classes) #Append confusion matrix.
        cmlgrs.append(cmlgr)
        report.append(np.round(max(1-cmlgr.diagonal()/cmlgr.sum(axis=1)), 3))#Appends max misclassification value.
        mistakes.append((cmlgr.sum(axis=1)-cmlgr.diagonal())/sum(cmlgr.sum(axis=1)-cmlgr.diagonal())) #Append the percent of misclassifications broken down by class.
        elapsed.append(np.round(time.time()-t,3)) #Stop timer.
        print('C value: ', cparams[it], '\n', 'elapsed (s): ', elapsed[it])       
    for it in range(len(cparams)):
        print('\n')
        sums.append('Metrics reports for C value of ' +  str(cparams[it]) + 
                       '\n Time elapsed (s): ' + str(elapsed[it]) +  
                       '\n Max misclassification rates: ' +  
                       np.array2string(report[it]) +
                       '\n Mistake %s: ' + np.array2string(mistakes[it])) #Append summary for hyperparmeter value
        print(sums[it])
    return report, mistakes, elapsed, cmlgrs, sums
        
def svmtune(X_minmax, y, classes, testfrac, tunefrac, cparams):
    Xtrain, ytrain, Xtest, ytest = org_data(X_minmax, y, testfrac, tunefrac)
    ##Inputs are data needed to train and hyperparameter values to try.
    ##Outputs report of max misclassification, % of mistakes by class, time
    ##elapsed, confustion matrixes, and summary strings for each hyperparameter
    mistakes = [] #Preallocate
    report = [] #Preallocate
    sums = [] #Preallocate
    elapsed = [] #Preallocate
    cmsvms = [] #Preallocate
    for it in range(len(cparams)): #Loop for hyperparameter values.
        t = time.time() #Start timer.
        svm = SVC(kernel='linear', class_weight='balanced', C=cparams[it], random_state=0) #Classifier, 'balenced' adjusts C by a weighted average of frequency of each class.
        svm.fit(Xtrain, ytrain) #Classifier training. 
        cmsvm = compute_metrics(svm, Xtest, ytest, classes) #Append confusion matrix.
        cmsvms.append(cmsvm)
        report.append(np.round(max(1-cmsvm.diagonal()/cmsvm.sum(axis=1)), 3)) #Appends max misclassification value.
        mistakes.append((cmsvm.sum(axis=1)-cmsvm.diagonal())/sum(cmsvm.sum(axis=1)-cmsvm.diagonal())) #Append the percent of misclassifications broken down by class. 
        elapsed.append(np.round(time.time()-t,3)) #Stop timer.
        print('C value: \n', cparams[it], '\n', 'elapsed (s): \n', elapsed[it])        
    for it in range(len(cparams)):
        print('\n')
        sums.append('Metrics reports for C value of ' +  str(cparams[it]) + 
                    '\n Time elapsed (s): ' + str(elapsed[it]) +  
                    '\n Max misclassification rates: ' + 
                    np.array2string(report[it]) +
                    '\n Mistakes %s: ' + np.array2string(mistakes[it])) #Append summary for hyperparmeter value
        print(sums[it])
    return report, mistakes, elapsed, cmsvms, sums
    ## SVM hyperparameters have low effect on misclassification rates or times here

def treetune(X_minmax, y, classes, testfrac, tunefrac, depth):
    Xtrain, ytrain, Xtest, ytest = org_data(X_minmax, y, testfrac, tunefrac)
    ##Inputs are data needed to train and hyperparameter values to try.
    ##Outputs report of max misclassification, % of mistakes by class, time
    ##elapsed, confustion matrixes, and summary strings for each hyperparameter
    mistakes = [] #Preallocate
    report = [] #Preallocate
    sums = [] #Preallocate
    elapsed = [] #Preallocate
    cmtrees = [] #Preallocate
    for it in range(len(depth)): #Loop for hyperparameter values.
        t = time.time() #Start timer.
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth[it], random_state=0) #Classifier
        tree.fit(Xtrain, ytrain) #Classifier training.
        cmtree = compute_metrics(tree, Xtest, ytest, classes) #Append the confusion matrix.
        cmtrees.append(cmtree)
        report.append(np.round(max(1-cmtree.diagonal()/cmtree.sum(axis=1)), 3))#Appends max misclassification value.
        mistakes.append((cmtree.sum(axis=1)-cmtree.diagonal())/sum(cmtree.sum(axis=1)-cmtree.diagonal()))  #Append the percent of misclassifications broken down by class.
        elapsed.append(np.round(time.time()-t,3)) #Stop timer.
        print('Depth value: \n', depth[it], '\n', 'elapsed (s): \n', elapsed[it])  
    for it in range(len(depth)):
        print('\n')
        sums.append('Metrics reports for depth value of ' +  str(depth[it]) + 
                    '\n Time elapsed (s): ' + str(elapsed[it]) +  
                    '\n Max misclassification rates: ' +  
                    np.array2string(report[it]) +
                    '\n Mistake %s: ' + np.array2string(mistakes[it])) #Append summary for hyperparmeter value
        print(sums[it])
    return report, mistakes, elapsed, cmtrees, sums
    ## tree depth seems to matter a large amount, especially for small values.
    ## Training time is extremely short compared to others if you give all of the 
    ## data to train. 
    
def lgr_gen(X_minmax, y, classes, c_lgr, testfrac, lgrfrac):
    Xtrain, ytrain, Xtest, ytest = org_data(X_minmax, y, testfrac, lgrfrac)
    ##Inputs are data needed to train with optimized hyperparameter values.
    ##Outputs lgrclassifier
    lgr = LogisticRegression(C=c_lgr, random_state=0) #Classifier
    lgr.fit(Xtrain, ytrain) #Classifier training.
    save_classifier(lgr, 'lgr.pkl') #Save classifier as .pkl file
    return lgr

def svm_gen(X_minmax, y, classes, c_svm, testfrac, svmfrac):
    Xtrain, ytrain, Xtest, ytest = org_data(X_minmax, y, testfrac, svmfrac)
    ##Inputs are data needed to train with optimized hyperparameter values.
    ##Outputs lgrclassifier
    svm = SVC(kernel='linear', class_weight='balanced', C=c_svm, random_state=0) #Classifier, 'balenced' adjusts C by a weighted average of frequency of each class.
    svm.fit(Xtrain, ytrain) #Classifier training.
    save_classifier(svm, 'svm.pkl')#Save classifier as .pkl file
    return svm

def tree_gen(X_minmax, y, classes, depth_tree, testfrac, treefrac):
    Xtrain, ytrain, Xtest, ytest = org_data(X_minmax, y, testfrac, treefrac)
    ##Inputs are data needed to train with optimized hyperparameter values.
    ##Outputs lgrclassifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth_tree, random_state=0) #Classifier
    tree.fit(Xtrain, ytrain) #Classifier training.
    save_classifier(tree, 'tree.pkl')#Save classifier as .pkl file
    return tree

# TUNING #
testfrac = .1 #Sets aside 10% of data for testing.
tunefrac_lgr = .12 #Use 12% of total X_minmax set for lgr tuning (lgr is very slow).
tunefrac_svm = .25 #Use 25% of total X_minmax set for svm tuning (svm is slow).
tunefrac_tree = 1 #Use 100% of total X_minmax set for tree tuning (tree is fast).
cparams = [.01, .1, 1, 10, 100, 1000, 10000] #Range of penalty values for misclassification.
depths = [2, 3, 4, 5, 6, 8, 10, 50, 100] #Range of max tree depths for tree tuning
X, X_minmax, y, classes = load_data() #Loads mnist data
lgrreports, lgrmistakes, lgrelapsed, cmlgrs, lgrsums = lgrtune(X_minmax, y, classes, testfrac, tunefrac_lgr, cparams) #Tuning lgr.
svmreports, svmmistakes, svmelapsed, cmsvms, svmsums = svmtune(X_minmax, y, classes, testfrac, tunefrac_svm, cparams) #Tuning svm.
treereports, treemistakes, treeelapsed, cmtrees, treesums = treetune(X_minmax, y, classes, testfrac, tunefrac_tree, depths) #Tuning tree.

# OPTIMIZATION VALUES #
lgrbestind = lgrreports.index(min(lgrreports)) #Index of most accurate run.
svmbestind = svmreports.index(min(svmreports)) #Index of most accurate run.
treebestind = treereports.index(min(treereports)) #Index of most accurate run.
lgrbestsums = lgrsums[lgrbestind] #Summary for tuned lgr.
svmbestsums = svmsums[svmbestind] #Summary for tuned svm.
treebestsums = treesums[treebestind] #Summary for tuned tree.
lgrC = cparams[lgrbestind] #Tuned C value for lgr
lgrrate = lgrreports[lgrbestind]
print('Optimal C for lgr: ', lgrC,
      'Minimum max misclassification: ', lgrrate)
plot_confusion_matrix(cmlgrs[lgrbestind], classes)
svmC = cparams[svmbestind] #Tuned C value for svm
svmrate = svmreports[svmbestind]
print('Optimal C for svm: ', svmC,
      'Minimum max misclassification: ', svmrate)
plot_confusion_matrix(cmsvms[svmbestind], classes)
treedepth = depths[treebestind] #Tuned max tree depth for tree
treerate = treereports[treebestind] #Minimal max misclassification
print('Optimal Max Tree Depth for tree: ', treedepth,
      'Minimum max misclassification: ', treerate)
plot_confusion_matrix(cmtrees[treebestind], classes)

# GENERATING FINAL CLASSIFIERS #
testfrac = .01 #Use 99% of the data for training.
lgrfrac = .15 #lgr trains slowly.
svmfrac = .3 #svm also trains slowly, but not too badly.
treefrac = 1 #tree trains quickly.
c_lgr = lgrC #Use best tuned C value.
c_svm = svmC #Use best tuned C value.
depth_tree = treedepth #Use best tuned depth value.
lgr = lgr_gen(X_minmax, y, classes, c_lgr, testfrac, lgrfrac) #Generate and save lgr.pkl
svm = svm_gen(X_minmax, y, classes, c_svm, testfrac, svmfrac) #Generate and save svm.pkl
tree = tree_gen(X_minmax, y, classes, depth_tree, testfrac, treefrac) #Generate and save tree.pkl
