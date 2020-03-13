import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


def evaluator(classifier, ypred, ytest):
    errors = abs(ypred - ytest)
    val = np.count_nonzero(errors)
    print('=== Classifier: ' + classifier + ' ===')
    #print("M A E: ", np.mean(errors))
    print(np.count_nonzero(errors), len(ytest))
    print('Accuracy (%): ' + str(val / len(ytest)))
    
    return str(val / len(ytest))
    
    
def lda_classifier(Xtrain, ytrain, Xtest, ytest):
    clf = LinearDiscriminantAnalysis()
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    return evaluator('LDA', ypred, ytest)

    
def qda_classifier(Xtrain, ytrain, Xtest, ytest):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    return evaluator('QDA', ypred, ytest)
    
    
def rf_classifier(Xtrain, ytrain, Xtest, ytest):
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
    clf.fit(Xtrain, ytrain)
    #print(clf.feature_importances_)
    # print(clf.oob_decision_function_)
    #print(clf.oob_score_)
    ypred = clf.predict(Xtest)
    return evaluator('RF', ypred, ytest)