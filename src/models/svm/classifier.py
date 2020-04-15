"""
ASL classifier
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


class ASLClassifier:
    """
    ASL classifier
    """

    def __init__(self, config):
        """
        Initialize this classifier with the given configuration
        Input:
            config: a YAML node with the classifier configuration
        """
        if config["classifier"] == "adaboost":
            self.__clf = self.__init_adaboost(config["adaboost"])

    def __init_adaboost(self, config):
        """
        Create an AdaBoost classifier using decision trees
        Input:
            config: YAML node with hyperparameters
        Output:
            clf: An AdaBoost classifier
        """
        tree_depth = config["tree_max_depth"]
        n_estimators = config["n_estimators"]
        learning_rate = config["learning_rate"]
        print("Initializing Adaboost classifier with")
        print("\tMax decision tree depth: {}".format(tree_depth))
        print("\tNumber of estimators: {}".format(n_estimators))
        print("\tLearning rate: {}".format(learning_rate))

        base_estimator = DecisionTreeClassifier(max_depth=tree_depth)
        clf = AdaBoostClassifier(base_estimator=base_estimator,
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate)
        return clf

    def fit(self, X, y):
        """
        Classify given data
        Input:
            X: (N, M) matrix of N training data samples
            y: (N) vector of N training data labels
        """
        self.__clf.fit(X, y)

    def predict(self, X):
        """
        Predict the labels of the given data
        Input:
            X: (N, M) matrix of N test data samples
        Output:
            y: (N) vector of N predicted labels
        """
        return self.__clf.predict(X)

    def score(self, X, y):
        """
        Calculate the mean accuracy on the given test data and labels.
        Input:
            X: (N, M) matrix of N test data samples
        Output:
            score: mean accuracy on the given test data and labels.
        """
        return self.__clf.score(X, y)

    def cross_val_score(self, X, y, k, nprocs=1):
        """
        Evaluate model by stratified k-fold cross validation
        Input:
            X: (N, M) matrix of N training data samples
            y: (N) vector of N training data labels
            k: number of folds
            nprocs: number of processes to use
        Output:
            score: Array of scores of the estimator for each fold
        """
        print("Performing {}-fold cross validation using {} processes".format(
            k, nprocs))
        return cross_val_score(self.__clf, X, y, cv=k, n_jobs=nprocs)
