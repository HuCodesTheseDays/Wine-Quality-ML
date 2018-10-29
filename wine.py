import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# this algo is not exactly giving me the best results
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
define = pd.read_csv(dataset_url, sep=';', quotechar='"')


def taste(quality):
    if quality >= 7:
        return 1
    else:
        return 0


define['tasty'] = define['quality'].apply(taste)

data = define[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
               'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
target = define['tasty']

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.33, random_state=123)

gbmTree = GradientBoostingClassifier(max_depth=5)
gbmTree.fit(data_train, target_train)

gbmTreePerformance = precision_recall_fscore_support(target_test, gbmTree.predict(data_test))

print('Precision, Recall, Fscore, and Support for gradient boosted:')
for treeMethod in [gbmTreePerformance]:
    print('Precision: ', treeMethod[0])
    print('Recall: ', treeMethod[1])
    print('Fscore: ', treeMethod[2])
    print('Support: ', treeMethod[3])
