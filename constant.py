from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

TRAIN_PATHS = [
    'data\\CK\\CKTrain\\ant1train.mat',
    'data\\CK\\CKTrain\\ivy2train.mat',
    'data\\CK\\CKTrain\\jedit4train.mat',
    'data\\CK\\CKTrain\\lucene2train.mat',
    'data\\CK\\CKTrain\\synapse1train.mat',
    'data\\CK\\CKTrain\\velocity1train.mat',
    'data\\CK\\CKTrain\\xalan2train.mat',
    'data\\NASA\\NASATrain\\cm1train.mat',
    'data\\NASA\\NASATrain\\kc3train.mat',
    'data\\NASA\\NASATrain\\mc2train.mat',
    'data\\NASA\\NASATrain\\mw1train.mat',
    'data\\NASA\\NASATrain\\pc1train.mat',
    'data\\NASA\\NASATrain\\pc3train.mat',
    'data\\NASA\\NASATrain\\pc4train.mat',
    'data\\NASA\\NASATrain\\pc5train.mat'
]

TEST_PATHS = [
    'data\\CK\\CKTest\\ant1test.mat',
    'data\\CK\\CKTest\\ivy2test.mat',
    'data\\CK\\CKTest\\jedit4test.mat',
    'data\\CK\\CKTest\\lucene2test.mat',
    'data\\CK\\CKTest\\synapse1test.mat',
    'data\\CK\\CKTest\\velocity1test.mat',
    'data\\CK\\CKTest\\xalan2test.mat',
    'data\\NASA\\NASATest\\cm1test.mat',
    'data\\NASA\\NASATest\\kc3test.mat',
    'data\\NASA\\NASATest\\mc2test.mat',
    'data\\NASA\\NASATest\\mw1test.mat',
    'data\\NASA\\NASATest\\pc1test.mat',
    'data\\NASA\\NASATest\\pc3test.mat',
    'data\\NASA\\NASATest\\pc4test.mat',
    'data\\NASA\\NASATest\\pc5test.mat'
]

DATASET_NAMES = [
    'ant1', 'ivy2', 'jedit4', 'lucene2', 'synapse1', 'velocity1',
    'xalan2', 'cm1', 'kc3', 'mc2', 'mw1', 'pc1', 'pc3', 'pc4', 'pc5'
]

MODELS = [
    ['rf', RandomForestClassifier(n_estimators=10)],
    ['lr', LogisticRegression(penalty='l2', max_iter=5000)],
    ['dt', DecisionTreeClassifier()],
    ['gdbt', GradientBoostingClassifier(n_estimators=200)],
    ['ada', AdaBoostClassifier()]
]
