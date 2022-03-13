"""
Train and save a GAM model based on the label column.
"""
# pylint: disable=too-many-locals, invalid-name
import pickle
import pandas as pd # type:ignore
from pygam import LogisticGAM # type:ignore
from sklearn import metrics # type:ignore
from sklearn.model_selection import train_test_split # type:ignore

def run(labeled_score_file: str, model_file: str) -> None:
    """ Read all the training at once (it's small, handmade), train, and save."""

    labeled_scores  = pd.read_csv(labeled_score_file, index_col=0)
    X = labeled_scores.drop(columns=labeled_scores.columns[0:3]) # pylint:disable=no-member
    y = labeled_scores['label'] # pylint:disable=unsubscriptable-object

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=62)

    model = LogisticGAM(constraints=None, n_splines=15).gridsearch(X_train.values, y_train.values)

    predictions = model.predict(X_test.values)
    print(f"Accuracy {metrics.accuracy_score(y_test.values, predictions):.3f}")

    cm = metrics.confusion_matrix(y_test.values, predictions)
    print("confusion matrix")
    print("  TN   FP")
    print("  FN   TP")
    print(cm)

    ravel = cm.ravel()
    if len(ravel) == 4:
        tn, fp, fn, tp = ravel
        print(f"precision {tp/(tp+fp):.3f} recall {tp/(tp+fn):.3f}")
        print(f"false positive rate {fp/(tn+fp):.3f} false negative rate {fn/(tp+fn):.3f}")

    with open(model_file, 'wb') as model_f:
        pickle.dump(model, model_f)

if __name__ == '__main__':
    run('sample-data/sample-labeled-scores.csv', 'sample-data/sample-model.pkl')
