import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer

from Models.Utils import stratified_group_k_fold, get_distribution, get_distribution_scalars
from Models.Metrics import performance_metrics
from Models.Utils import smote

class XGBoostClassifier():
    def __init__(self, X, y,outcome, grouping):

        self.predicted_probabilities = pd.DataFrame()
        self.X = X
        self.y = y.astype(int)
        self.outcome = outcome
        self.grouping = grouping
        class_distributions = [get_distribution_scalars(y.astype(int))]

        class_weights = class_distributions[0][0] / class_distributions[0][1]

        self.model = xgb.XGBClassifier(scale_pos_weight=class_weights,
                                 learning_rate=0.007,
                                 n_estimators=100,
                                 gamma=0,
                                 min_child_weight=2,
                                 subsample=1,
                                 eval_metric='error')


    def fit(self, label, groups):
        def tn ( y_true, y_pred ) : return confusion_matrix(y_true, y_pred)[0, 0]
        def fp ( y_true, y_pred ) : return confusion_matrix(y_true, y_pred)[0, 1]
        def fn ( y_true, y_pred ) : return confusion_matrix(y_true, y_pred)[1, 0]
        def tp ( y_true, y_pred ) : return confusion_matrix(y_true, y_pred)[1, 1]

        scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn), 'fp' : make_scorer(fp), 'fn': make_scorer(fn)}

        x_columns = ((self.X.columns).tolist())
        X = self.X[x_columns]
        X.reset_index()
        y = self.y
        y.reset_index()

        #X, y  = smote(X, y)

        print(label+" Y distribution after smoting ", get_distribution(y))

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_validate(self.model.fit(X,y), X, y, scoring=['f1_macro', 'precision_macro',
                                                                    'recall_macro'], cv=cv, n_jobs=-4)
        #print('Mean F1 Macro: %.3f' % np.mean(scores['test_tp']))
        print(label+'Mean F1 Macro:', np.mean(scores['test_f1_macro']), 'Mean Precision Macro: ',
              np.mean(scores['test_precision_macro']), 'mean Recall Macro' ,
              np.mean(scores['test_recall_macro']))

        #print(cv_results['test_tp'])
        self.model.fit(X,y)
        #return predicted_Y, predicted_thredholds, predicted_IDs, self.model.feature_importances_


    def predict( self, holdout_X, holdout_y):

        x_columns = ((holdout_X.columns).tolist())
        #x_columns.remove(self.grouping)

        holdout_X = holdout_X[x_columns]
        holdout_X.reset_index()

        yhat = (self.model).predict_proba(holdout_X)[:, 1]
        precision_rt, recall_rt, thresholds = precision_recall_curve(holdout_y, yhat)
        fscore = (2 * precision_rt * recall_rt) / (precision_rt + recall_rt)

        ix = np.argmax(fscore)
        best_threshold = thresholds[ix]
        y_pred_binary = (yhat > thresholds[ix]).astype('int32')

        return y_pred_binary, best_threshold, precision_rt, recall_rt


    def plot_pr( self, precision, recall, label):
        pr_auc =  auc(recall, precision)
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, linewidth=5, label='PR-AUC = %0.3f' % pr_auc)
        plt.plot([0, 1], [1, 0], linewidth=5)

        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title(self.outcome+' Precision Recall Curive-'+label)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        prediction_path = "Run/XGBoost/"

        plt.savefig(prediction_path+self.outcome+label+"precision_recall_auc.pdf", bbox_inches='tight')