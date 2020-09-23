import pandas as pd

class DecisionMaker():

    def __init__(self):
        self.data = pd.DataFrame()
        self.no_classifiers = 0

    def add_classifier( self, name, predicted, thresholds, ids, imp= None):
        if self.no_classifiers == 0:
            self.data[name+'ids'] = ids
            self.data[name+'predicted'] = predicted
            self.data[name+'thresholds'] = thresholds

        else:
            new_classifier = pd.DataFrame()
            new_classifier[name+'ids'] = ids
            new_classifier[name+'predicted'] = predicted
            new_classifier[name+'thresholds'] = thresholds
            self.data = pd.concat([self.data, new_classifier], axis=1, join='inner')

        self.no_classifiers = self.no_classifiers+1