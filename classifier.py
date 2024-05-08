import numpy as np
import pandas as pd
import array
from collections import Counter, defaultdict

'''
Link of guide 
https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9 
'''

class NBclassifier:
    def __init__(self, y_train, x_train):
        self.class_priors = {}
        self.likelihoods = defaultdict(dict)
        self.pred_poiors = {}
        self.features = {}
        self.y_train = y_train
        self.x_train = x_train

    def fit(self):
        self.features = self.x_train.columns
        unique_classes = np.unique(self.y_train)
        num_classes = len(unique_classes)

        self.clac_class_priors()
        self.calc_likehood()

    
    def clac_class_priors(self):
        '''
            Calculate Prior Class Probability for each class - P(class)
        '''
        for outcome in np.unique(self.y_train):  # iterate over each unique class 
            outcome_count = sum(self.y_train == outcome)  # count occurncace of currect class 
            self.class_priors[outcome] = outcome_count / len(self.y_train)  # claculate and store class prior for particular class(outcome)
            print(f'The prior of class {outcome} is:  {self.class_priors[outcome]} \n')
        

    
    def calc_likehood(self):
        '''
            Calculate likehood of each feature value given each outcome class - P(feature|class)
        '''
        for feature in range(len(self.features)):  # Iterate over each feature 
            for outcome in np.unique(self.y_train):  # iterate over each unique class 
                outcome_count = sum(self.y_train == outcome)  # Count occurrences of particular class(outcome) over whole dataset
                outcome_indices = np.where(self.y_train == outcome)[0]  # Get index of samples with current outcome

                # Obtain ( unique value , count of each unique value) for feature(feature) that corresponding to currect class(outcome) 
                #feat_likelihood = self.x_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].value_counts().to_dict()
                feat_vals_outcome = self.x_train.iloc[outcome_indices.min():outcome_indices.max(), feature]

                # Count occurrences of each unique feature value
                feat_val_counts = dict(zip(*np.unique(feat_vals_outcome, return_counts=True)))

                for feat_val, count in feat_val_counts.items():  # Iterate over each unique value on feature (feature) correspond to class (outcome)
                    feat_val = feat_val if isinstance(feat_val, str) else str(feat_val)
                    self.likelihoods[feature][feat_val + '_' + outcome] = count/outcome_count  # calculate it by ( number of occurances / count of outcome class )

    def _calc_predictor_prior(self):
        '''
            calculates the prior probabilities of each value of each feature - P(x) 
        '''
        for feature in self.features:  
            feat_vals = self.X_train[feature].value_counts().to_dict()  # count uniqe each value in feature (feature) 

            for feat_val, count in feat_vals.items():   # Iterate over each unique value
                self.pred_priors[feature][feat_val] = count/self.train_size  # calculate prior for each value.

    
    # Calculates Posterior probability P(class |features)
    def predict(self):
        '''
            Calculates the postirior propability 
        '''
        


def test():
    data = pd.read_csv('breast-cancer-training.csv')
    x_train = data.drop(data.columns[1], axis=1)
    y_train = data[data.columns[1]]
    
    clf = NBclassifier(y_train, x_train.iloc[:, 1:])
    clf.fit()

if __name__ == '__main__':
  test()