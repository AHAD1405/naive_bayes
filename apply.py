import numpy as np
import pandas as pd
import array
import classifier 

def sample_output(predictions, class_scores):
   
   with open("sampleoutput.txt", "w") as file:
      for predicted, class_probs, index in zip(predictions, class_scores, range(len(predictions))):
        class_score_ = class_probs[0] if predicted == 'no-recurrence-events' else class_probs[1]
        contect = "Instance ("+ str(index) +"):\n Predicted class is ("+ str(predicted) +"), its score is ("+ str(class_score_) +") \n\n" 
        file.writelines(contect)
   

def main():
    # Load train data
    data = pd.read_csv('breast-cancer-training.csv')
    x_train = data.drop(data.columns[1], axis=1)
    y_train = data[data.columns[1]]
    
    clf = classifier.NBclassifier(y_train, x_train.iloc[:, 1:])
    clf.fit()

    # Load test data 
    test_data = pd.read_csv('breast-cancer-test.csv')
    x_test = test_data.drop(test_data.columns[1], axis=1)
    y_test = test_data[data.columns[1]]

    # Predict test data
    predictions, class_score = clf.predict(x_test.iloc[:, 1:])
    # HINT: for(class_score): score of (no-recurrence-events) = [idx][0] , score of (recurrence-events) = [idx][1]

    # Export score and class predicted into file(sampleoutput.txt)
    sample_output(predictions, class_score)

    # Print out: conditional prob for each feature:(Condirtion Prob, Possible values) & Prob of each class label

    # print score of (no-recurrence-events, recurrence-events) and prediction class for each test instance.

if __name__ == '__main__':
  main()