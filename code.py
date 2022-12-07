# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:05:08 2022

@author: amrane
"""

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.model_selection import cross_val_score

import optuna
import pickle

class DataSet:
    """
    Read data set form .csv file 
    Args:
        filename: str
    Retrun:
        the data set to train our model
    """
    
    def __init__(self, fname, n_split, features, labels):
        """
        

        Parameters
        ----------
        fname : String
            The dataset file name.
        features : string
            Inputs of the model (Features).
        labels : String
            Outputs of the model (Labels.

        Returns
        -------
        None.

        """
        self.filename = fname
        self.features = features[0]
        self.labels = labels[0]
        self.n_split = n_split
        
    def get_data(self, fname):
        """
        Read data from .csv file, this file contain dataset. 

        Parameters
        ----------
        fname : str
            The name of the file name to read. 

        Returns
        -------
        None.

        """
        self.df = pd.read_csv(fname)
        
    
    def get_features_and_labels(self, df):   
        """
        Get the features and the labels from data frame. 

        Parameters
        ----------
        df : list
            the list containing the data frame

        Returns
        -------
        None.

        """
        self.df = df
        #Get features
        self.x = self.df[self.features].values.tolist() # players names
        #Get labels
        self.y = self.df[self.labels].values # labels
        #Get data frame values 
        self.df_vals = self.df.drop([self.labels, self.features],axis=1).values
        
        for x in np.argwhere(np.isnan(self.df_vals)):
            self.df_vals[x]=0.0
            
    
    def data_preprocessing(self):
        """
        Transform features by scaling each feature to a given range. 

        Returns
        -------
        None.

        """
        self.X = MinMaxScaler().fit_transform(self.df_vals)
        
        
    def set_classifier_name(self, classifier_name):
        """
        Choose the classifier to use for data classification

        Parameters
        ----------
        classifier_name : STRING
            The name of the classifier.

        Raises
        ------
        Exception
            Classifier name does not exist.
        Returns
        -------
        None.

        """
        #SUPER VECTOR CLASSIFIER
        self.classifier_name = classifier_name
        if classifier_name == 'SVC':
            self.classifier = SVC()
        
        #RANDOM FOREST CLASSIFIER
        elif classifier_name == 'Random forest':
            self.classifier = RandomForestClassifier()
            
        #LOGISTIC REGRESSION CLASSIFIER     
        elif classifier_name == 'Logistic Regression':
            self.classifier = LogisticRegression(random_state=0)
            
        else:
            raise Exception ('The loss function is not implemented in this case')    
        

        
        
    def classification (self, verbose = False):
        """
        Split data used for classification to train and test.
        Calculate the metrics used in the chosen classification model (recall, precision, f1 score).

        Parameters
        ----------
        n_split : INTEGER
            Number of split data.
        verbose : BOOL, optional
            Print only last call of metrics. The default is False.

        Returns
        -------
        f1score : FLOAT
            Measure the accuracy of the test. It is calculated from the precision and recall of the test.

        """
        #Initializations
        self.y_pred =[]
        
        recall = 0
        precision = 0
        f1score = 0
        confusion_mat = np.zeros((2,2))
        
        #Set the number of data splits to train and test 
        self.kf= KFold(n_splits= self.n_split, random_state=50, shuffle=True)
        for train_ids, test_ids in self.kf.split(self.X):
            
            self.x_train = self.X[train_ids]
            self.y_train = self.y[train_ids]
            
            self.x_test = self.X[test_ids]
            self.y_test = self.y[test_ids]
            
        #fit the model using the training dataset and make predictions on the test dataset
            self.classifier.fit(self.x_train, self.y_train)
            self.y_pred = self.classifier.predict(self.x_test)
        
        #Calculation of metrics for each split(recall, precision, f1 score)
            confusion_mat += confusion_matrix(self.y_test, self.y_pred)
            recall += recall_score(self.y_test, self.y_pred)
            precision += precision_score(self.y_test, self.y_pred)
            f1score += f1_score(self.y_test, self.y_pred, average= 'binary')
         
            
        recall/= self.n_split
        precision/=  self.n_split
        f1score/= self.n_split
        self.confusion_mat = confusion_mat
        
        if verbose:
            print('Confusion matrix = \n',self.confusion_mat)
            print('Precision = ',precision)
            print('Recall = ', recall)
            print("F1 Score = ", f1score)
          
        return f1score
    

    def get_best_params(self, trial):
        """
        Classifier hyperparameter optimization using optuna.
        Optimization is based on the results of the f1-score metric.

        Parameters
        ----------
        trial : Integer
            Number of times hyperparameters are suggested for optimization.

        Returns
        -------
        f1score : Float
            Measure the accuracy of the test. It is calculated from the precision and recall of the test.

        """
        
        if self.classifier_name == 'SVC':
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            self.classifier = SVC(kernel=kernel, gamma="scale", random_state=0)
            f1score = self.classification() 
            return f1score
            
        
        elif self.classifier_name == 'Random forest':
            #Range of hyperparameters
            params_clf = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
            }
            #Do the classification with the hyperparameter chosen
            self.classifier = RandomForestClassifier(**params_clf, random_state=0)
            f1score = self.classification() 
            return f1score
        
        elif self.classifier_name == 'Logistic Regression':
            params_clf = {
            'tol' : trial.suggest_uniform('tol' , 1e-4 , 1e-3),
            'C' : trial.suggest_loguniform("C", 1e-2, 1),
            'fit_intercept' : trial.suggest_categorical('fit_intercept' , [True, False]),
            #'random_state' : trial.suggest_categorical('random_state' , [0, 42, 202, 555]),
            'solver' : trial.suggest_categorical('solver' , ['lbfgs','liblinear']),
            "n_jobs" : -1
            }
            
            self.classifier = LogisticRegression(**params_clf, random_state=0)
            f1score = self.classification() 
            return f1score
        
        
    def do_optimisation(self, n_trials):
        """
        Optimization of hyperparameters of classifier, based on maximisation of the F1-score
        Do the classification with the best hyperparameters found

        Parameters
        ----------
        n_trials : Integer
            Number of times we want Optuna to carry out a study.

        Returns
        -------
        None.

        """
        #Dont print each optimisation done 
        optuna.logging.set_verbosity(optuna.logging.WARNING)
       
        #Creation of the study by maximizing the F1-score
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.get_best_params, n_trials= n_trials)
        
        self.model_best_params = self.study.best_params
        
        print("The best parameters for "+ str(self.classifier_name) + " are : \n" ,self.study.best_params, "\n")
        print("Results of classification: ")
        
        
        if self.classifier_name == 'Random forest':
            self.classifier = RandomForestClassifier(**self.model_best_params, random_state=0)
            self.classification(verbose = True)
            
        elif self.classifier_name == 'Logistic Regression':
            self.classifier = LogisticRegression(**self.model_best_params, random_state=0)
            self.classification(verbose = True)
            
        elif self.classifier_name == 'SVC':
            self.classifier = SVC(**self.model_best_params, random_state=0)
            self.classification(verbose = True)
     
          
    def plot_optimisation(self):
        """
        Plot the value of the F1-score for each time step durong the optimisation

        Returns
        -------
        None.

        """
        plt.figure(figsize= [20, 10])
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.tight_layout()
        
        
    def get_data_shape(self):
        """
        Get features and labels shape of train and test

        Returns
        -------
        None.

        """
    
        print("x_train = ", np.shape(self.x_train), 'y_train = ', np.shape(self.y_train))
        print("x_test  = ", np.shape(self.x_test) , 'y_test  = ', np.shape( self.y_test))        

       
    def features_reduction(self):
        """
        Reduce the input (features) for the classification 

        Returns
        -------
        List 
            New data frame containing less features

        """
        #Train model selection
        selector = SelectFromModel(estimator= RandomForestClassifier()).fit(self.x_train, self.y_train)
        
        #Get features selected
        labels_ = selector.get_support()
        
        #Drop indexes not selected
        drop_index = []
        for i in range(len(labels_)):
            if labels_[i] == False:
                drop_index.append(i)
        
        self.df.drop(self.df.columns [drop_index], axis = 1, inplace= True)  
        
        return self.df
    
    
    def save_model(self):
        """
        Save the classification model

        Returns
        -------
        None.

        """
        # save the model to disk
        filename = self.classifier_name +'.sav'
        pickle.dump(self.classifier, open(filename, 'wb'))
        
        
        # load the model from disk
        #loaded_model = pickle.load(open(filename, 'rb'))
        #result = loaded_model.score(self.x_test, self.y_test)
        
        
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix 
        Plot the pourcentage of the good and bad classifications

        Returns
        -------
        None.

        """
        
        group_names = ['True Neg','False Pos','False Neg','True Pos']

        group_counts = ["{0:0.0f}".format(value) for value in
                self.confusion_mat.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in
                     self.confusion_mat.flatten()/np.sum(self.confusion_mat)]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

        labels = np.asarray(labels).reshape(2,2)
        plt.figure()
        ax = sns.heatmap(self.confusion_mat, annot=labels, fmt='', cmap='Blues')

        ax.set_title('Seaborn Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])

        ## Display the visualization of the Confusion Matrix.
        plt.show()


if __name__ == "__main__":
    '''
    c = DataSet(fname = ".\\nba_logreg.csv", n_split = 3, features = ['Name'], labels = ['TARGET_5Yrs'])
    c.get_data(".\\nba_logreg.csv")
    c.get_features_and_labels(c.df)
    c.data_preprocessing()
    c.set_classifier_name('Random forest')
    c.do_optimisation(n_trials = 1)
    c.plot_optimisation()
    #c.features_reduction()
    #c.plot_predections()
    #c.cross_val()
    c.get_data_shape()
    c.plot_confusion_matrix()
    
    
    c.features_reduction()
    
    c.get_features_and_labels(c.df)
    c.data_preprocessing()
    c.set_classifier_name('Random forest')
    c.do_optimisation(n_trials = 1)
    #c.plot_predections()
    c.plot_confusion_matrix()
    #c.cross_val()
    c.get_data_shape()
    '''