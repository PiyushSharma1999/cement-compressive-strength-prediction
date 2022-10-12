from email.errors import StartBoundaryNotFoundDefect
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessing:
    def __init__(self,file_obj,log_obj):
        self.file_obj = file_obj
        self.log_obj = log_obj

    def column_removal(self,data,columns):
        self.log_obj.log(self.file_obj,'Entered column_removal method of Preprocessing class')
        self.data = data
        self.columns=columns
        try:
            self.useful_data = self.data.drop(labels=self.columns,axis=1)
            self.log_obj.log(self.file_obj,'Successfully removed columns. Exited the column_removal method of Preprocessing class')
            return self.useful_data
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception Occured at column_removal of Preprocessing class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Column removal unsuccessful. Exited the column_removal method of the Preprocessing class')
            raise Exception()
    
    def separate_label_feature(self,data,target_name):
        self.log_obj.log(self.file_obj,'Entered the separate_label_feature method of Preprocessing class')
        self.data=data
        try:
            self.X = data.drop(labels=target_name,axis=1)
            self.Y = data[target_name]
            self.log_obj.log(self.file_obj,'Feature separation successful. Exited the separate_label_feature method of Preprocessing class')
            return self.X,self.Y
        
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured at separate_label_feature method of Preprocessing class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Separating label and feature Unsuccessful. Exited the separate_label_feature method of Preprocessing class')
            raise Exception()
    
    def dropUnnecessaryColumns(self,data,columnList):
        data =data.drop(columnList)
        return data
    
    def replaceInvalidValueswithNull(self,data):
        for column in data.columns:
            count = data[column][data[column]=='?'].count()
            if count != 0:
                data[column] = data[column].replace('?',np.nan)
        return data
    
    def is_null_present(self,data):
        self.log_obj.log(self.file_obj,'Entered the is_null_present method of the Preprocessing class')
        self.null_present = False
        try:
            self.null_count = data.isna().sum()
            for i in self.null_count:
                if i>0:
                    self.null_present =True
                    break
                if self.null_present:
                    df_with_null = pd.DataFrame()
                    df_with_null['columns'] = data.columns
                    df_with_null['null_value_count'] = np.asarray(data.isna().sum())
                    df_with_null.to_csv('Preprocessing_Data/null_values.csv')
            self.log_obj.log(self.file_obj,'Successfully found missing values. Created separate dataframe for null values count. Exited null_values_presence method of class Preprocessing')
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception Occured at is_null_present method of the Preprocessing class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Finding missing values unsuccessful. Exited the is_null_present method of the Preprocessing class.')
            raise Exception()

    def fill_null_values(self,data):
        self.log_obj.log(self.file_obj,'Entered the fill_null_values method of the Preprocessing class')
        self.data = data
        self.cols_with_missing_values=[]
        try:
            imputer = KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(data = self.new_array, columns=self.data.columns)
            self.log_obj.log(self.file_obj,'Successfully imputed missing values. Exited fill_null_values method of class Preprocessing')
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured at fill_null_present method of Preprocessing class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Filling null values failed. Exited fill_null_values method of Preprocessing class')
            raise Exception()
    
    def get_columns_with_zero_std_deviation(self,data):
        self.log_obj.log(self.file_obj,'Entered get_columns_with_zero_std_deviation method of Preprocessing class')
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []
        try:
            for x in self.columns:
                if self.data_n[x]['std']==0:
                    self.col_to_drop.append(x)
            self.log_obj.log(self.file_obj,'Successfully identified columns with zero standard deviation. Exited the get_columns_with_zero_std_deviation of Preprocessing class.')
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured at get_columns_with_zero_std_deviation of Preprocessing class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Getting column with zero std deviation failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessing class')
            raise Exception()
    def logTransformation(self,X):
        for column in X.columns:
            X[column] += 1
            X[column] = np.log(X[column])
        return X

    def standardScalar(self,X):
        scalar = StandardScaler()
        x_scaled = scalar.fit_transform(X)
        return x_scaled
    
    def imbal_data_handling(self,X,y):
        sample = SMOTE()
        X,y = sample.fit_resample(X,y)
        return X,y
        