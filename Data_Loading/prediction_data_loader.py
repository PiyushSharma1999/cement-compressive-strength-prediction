import pandas as pd

class Pred_Data_Getter:
    def __init__(self,file_obj,log_obj):
        self.prediction_file = 'prediction_data/input.csv'
        self.file_obj = file_obj
        self.log_obj = log_obj
    
    def get_data(self):
        self.log_obj.log(self.file_obj,'Entered the get data method of Pred_Data_Getter class')
        try:
            self.data = pd.read_csv(self.prediction_file)
            self.log_obj.log(self.file_obj,'Data loaded successfully. Exited the get_data method of the Pred_Data_Getter class')
            return self.data
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured in get_data method of Pred_Data_Getter class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Data loading Unsuccessful. Exited the get data method of Pred_Data_Getter class')
            raise Exception()