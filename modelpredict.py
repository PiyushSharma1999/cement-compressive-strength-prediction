import pandas as pd
from File_Operations import file_methods
from Data_Preprocessing import preprocessing, clustering
from Data_Loading import prediction_data_loader
from App_Logging import logger
import os

class Prediction:
    def __init__(self,path):
        
        self.file_obj = open("Prediction_Logs/prediction_log.txt",'a+')
        self.write_log = logger.Logging_App()
        print('Prediction onset')
    
    def predict(self):
        try:
            self.write_log.log(self.file_obj,'Prediction started')
            getting_data = prediction_data_loader.Pred_Data_Getter(self.file_obj,self.write_log)
            data = getting_data.get_data()

            preprocessor = preprocessing.Preprocessing(self.file_obj,self.write_log)

            is_null_present = preprocessor.is_null_present(data)
            if(is_null_present):
                data = preprocessor.fill_null_values(data)
            
            data = preprocessor.logTransformation(data)

            data_scaled = pd.DataFrame(preprocessor.standardScalar(data),columns=data.columns)


            load_file = file_methods.File_Operations(self.file_obj,self.write_log)
            kmeans = load_file.load_model('KMeans_clustering')

            clusters = kmeans.predict(data_scaled)  # drop the first column for cluster prediction
            data_scaled['clusters'] = clusters
            clusters = data_scaled['clusters'].unique()
            result = []

            for i in clusters:
                cluster_data = data_scaled[data_scaled['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = load_file.find_correct_model(i)
                model = load_file.load_model(model_name)
                for val in (model.predict(cluster_data.values)):
                    result.append(val)
            result = pd.DataFrame(result,columns=['Prediction'])
            path = "prediction_output"
            if len(os.listdir(path))==0:
                result.to_csv(path+'/predictions.csv',header=True,mode='a+')
                self.write_log.log(self.file_obj,'End of Prediction')

            else:
                for files in os.listdir(path):
                    os.remove(path+'/'+files)
                    result.to_csv(path+'/predictions.csv',header=True,mode='a+')
                    self.write_log.log(self.file_obj,'End of Prediction')
                    
        except Exception as e:
            self.write_log.log(self.file_obj,'Error occured while runinng prediction. Error message: '+str(e))
            raise Exception()
        
        return path
                