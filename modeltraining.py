from pyexpat import model
from sklearn.model_selection import train_test_split
from Data_Loading import training_data_loader
from Data_Preprocessing import preprocessing,clustering
from Find_model import find_model
from File_Operations import file_methods
from App_Logging import logger

class TrainModel:
    def __init__(self):
        self.write_log = logger.Logging_App()
        self.file_obj = open('Training_Logs/Training_model_logs.txt','a+')
    
    def model_training(self):
        self.write_log.log(self.file_obj,'Training started')
        try:
            data_getter = training_data_loader.Data_Getter(self.file_obj,self.write_log)
            data = data_getter.get_data()

            """Data Preprocessing"""
            preprocessor = preprocessing.Preprocessing(self.file_obj,self.write_log)
            is_null_present = preprocessor.is_null_present(data) 

            if(is_null_present):
                data = preprocessor.fill_null_values(data)  # impute missing values

            # create X & Y
            X,Y = preprocessor.separate_label_feature(data,target_name='concrete_compressive_strength')

            X = preprocessor.logTransformation(X)

            """Apply clustering"""              
            kmeans = clustering.Clustering(self.file_obj,self.write_log) # object initialization
            number_of_clusters = kmeans.elbow_plot(X)  # using elbow plot to find optimum number of clusters

            # Divide data into clusters
            X = kmeans.create_cluster(X,number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignment
            X['labels'] = Y

            # getting the unique cluster from our dataset
            list_of_clusters = X['cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X['cluster']==i]  #  filter the data from one cluster

                # Prepare the featue and label columns
                cluster_features = cluster_data.drop(['labels','cluster'],axis=1)
                cluster_labels = cluster_data['labels']

                # splitting the data into training and test set for each cluster one by one
                x_train , x_test , y_train , y_test = train_test_split(cluster_features,cluster_labels,test_size=1/3,random_state=36)

                x_train_scaled = preprocessor.standardScalar(x_train)
                x_test_scaled = preprocessor.standardScalar(x_test)

                model_finder = find_model.Find_Model(self.file_obj,self.write_log)  # object initialization

                # getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(x_train_scaled,y_train,x_test_scaled,y_test)

                # saving the best model to the directory
                file_op = file_methods.File_Operations(self.file_obj,self.write_log)
                save_model = file_op.model_saving(best_model,best_model_name+str(i))

            self.write_log.log(self.file_obj,'Successful End of Model Training')
            self.file_obj.close()

        except Exception as e:
            self.write_log.log(self.file_obj,'Model training Unsuccessful')
            self.file_obj.close()
            raise Exception()