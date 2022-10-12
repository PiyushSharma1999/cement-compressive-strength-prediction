import pickle
import os
import shutil

class File_Operations:
    def __init__(self, file_obj, log_obj):
        self.file_obj = file_obj
        self.log_obj = log_obj
        self.model_directory = 'models/'
    
    def model_saving(self,model,filename):
        self.log_obj.log(self.file_obj,'Entered model_saving methods of File_Operation Class')
        try:
            path = os.path.join(self.model_directory, filename)
            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)
            
            with open(path+'/'+filename+'.sav','wb') as f:
                pickle.dump(model,f)
            self.log_obj.log(self.file_obj,'Model File'+filename+' saved. Exited the model_Saving method of class File_Operation')
            return 'Success'
            print('model_saved')

        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured save_model method of File_Operations class. Exception message '+str(e))
            self.log_obj.log(self.file_obj,'Model File'+filename+' could not be saved. Exited the model_saving method of class File_Operations.')
            
            raise Exception()

    def load_model(self,filename):
        self.log_obj.log(self.file_obj,'Entered the load_model method of class File_Operations')
        try:
            with open(self.model_directory+filename+'/'+filename+'.sav','rb') as f:
                self.log_obj.log(self.file_obj,'Model file '+filename+' loaded. Exited the load_model method of File_Operations class')
                return pickle.load(f)
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured in load_model method of File_Operations class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Loading model Unsuccessful. Exited the load_model method of File_Operations class')
            raise Exception()
    
    def find_correct_model(self,cluster_no):
        self.log_obj.log(self.file_obj,'Entered the find_correct_model of class File_Operations')
        try:
            self.cluster_no = cluster_no
            self.folder_name = self.model_directory
            self.list_of_model_files = []
            self.list_of_model_files = os.listdir(self.folder_name)
            for self.file in self.list_of_model_files:
                try:
                    if (self.file.index(str(self.cluster_no))!=-1):
                        self.model_name = self.file
                except:
                    continue
            self.model_name = self.model_name.split('.')[0]
            self.log_obj.log(self.file_obj,'Exited the find_correct_model of File_Operations class with success')
            return self.model_name
        
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured in finde_correct_model of File_Operations class. Exception message: '+str(e))
            self.file_obj.log(self.file_obj,'Finding model unsuccessful. Exited the find_correct_model_method of class File_Operation')
            raise Exception()
            