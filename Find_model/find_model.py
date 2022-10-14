from shutil import ExecError
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class Find_Model:
    def __init__(self,file_obj,log_obj):
        self.file_obj = file_obj
        self.log_obj = log_obj
        self.random_forest_regressor = RandomForestRegressor()
        self.linear_reg = LinearRegression()

    def get_best_params_for_random_regressor(self,train_X,train_Y):
        self.log_obj.log(self.file_obj,'Entered get_best_params_for_random_regressor method of Find_Model class')
        try:
            self.param_grid_rfr = {
                                "n_estimators":[10,20,30],
                                "max_features":["auto","sqrt","log2",None],
                                "min_samples_split":[2,4,8],
                                "bootstrap":[True,False]
                                }
            self.grid = GridSearchCV(self.random_forest_regressor,self.param_grid_rfr,verbose=3,cv=5,n_jobs=-1)
            self.grid.fit(train_X,train_Y)

            # Extract best params
            self.n_estimators = self.grid.best_params_["n_estimators"]
            self.max_features = self.grid.best_params_["max_features"]
            self.min_samples_split = self.grid.best_params_["min_samples_split"]
            self.bootstrap = self.grid.best_params_["bootstrap"]

            # Creating new model with best params
            self.rfregrssor = RandomForestRegressor(n_estimators=self.n_estimators,
                                                    max_features=self.max_features,
                                                    min_samples_split=self.min_samples_split,
                                                    bootstrap=self.bootstrap)
        
            # Training new model
            self.rfregrssor.fit(train_X,train_Y)
            self.log_obj.log(self.file_obj,'RandomForestRegressor best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_regressor method of Find_Model class')
            return self.rfregrssor
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured in get_best_params_for_random_regressor method of Find_model class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'RandomForestRegressor parameter tuning failed. Exited the get_best_params_for_random_regressor method of Find_Model class')
            raise Exception()

    def get_best_params_linear_regressor(self,train_X,train_Y):
        self.log_obj.log(self.file_obj,'Entered the get_best_params_linear_regressor of Find_Model class')
        try:
            self.param_grid_linear = {
                                       "fit_intercept":[True,False],
                                       "normalize":[True,False],
                                       "copy_X":[True,False] 
                                        }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.linear_reg,self.param_grid_linear,verbose=3,cv=5,n_jobs=-1)
            # finding the best parameters
            self.grid.fit(train_X,train_Y)

            # extracting the best parameters
            self.fit_intercept = self.grid.best_params_["fit_intercept"]
            self.normalize = self.grid.best_params_["normalize"]
            self.copy_X = self.grid.best_params_["copy_X"]

            # creating a new model with best parameters
            self.linreg = LinearRegression(fit_intercept = self.fit_intercept,
                                           normalize = self.normalize,
                                           copy_X = self.copy_X)
            # training the new model
            self.linreg.fit(train_X,train_Y)
            self.log_obj.log(self.file_obj,'LinearRegression best params: '+str(self.grid.best_params_)+' .Exited the get_best_params_linear_regressor method of Find_Model class')
            return self.linreg
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured at get_best_params_linear_regressor in Find_Model class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'LinearRegression parameter tuning failed. Exited the get_best_params_linear_regressor method of Find_Model class.')
            raise Exception()

    def get_best_model(self,train_X,train_Y,test_X,test_Y):
        self.log_obj.log(self.file_obj,'Entered the get_best_model method of the Find_Modle class')
        # create best model for RandomForestRegressor
        try:
            self.random_regressor = self.get_best_params_for_random_regressor(train_X,train_Y)
            self.prediction_random_forest = self.random_regressor.predict(test_X)  # Prediction using RandomForestRegressor
            self.prediction_random_forest_error = r2_score(test_Y,self.prediction_random_forest)

        # create best model for Linear Regression
            self.lin_reg = self.get_best_params_linear_regressor(train_X,train_Y)
            self.prediction_linreg = self.lin_reg.predict(test_X)  # Prediction using Linear Regression model
            self.linreg_error = r2_score(test_Y,self.prediction_linreg) 

        # comparing the two models
            if (self.linreg_error<self.prediction_random_forest_error):
                return 'RandomForestRegressor',self.random_regressor
            else:
                return 'LinearRegression' , self.lin_reg
        except Exception as e:
            self.log_obj.log(self.file_obj,'Exception occured at get_best_model method of Find_Model class. Exception message: '+str(e))
            self.log_obj.log(self.file_obj,'Failed to Select best model. Exited the get_best_model of class Find_Model.') 
            raise Exception()
