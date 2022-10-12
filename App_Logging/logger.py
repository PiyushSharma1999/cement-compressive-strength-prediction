from datetime import datetime

class Logging_App:
    def __init__(self):
        pass

    def log(self,file_obj,message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.time = self.now.strftime("%H %M %S" )
        file_obj.write(str(self.date)+'/'+str(self.time)+'\t\t'+message+'\n')
        print('logging is working')