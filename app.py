from flask import Flask, request , render_template , url_for, send_from_directory
from flask import Response
import os
from flask_cors import CORS , cross_origin
from flask import Flask , flash , request , redirect , url_for
from werkzeug.utils import secure_filename
from modeltraining import TrainModel
from modelpredict import Prediction

UPLOAD_FOLDER = 'prediction_data'
TRAIN_FOLDER = 'training_data'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRAIN_FOLDER'] = TRAIN_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
    

@app.route('/',methods = ['GET','POST'])
def upload_file():
    new_name = 'input.csv'
    if request.method == 'POST':
        # check if the post request has the file
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename
        if file.filename =='':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename  = secure_filename(file.filename)
            path = 'prediction_data'

            if len(os.listdir(path)) == 0:

                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                for files in os.listdir(path):
                    print(files)
                    os.rename(path+'/'+files, path+'/'+new_name)
            
            else:
                for files in os.listdir(path):
                    os.remove(path+'/'+files)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
                for files in os.listdir(path):
                    print(files)
                    os.rename(path+'/'+files,path+'/'+new_name)
            
            path = 'prediction_data/input.csv'
            pred = Prediction(path)  # object initialization
            path = pred.predict()

            return redirect(url_for('download'))
    
    return render_template('main.html')

@app.route("/download")
def download():
    return render_template('result_predict.html',files=os.listdir('prediction_output'))

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory('prediction_output',filename)

@app.route("/train",methods=['POST','GET'])
@cross_origin()
def train_file():
    new_name = 'InputFile.csv'
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select afile, the browser submits an
        # empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = 'training_data'

            if len(os.listdir(path)) == 0:
                file.save(os.path.join(app.config['TRAIN_FOLDER'], filename))
                for files in os.listdir(path):
                    print(files)
                    os.rename(path+'/'+files,path+'/'+new_name)

            else:
                for files in os.listdir(path):
                    os.remove(path+'/'+files)
                file.save(os.path.join(app.config['TRAIN_FOLDER'], filename))
                for files in os.listdir(path):
                    print(files)
                    os.rename(path+ '/'+files,path +'/'+new_name)
            train = TrainModel() # object initialization
            path = train.model_training()
            return render_template('result_train.html')

    return render_template('train.html')

if __name__ == "__main__":
    app.run()

print('The End')
