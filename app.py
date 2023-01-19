'''
FLASK_APP=app.py FLASK_DEBUG=1 TEMPLATES_AUTO_RELOAD=1 flask run
'''

import os, json
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory, jsonify, render_template
from ml_module import ML_Module

UPLOAD_FOLDER = './temp'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}
DELETE_AFTER_PREDICTION=True
current_status= "running"
tokens={"id":101,"filename":""}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024*1024  # 16MB


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            token_dir=os.path.join(app.config['UPLOAD_FOLDER'], str(tokens['id'])) 
            os.mkdir(token_dir) if not os.path.isdir(token_dir) else None
            tokens['filename']=os.path.join(token_dir, filename)
            file.save(tokens['filename'])
            print ('File saved at ',tokens['filename'])
            return redirect(url_for('predict', name=tokens['filename']))

    return render_template('main.html')


@app.route('/uploads/<name>')
def download_file(name):
    token_dir=tokenize['id']
    os.mkdir(token_dir) if not os.path.isdir(token_dir) else None
    token_filename=os.path.join(token_dir,name)
    tokens['filename']=token_filename
    return send_from_directory(app.config["UPLOAD_FOLDER"], token_filename)

@app.route ('/tokenize/<token>')
def tokenize(token):
    tokens['id']=token
    return redirect(url_for('status'))

@app.route('/status')
def status():
    response_json= jsonify ({"status":current_status, 'token':tokens['id']})
    return response_json

@app.route("/test")
def test():
    test_payload={"payload": tokens['filename']}
    return jsonify(test_payload)

@app.route('/predict')
def predict():
    path_file=tokens['filename']
    # todo: call function for prediction
    ml = ML_Module()
    ml.load_model()
    payload=ml.predict(ifile=path_file)
    response_json=json.dumps( payload)
    os.unlink(path_file) if DELETE_AFTER_PREDICTION else None #delete token_dir on token expiry
    return response_json

