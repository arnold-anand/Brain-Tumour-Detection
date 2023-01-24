from flask import Flask, render_template, url_for, request
from keras.models import load_model
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
print("imports-check")
app = Flask(__name__)
print("Flask initialisation - check")
UPLOAD_FOLDER = '/content/drive/MyDrive/mini-project/static/images/'
app.secret_key = "braintumour"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 *1024 *1024
model = load_model('/content/drive/MyDrive/mini-project/model.h5')
print("app config and model load check")
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
def predict(imge, model):
    img = cv2.imread(imge)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(150,150))
    img = img.reshape(1,150,150,3)
    p = model.predict(img)
    p = np.argmax(p,axis=1)[0]
    return p

@app.route("/")
def index():
    return render_template('index.html', mimetypes='images/svg')

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    
    target_img = os.path.join(os.getcwd() , 'static/images')

    if request.method == 'POST':
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                p = predict(img_path, model)
                if p==0:
                    x='Glioma Tumor'
                elif p==1:
                    x = 'No Tumor'
                elif p==2:
                    x ='Meningioma Tumor'
                else:
                    x = 'Pituitary Tumor'

    return render_template('success.html', mimetypes='images/svg', variable = x)

app.run()




