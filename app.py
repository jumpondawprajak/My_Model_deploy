
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image



from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
#model ประเมินจากความเสี่ยง choise
import pickle
file = open('model_pickle','rb')
clf = pickle.load(file)

#model upload รูป ct scan
model =tf.keras.models.load_model('model.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds



@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        mydict=request.form
        fever = int(mydict['fever'])
        age = int(mydict['age'])
        bodypain = int(mydict['bodypain'])
        runnynose = int(mydict['cough'])
        diffbreathing = int(mydict['diffbreathing'])
        inputfeatures = [fever,age,bodypain,runnynose,diffbreathing]
        infprob = clf.predict_proba([inputfeatures])[0][1]
        print(infprob)
        return render_template('show.html',inf = round(infprob*100.0))
    return render_template('index.html')

@app.route('/blog', methods=['GET', 'POST'])
def blog():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return render_template('blog.html')

    # show the form, it wasn't submitted
    return render_template('blog.html')


@app.route('/ct', methods=['GET'])
def ct():
    # Main page
    return render_template('ct.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['เป็น Covid-19','ไม่เป็น Covid-19']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        return result
    return None
    

if __name__ == "__main__":
    app.run(debug=True)
