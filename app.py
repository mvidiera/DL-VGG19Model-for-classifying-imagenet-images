#any browse or upload will be stored in upload folders


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'vgg19.h5'


# Load your trained model
model = load_model(MODEL_PATH)   #load vgg model
model.make_predict_function()          # Necessary to make predicition 

#preprocessing function

#fucntion to create image into array
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224)) #input image of 244, 244 as it is vggnet

    # Preprocessing the image
    x = image.img_to_array(img) #convert this image into array 
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0) #expanding dimensions 

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

#App Routing means mapping the URLs to a specific function that will handle the logic for that URL.
@app.route('/', methods=['GET']) #/ is root folder. get method:refers to a HTTP) method that is applied while requesting information from a particular source
def index():
    # Main page
    return render_template('index.html') # shows first starting page : generates op 
#index. html have browse button to upload image
# if I give /, this func executes where generates/shows html


#to do prediction
@app.route('/predict', methods=['GET', 'POST'])#if I type /predict this func will be executes. when I click predict in html this func executes
def upload(): # to upload image
    if request.method == 'POST':  #if user is uploading picture/post to predict. 
        # Get the file from post request
        f = request.files['file'] #check templates/index folder
#why file: in my index.html file, ip type is file, whose button is imagae upload 

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__) # store in uploads folder: base path __file__: this gives root path 
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)) #now join basepath and save in uploads 
        f.save(file_path) #save filepath 


        # Make prediction
        preds = model_predict(file_path, model) #here we are predicting where my arg are file_path and model 
#file path has the image that I have uploaded. Model has the model which I have created vggnet
        # Process your result for human.imported lib decode_predictions in keras: 
  # model predicts givesme class label, I will decode predictions: whatever class which Im getting, map those class index to my op 
  #classname(cat/dog etc) with the name of class. because vgg gives 1000 vector of classes as op
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None #comment out this line and check. 
#return none: if it is not post method, then it returns none, no error will be displayed

#html page: get the image with /. 
# save in index page 

# html page click on predict- that file will be saved in uploads
#model_predict() will be called andfile path and model is passed as arg
#once it is predicted using this func(), it returns class, 1000 vector. It is decoded using decode_predictions() where
#preds is 1000  last layer features and top ishow many top guesses to return. as softmax is applied, top 3 pred are returned if I give top=3
# in this case I have given top=1, that is only 1 class will be returned
#pred class will be in inagenetform and we need to convert into string. 

if __name__ == '__main__':
    app.run(debug=True)


#flask app skeleton 


#import numpy as np
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image

# Flask utils
#from flask import Flask, redirect, url_for, request, render_template
# Define a flask app
#app = Flask(__name__)

#if __name__ == '__main__':
 #   app.run(debug=True)
