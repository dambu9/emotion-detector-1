# Let us import the Libraries required.
from operator import ne
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib

# To use the model saved in the Json format, We are importing "model_from_json"
from tensorflow.keras.models import model_from_json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotionanalysis.db'
app.config['SECRET_KEY'] = 'ec9439cfc6c796ae2029594d'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class ImageAnalysis(db.Model):
    imageid = db.Column(db.Integer(), primary_key=True)
    details = db.Column(db.String(length=1024), nullable=True)
    Angry = db.Column(db.Float())
    Disgust = db.Column(db.Float())
    Fear = db.Column(db.Float())
    Happy = db.Column(db.Float())
    Neutral = db.Column(db.Float())
    Sad = db.Column(db.Float())
    Surprise = db.Column(db.Float())

    def __repr__(self):
        return f'ImageAnalysis {self.id}' 

def mood(result):
    if result=="Happy":
        return 'Since you are happy, lets keep up the good mood with some amazing music!'
    elif result=="Sad":
        return 'It seems that you are having a bad day, lets cheer you up with some amazing music!'
    elif result=="Disgust":
        return 'It seems something has got you feeling disgusted. Lets improve your mood with some great music!'
    elif result=="Neutral":
         return 'It seems like a normal day. Lets turn it into a great one with some amazing music!'
    elif result=="Fear":
        return 'You seem very scared. We are sure that some music will help!'
    elif result=="Angry":
        return 'You seem angry. Listening to some music will surely help you calm down!'
    elif result=="Surprise":
        return 'You seem surprised! Hopefully its some good news. Lets celebrate it with some great music!'



def Emotion_Analysis(img):
    """ It does prediction of Emotions found in the Image provided, does the 
    Graphical visualisation, saves as Images and returns them """

    # Read the Image through OpenCv's imread()
    path = 'static/images/' + str(img)
    image = cv2.imread(path)

    # Convert the Image into Gray Scale
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image size is reduced by 30% at each image scale.
    scaleFactor = 1.3

    # 5 neighbors should be present for each rectangle to be retained.
    minNeighbors = 5

    # Detect the Faces in the given Image and store it in faces.
    faces = facec.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

    # When Classifier could not detect any Face.
    if len(faces) == 0:
        return [img]

    for (x, y, w, h) in faces:

        # Taking the Face part in the Image as Region of Interest.
        roi = gray_frame[y:y+h, x:x+w]

        # Let us resize the Image accordingly to use pretrained model.
        roi = cv2.resize(roi, (48, 48))

        # Let us make the Prediction of Emotion present in the Image
        prediction = test_model.predict_emotion(
            roi[np.newaxis, :, :, np.newaxis])

        # Custom Symbols to print with text of emotion.
        Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!",
                   "Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}
    

        ## based on the prediction recommend music


        # Defining the Parameters for putting Text on Image
        Text = str(prediction) + Symbols[str(prediction)]
        Text_Color = (180, 105, 255)

        Thickness = 2
        Font_Scale = 1
        Font_Type = cv2.FONT_HERSHEY_SIMPLEX

        # Inserting the Text on Image
        cv2.putText(image, Text, (x, y), Font_Type,
                    Font_Scale, Text_Color, Thickness)

        # Finding the Coordinates and Radius of Circle
        xc = int((x + x+w)/2)
        yc = int((y + y+h)/2)
        radius = int(w/2)

        # Drawing the Circle on the Image
        cv2.circle(image, (xc, yc), radius, (0, 255, 0), Thickness)

        # Saving the Predicted Image
        predictedImagePath = "static/images/pred_" + str(img)
        cv2.imwrite(predictedImagePath, image)
        plt.gray()
        plt.imshow(image)
        plt.show()

        

        # List of Emotions
        EMOTIONS = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

        # List of Emotions
        NEW_EMOTIONS = [ "Happy",
                    "Neutral", "Other"]            

        # Finding the Probability of each Emotion
        preds = test_model.return_probabs(roi[np.newaxis, :, :, np.newaxis])

        # Converting the array into list
        data = preds.tolist()[0]

        happy_prop = 0
        neutral_prop = 0
        other = 0
        for i in range(0,len(data)):
            if i == 4:
                happy_prop = data[i]
            elif i == 5:
                neutral_prop = data[i]
            else:
                other += other

        new_cordinates = [happy_prop, neutral_prop, other]

        # Initializing the Figure for Bar Graph
        plt.switch_backend('Agg')
        fig = plt.figure(figsize=(8, 5))

        # Creating the bar plot
        plt.bar(NEW_EMOTIONS, new_cordinates, color='green',
                width=0.4)

        # Labelling the axes and title
        plt.xlabel("Types of Emotions")
        plt.ylabel("Probability")
        plt.title("Facial Emotion Recognition")
        
        emotionsList = []
        for i in range(0,len(data)):
            emotionsList.append(round(100*data[i],2))
        
        print(str(list(zip(EMOTIONS, emotionsList))))
        image_info= list(zip(EMOTIONS, emotionsList))
        image_analysis_to_create = ImageAnalysis(details=str(list(zip(EMOTIONS, emotionsList))), Angry=data[0], Disgust=data[1], Fear=data[2], Happy=data[3], Neutral=data[4], Sad=data[5], Surprise=data[6])
        db.session.add(image_analysis_to_create)
        db.session.commit()

        # Saving the Bar Plot
        path = "static/images/" + "bar_plot" + str(img)
        plt.savefig(path)
       
    # Returns a list containing the names of Original, Predicted, Bar Plot Images
    return ([img, "pred" + img, "bar_plot" + img, prediction])

class FacialExpressionModel(object):

    """ A Class for Predicting the emotions using the pre-trained Model weights"""

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    # Whenever we create an instance of class , these are initialized
    def __init__(self, model_json_file, model_weights_file):

        # Now Let us load model from JSON file which we created during Training
        with open(model_json_file, "r") as json_file:

            # Reading the json file and storing it in loaded_model
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # Now, Let us load weights into the model
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        """ It predicts the Emotion using our pre-trained model and returns it """

        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

    def return_probabs(self, img):
        """  It returns the Probabilities of each emotions using pre-trained model """

        self.preds = self.loaded_model.predict(img)
        return self.preds
    
    


# Creating an instance of the class with the parameters as model and its weights.
test_model = FacialExpressionModel("model.json", "model_weights.h5")

# Loading the classifier from the file.
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#####Main
captureJpeg = 'capture.jpg'
 


result = Emotion_Analysis(captureJpeg)


print(str(result))
sentence = mood(result[3])
print(str(sentence))
