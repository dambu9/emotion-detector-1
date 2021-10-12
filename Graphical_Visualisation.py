# Let us import the Libraries required.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FacialExpressionModel
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import logging

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotionanalysis.db'
app.config['SECRET_KEY'] = 'ec9439cfc6c796ae2029594d'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Creating an instance of the class with the parameters as model and its weights.
test_model = FacialExpressionModel("model.json", "model_weights.h5")

# Loading the classifier from the file.
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

def new_values(result):
    if result=="Happy":
        return 'Happy'
    elif result=="Neutral":
         return 'Neutral'
    else :
        return 'Anxiety'

def Emotion_Analysis(img):
    """ It does prediction of Emotions found in the Image provided, does the 
    Graphical visualisation, saves as Images and returns them """

    # Read the Image through OpenCv's imread()
    path = "static/images/" + str(img)
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
        Text = str(new_values(prediction)) + Symbols[str(prediction)]
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
        path = "static/images/" + "pred" + str(img)
        cv2.imwrite(path, image)

        # List of Emotions
        EMOTIONS = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

         # List of Emotions
        NEW_EMOTIONS = [ "Happy",
                    "Neutral", "Anxiety/Stress"]            

        # Finding the Probability of each Emotion
        preds = test_model.return_probabs(roi[np.newaxis, :, :, np.newaxis])

        # Converting the array into list
        data = preds.tolist()[0]

        happy_prop = 0
        neutral_prop = 0
        other = 0
        for i in range(0,len(data)):
            if i == 3:
                happy_prop = data[i]
            elif i == 4:
                neutral_prop = data[i]
            else:
                other += data[i]

        new_cordinates = [happy_prop, neutral_prop, other]
        logging.info("cordintates sum agg",new_cordinates[2])
        # Converting the array into list
        data = preds.tolist()[0]
        print(str(data))
        image_analysis_to_create = ImageAnalysis(details=str(list(zip(EMOTIONS, data))), Angry=data[0], Disgust=data[1], Fear=data[2], Happy=data[3], Neutral=data[4], Sad=data[5], Surprise=data[6])
        db.session.add(image_analysis_to_create)
        db.session.commit()
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

        # Saving the Bar Plot
        path = "static/images/" + "bar_plot" + str(img)
        plt.savefig(path)
       
    # Returns a list containing the names of Original, Predicted, Bar Plot Images
    return ([img, "pred" + img, "bar_plot" + img, prediction])
