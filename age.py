from pyagender import PyAgender
from fer import FER
import face_recognition
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


image=cv2.imread("./nallam.jpg")

agender = PyAgender()
faces = agender.detect_genders_ages(image)

ages = round(faces[0]["age"])
gend=(faces[0]["gender"])

detector = FER()
res = detector.detect_emotions(image)

angry = res[0]["emotions"]["angry"]
disgust = res[0]["emotions"]["disgust"]
fear = res[0]["emotions"]["fear"]
happy = res[0]["emotions"]["happy"]
sad = res[0]["emotions"]["sad"]
surprise = res[0]["emotions"]["surprise"]
neutral = res[0]["emotions"]["neutral"]

print("************************")
print("Age : ",round(faces[0]["age"]))

if gend > 0.5:
	print("Gender : female")
	gender="Female"

else:
	print("Gender : male")
	gender="Male"

if angry > 0.5:
	print("Emotion : Angry")
	emo="Angry"
elif disgust > 0.5:
	print("Emotion : Disgust")
	emo="Disgust"
elif fear > 0.5:
	print("Emotion : Fear")
	emo="Fear"
elif happy > 0.5:
	print("Emotion : Happy")
	emo="Happy"
elif sad > 0.5:
	print("Emotion : Sad")
	emo="Sad"
elif surprise > 0.5:
	print("Emotion : Surprise")
	emo="Surprise"
else:
	print("Emotion : Neutral")
	emo="Neutral"
print("************************")
#cv2.imwrite("sampleimage.jpg",image)


cascPath = "haarcascade_frontalface_default.xml"
	
	# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
	
	# Read the image
	
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Detect faces in the image
faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30),
	#flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (00, 185) 
  
# fontScale 
fontScale = 1
   
# Red color in BGR 
color = (0, 0, 255) 
  
# Line thickness of 2 px 
thickness = 4


text="{}, {}, {}".format(ages,gender,emo)
	
	# Draw a rectangle around the faces
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 256, 0), 2)
	cv2.putText(image,text ,(x, y), font, fontScale,color, thickness, cv2.LINE_AA, False)
# display the output image
cv2.namedWindow("face detection",cv2.WINDOW_NORMAL)
imS = cv2.resize(image,(2000,2000),thickness)
cv2.imshow("face detection",imS)
cv2.waitKey(0)
