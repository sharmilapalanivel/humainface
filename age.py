from pyagender import PyAgender
from fer import FER
import face_recognition
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


image=cv2.imread("./training_img/staff.jpg")

cascPath = 'haarcascade_frontalface_default.xml'
	
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
count=0
for (x, y, w, h) in faces: 
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 256, 0), 2)
    count=count+1
print(count)
print("*****")
for i in range(count):
    agender = PyAgender()
    face = agender.detect_genders_ages(image)
    detector = FER()
    res = detector.detect_emotions(image)
#print(face)
ages=[]
gend=[]
genderr=[]
disgust=[]
angry=[]
fear=[]
sad=[]
surprise=[]
neutral=[]
happy=[]
emo=[]
for j in range(count):
    ages.append(round(face[j]["age"]))
    gend.append(face[j]["gender"])
    print(ages[j])
    if gend[j] > 0.5:
	    print("Gender : female")
	    genderr.append("Female")

    else:
	    print("Gender : male")
	    genderr.append("Male")
    angry.append(res[j]["emotions"]["angry"])
    disgust.append(res[j]["emotions"]["disgust"])
    fear.append(res[j]["emotions"]["fear"])
    happy.append(res[j]["emotions"]["happy"])
    sad.append(res[j]["emotions"]["sad"])
    surprise.append(res[j]["emotions"]["surprise"])
    neutral.append(res[j]["emotions"]["neutral"])
    if angry[j] > 0.5:
	    print("Emotion : Angry")
	    emo.append("Angry")
    elif disgust[j] > 0.5:
	    print("Emotion : Disgust")
	    emo.append("Disgust")
    elif fear[j] > 0.5:
	    print("Emotion : Fear")
	    emo.append("Fear")
    elif happy[j] > 0.5:
	    print("Emotion : Happy")
	    emo.append("Happy")
    elif sad[j] > 0.5:
	    print("Emotion : Sad")
	    emo.append("Sad")
    elif surprise[j] > 0.5:
	    print("Emotion : Surprise")
	    emo.append("Surprise")
    elif neutral[j]>0.5:
	    print("Emotion : Neutral")
	    emo.append("Neutral")
    else:
        print("no")
#print(emo)

font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (00, 185) 
  
# fontScale 
fontScale = 1
   
# Red color in BGR 
color = (0, 0, 255) 
  
# Line thickness of 2 px 
thickness = 2
text=[]
for i in range(count):
    text.append("{},{},{}".format(ages[i],genderr[i],emo[i]))
#print(text)	
	# Draw a rectangle around the faces

for (x, y, w, h),j in zip(faces,range(count)): 
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 256, 0), 2)
	cv2.putText(image,text[j] ,(x, y), font, fontScale,color, thickness, cv2.LINE_AA, False)
# display the output image
cv2.namedWindow("face detection",cv2.WINDOW_NORMAL)
imS = cv2.resize(image,(2000,2000))
cv2.imshow("face detection",imS)
cv2.waitKey(0)
