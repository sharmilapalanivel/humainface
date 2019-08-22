import json

with open("face_recognition.json", "r") as file:
  json_data = [json.loads(line) for line in file]

links=[]
for i in range(len(json_data)):
  links.append(json_data[i]["content"])

import os

os.makedirs("training_images")

import requests 
import cv2

for i in range(len(links)):
  
# URL of the image to be downloaded is defined as image_url 
  r = requests.get(links[i]) # create HTTP response object 
  
  image_name= "train_"+ str(i)+".png"
    #cv2.imwrite("training_images",image_name)

 with open(image_name,'wb') as f: 
  
     f.write(r.content)

labels=[]
point=[]
for i in range(len(json_data)):
	length=(len(json_data[i]["annotation"]))
#print (length)
	for j in range(length):
		labels.append(json_data[i]["annotation"][j]["label"])
		point.append(json_data[i]["annotation"][j]["points"])

print(labels)
print(point)

