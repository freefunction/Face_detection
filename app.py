import streamlit as st
import cv2 
from PIL import Image
import numpy as np
import os

#load pretrained parameters for the classifier 

try:
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
	smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
	st.write("Error loading classifier")
	
def detect(image):
	''' function to detect faces, eyes and smiles '''
	image = np.array(image.convert('RGB'))
	faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
	for(x,y,w,h) in faces:
		cv2.rectangle(img=image, pt1=(x,y), pt2=(x+w, y+h), color = (255,0,0), thickness = 2)
		roi =image[y:y+h, x:x+w]
		eyes  = eye_cascade.detectMultiScale(roi)
		smile = smile_cascade.detectMultiScale(roi, minNeighbors=25)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi,(ex,ey),(ex+ew, ey+eh), (0,255,0),2)
		for (sx,sy,sw,sh) in smile:
			cv2.rectangle(roi,(sx,sy),(sx+sw, sy+sh), (0,0,255),2)
	return image, faces

def about():
	st.write(
		'''
		**Haar Cascade** :point_right: is an object detection algorithm.
		it can be used to detect objects in images or videos
		''')
def main():
	st.title("face detection App :sunglasses:")
	st.write("**using the Haar Cascade**")
	activities =["Home", "About"]
	
	choice = st.sidebar.selectbox("Pick something fun", activities)
	
	if choice == "Home":
		st.write("Go to the about section from the sidebar to learn more about it")
		image_file = st.file_uploader("Upload image", type = ['jpeg','png','jpg','webp'])
		
		if image_file is not None:
			image = Image.open(image_file)
			
			if st.button("Process"):
				result_img, result_faces = detect(image=image) #result_img is image with rectangle and result_faces is the array coordinates
				st.image(result_img, use_column_width = True)
				st.success("Found {}faces".format({len(result_faces)}))
	
	elif choice == "About":
		about()

if __name__ == "__main__":
	main()
				

	
		
	