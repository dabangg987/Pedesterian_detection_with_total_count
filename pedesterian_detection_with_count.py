import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

cap = cv.VideoCapture(r"C:\Users\gangw\OneDrive\Desktop\OPencv\video_1.mp4")
ret, frame1 = cap.read()
ret, frame2  = cap.read()

max_count = 0

while cap.isOpened():
	ret, frame3 = cap.read()
	if not ret:
		break

	diff = cv.absdiff(frame1, frame3)
	diff_gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)

	blur = cv.GaussianBlur(diff_gray, (5,5), 0)
	_, thresh = cv.threshold(blur, 20 , 255,cv.THRESH_BINARY)

	dilated = cv.dilate(thresh, None, iterations =3)
	contours, _ = cv.findContours(
		dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	people_count = 0
	pedesterian = 0

	for contour in contours:
		(x,y,w,h) = cv.boundingRect(contour)
        
		# For large crowd set less or qual 300 or small set less then 900 or more
		if cv.contourArea(contour) < 300:
			continue

		cv.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
		cv.putText(frame1, "Pedestrian", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		people_count += 1


	max_count = max(people_count, max_count)
	
	#cv.drawContors (frame1 ,contours, -1 ,(0,255,0),2)
	cv.putText(frame1, f"Pedestrians: {people_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
	people_count = 0

	cv.imshow("Video",frame1)
	frame1 = frame2
	ret, frame2 = cap.read()

     # For exit in between video press 'q'
	if cv.waitKey(5) & 0xFF == ord('q'):
		exit()

cap.release()
cv.destroyAllWindows()

print(max_count)










