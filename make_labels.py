import cv2
import numpy as np

cv2.namedWindow('vid', cv2.WINDOW_NORMAL)

for i in xrange(1, 3845):
	fn = '../training_data/all/output_' + '{:04d}'.format(i) + '.png'

	img = cv2.imread(fn, cv2.IMREAD_COLOR)

	blur = cv2.blur(img, (15, 15))
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	thresh = cv2.inRange(hsv, np.array([24, 46, 60]), np.array([180, 255, 255]))

	contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	

	mc = None
	ma = 0
	for cnt in contours:
		rep = cv2.contourArea(cnt)
		if rep > ma:
			mc = cnt
			ma = rep
	
	x, y, w, h = cv2.boundingRect(mc)
	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

	fi = open('../training_data/labels.csv', 'a+')
	fi.write(str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + '\n')
	fi.close()
	
	cv2.imshow('vid', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindow()

