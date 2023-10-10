import cv2
pingwin = cv2.imread("pingwin.jpg")
nowy_pingwin = cv2.resize(pingwin,(300,300))
#cv2.imshow('image',pingwin)
cv2.imshow('image',nowy_pingwin)
cv2.waitKey(0)