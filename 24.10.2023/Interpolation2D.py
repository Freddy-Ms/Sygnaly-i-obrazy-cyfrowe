import numpy as np
import cv2
import math

def resize(image,width,height):
    if len(image.shape) == 3:
        old_h, old_w, c = image.shape
    elif len(image.shape) == 2:
         old_h, old_w = image.shape
         c = 1
         image = image.reshape((old_h,old_w,1))
    new_image = np.zeros((height,width,c))
    w_factor = old_w / width if width != 0 else 0
    h_factor = old_h / height if height != 0 else 0
    for i in range (height):
        for j in range (width):
            x = i * h_factor
            y = j * w_factor
            x_floor = math.floor(x)
            y_floor = math.floor(y)
            x_ceil = min(old_h -1, math.ceil(x))
            y_ceil = min(old_w -1, math.ceil(y))

            if(x_floor == x_ceil) and (y_floor == y_ceil):
                new_image[i,j,:] = image[x_floor,y_floor,:]
            elif(x_ceil == x_floor):
                q1 = image[int(x),int(y_floor),:]
                q2 = image[int(x),int(y_ceil),:]
                new_image[i,j,:] = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif(y_ceil == y_floor):
                q1 = image[x_floor,y_floor,:]
                q2 = image[x_ceil, y_floor,:]
                new_image[i,j,:] = q1 * (x_ceil - x) + q2 * (x - x_floor)
            else:
                v1 = image[x_floor,y_floor,:]
                v2 = image[x_ceil, y_floor,:]
                v3 = image[x_floor, y_ceil,:]
                v4 = image[x_ceil, y_ceil,:]
                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                new_image[i,j,:] = q1 * (y_ceil - y) + q2 * (y - y_floor)
    return new_image.squeeze() if c== 1 else new_image







image =cv2.imread("Obraz.jpg")
#image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#print(image_gray.shape)
obraz = resize(image,500,300)
cv2.imwrite("Output.png",obraz)
cv2.waitKey(0)