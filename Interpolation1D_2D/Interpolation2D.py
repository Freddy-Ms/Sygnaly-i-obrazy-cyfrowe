import numpy as np
import cv2
import math

def mse(image1, image2):
    return np.square(np.subtract(image1,image2)).mean()

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
    for i in range (height):    # width
        for j in range (width):  #height
            x = i * h_factor  #w_factor
            y = j * w_factor #h_factor
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







image1 = cv2.imread("/Input1_256x256.png")
image2 = cv2.imread("/Input2_640x635.jpg")
image1_500x500 = resize(image1,500,500)
image1_355x420 = resize(image1,355,420)
image1_100x100 = resize(image1,100,100)
image2_1000x700 = resize(image2,1000,700)
image2_200x200 = resize(image2,200,200)
image2_500x500 = resize(image2,500,500)
cv2.imwrite("Image1_500x500.png", image1_500x500)
cv2.imwrite("Image1_355x420.png", image1_355x420)
cv2.imwrite("Image1_100x100.png", image1_100x100)
cv2.imwrite("Image2_1000x700.jpg", image2_1000x700)
cv2.imwrite("Image2_200x200.jpg", image2_200x200)
cv2.imwrite("Image2_500x500.jpg", image2_500x500)
print("MSE(image1,image1_100x100):", mse(image1,resize(image1_100x100,256,256)))
print("MSE(image1,image1_355x420):", mse(image1,resize(image1_355x420,256,256)))
print("MSE(image1,image1_500x500):", mse(image1,resize(image1_500x500,256,256)))
print("MSE(image2,image2_1000x700):", mse(image2,resize(image2_1000x700,640,635)))
print("MSE(image2,image2_200x200):", mse(image2,resize(image2_200x200,640,635)))
print("MSE(image2,image2_500x500):", mse(image2,resize(image2_500x500,640,635)))