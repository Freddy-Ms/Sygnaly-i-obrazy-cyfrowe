import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt
datas = [r"Data/kwiaty.jpg",r"Data/panda.npy"]
datass = [r"Data/circle.npy",r"Data/panda.npy"]

def Laplace_detection(image):
    Laplace = np.array([[0,1,0],
                        [1,-4,1],
                        [0,1,0]])
    return np.dstack([cv2.filter2D(image[:,:,i],-1,Laplace) for i in range(3)])

def Sobel(image): # return SobelX, SobelY, SobelXY
    Sobel_X= np.array([[1,0,-1],
                       [2,0,-2],
                       [1,0,-1]])
    Sobel_Y= np.array([[1,2,1],
                       [0,0,0],
                       [-1,-2,-1]])
    result_X = np.dstack([cv2.filter2D(image[:,:,i],-1,Sobel_X) for i in range(3)])
    result_Y = np.dstack([cv2.filter2D(image[:,:,i],-1,Sobel_Y) for i in range(3)])
    result_XY = result_X + result_Y
    return result_X, result_Y, result_XY

def Prewitt(image):
    Prewitt_X = np.array([[-1,0,1],
                          [-1,0,1],
                          [-1,0,1]])
    Prewitt_Y = np.array([[1,1,1],
                          [0,0,0],
                          [-1,-1,-1]])
    result_X = np.dstack([cv2.filter2D(image[:,:,i],-1,Prewitt_X) for i in range(3)])
    result_Y = np.dstack([cv2.filter2D(image[:,:,i],-1,Prewitt_Y) for i in range(3)])
    result_XY = result_X + result_Y
    return result_XY
def Scharr(image):
    Scharr_X = np.array([[-3,0,3],
                         [-10,0,10],
                         [-3,0,3]])
    Scharr_Y = np.array([[-3,-10,-3],
                         [0,0,0],
                         [3,10,3]])
    result_X = np.dstack([cv2.filter2D(image[:,:,i],-1,Scharr_X) for i in range(3)])
    result_Y = np.dstack([cv2.filter2D(image[:,:,i],-1,Scharr_X) for i in range(3)])
    result_XY = result_X + result_Y
    return result_XY
def Gaussian(image):
    GaussianA = np.array([[1,2,1],
                         [2,4,2],
                         [1,2,1]]) / 16
    return np.dstack([cv2.filter2D(image[:,:,i],-1,GaussianA) for i in range(3)])

# Function making Gaussian filter of given size 
def Gaussian_array(size):
    GaussianA = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            GaussianA[i][j]= np.exp(-0.5 * ((i - size / 2) ** 2 + (j - size / 2) ** 2) / (2 * 1 ** 2))
    return GaussianA / np.sum(GaussianA)

def Gaussian_55(image):
    GaussianA = Gaussian_array(5)
    return np.dstack([cv2.filter2D(image[:,:,i],-1,GaussianA) for i in range(3)])

def Gaussian_77(image):
    GaussianA = Gaussian_array(7)
    return np.dstack([cv2.filter2D(image[:,:,i],-1,GaussianA) for i in range(3)])

def AverageBlurr(image,size):
    Avg_filter = np.ones((size,size)) / size ** 2
    return np.dstack([cv2.filter2D(image[:,:,i],-1,Avg_filter) for i in range(3)])


def Sharpen(image):
    SharpenA = np.array([[0,-1,0],
                         [-1,5,-1],
                         [0,-1,0]])
    return np.dstack([cv2.filter2D(image[:,:,i],-1,SharpenA) for i in range(3)])


def EdgeDetection_test():
    for data in datass:
        image = np.load(data)
        Laplace = Laplace_detection(image)
        _, _ , SobelDet = Sobel(image)
        PrewittDet = Prewitt(image)
        Scharrr = Scharr(image)

        plt.subplot(1, 2, 1)
        plt.imshow(Laplace, cmap='gray')  
        plt.title("Laplace'a detection")

        plt.subplot(1, 2, 2)
        plt.imshow(SobelDet, cmap="gray")  
        plt.title("Sobel detection")
        plt.show()

        plt.subplot(1, 2, 1)
        plt.imshow(PrewittDet, cmap="gray")  
        plt.title("Prewitt detection")

        plt.subplot(1, 2, 2)
        plt.imshow(Scharrr, cmap="gray")  
        plt.title("Scharr detection")

        plt.show()
def Blurr_test():
    for data in datas:
        if data == r"Data/kwiaty.jpg":
            image = cv2.imread(data)
        else:
            image = np.load(data)
        Gauss = Gaussian(image)
        Gauss55 = Gaussian_55(image)
        Gaus77 = Gaussian_77(image)
        AvgBlurr = AverageBlurr(image,10)

        plt.subplot(1, 2, 1)
        plt.imshow(Gauss, cmap='gray')  
        plt.title("Rozmycie Gaussowskie")

        plt.subplot(1, 2, 2)
        plt.imshow(AvgBlurr, cmap="gray")  
        plt.title("Rozmycie uśredniające 10x10")
        plt.show()

        plt.subplot(1, 2, 1)
        plt.imshow(Gauss55, cmap="gray")  
        plt.title("Rozmycie Gaussowskie 5x5")

        plt.subplot(1, 2, 2)
        plt.imshow(Gaus77, cmap="gray")  
        plt.title("Rozmycie Gaussowskie 7x7")

        plt.show()
def Sharpen_test():
    image1 = cv2.imread(datas[0])
    image2 = np.load(datas[1])
    Sharpen1 = Sharpen(image1)
    Sharpen2 = Sharpen(image2)

    plt.subplot(1, 2, 1)
    plt.imshow(Sharpen1, cmap='gray')  
    plt.title("Wyostrzenie")

    plt.subplot(1, 2, 2)
    plt.imshow(Sharpen2, cmap="gray")  
    plt.title("Wyostrzenie")
    plt.show()     

if __name__ == "__main__":
    EdgeDetection_test()
    Blurr_test()
    Sharpen_test()