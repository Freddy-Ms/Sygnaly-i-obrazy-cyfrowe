import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt

def calculate_mse(img1, img2):
    return np.square(np.subtract(img1, img2)).mean()

def MosaicingBayer(image):
    height, width, c = image.shape
    image = transform.resize(image, output_shape=(height,width,c))
    
    red = np.zeros((height,width), dtype=np.uint8)
    red[::2, 1::2] = 1

    green = np.zeros((height,width), dtype=np.uint8)
    green[::2,::2] = 1
    green[1::2, 1::2] = 1

    blue = np.zeros((height,width), dtype=np.uint8)
    blue[1::2,::2] = 1
    
    bayer = np.dstack((red,green,blue))

    return bayer * image 

def MosaicingFuji(image):
    pattern = np.array([
        [
                [0,0,1,0,0,1],
                [1,0,0,0,0,0],
                [0,0,0,1,0,0],                 #RED
                [0,0,1,0,0,1],
                [1,0,0,0,0,0],
                [0,0,0,1,0,0]        
        ],
        [
                [1,0,0,1,0,0],
                [0,1,1,0,1,1],
                [0,1,1,0,1,1],                 #GREEN
                [1,0,0,1,0,0],
                [0,1,1,0,1,1],
                [0,1,1,0,1,1]  
        ],
        [
                [0,1,0,0,1,0],
                [0,0,0,1,0,0],
                [1,0,0,0,0,0],                 #BLUE
                [0,1,0,0,1,0],
                [0,0,0,1,0,0],
                [1,0,0,0,0,0]  
        ]
    ])

    height, width, c = image.shape
    rows_repeat = height // 6 + 1
    cols_repeat = width // 6 + 1

    #Reapat for red channel sequence
    reapeated_pattern_red = np.tile(pattern[0],(rows_repeat,cols_repeat))
    result_red = reapeated_pattern_red[:height, :width]

    #Reapat for green channel sequence
    reapeated_pattern_green = np.tile(pattern[1],(rows_repeat,cols_repeat))
    result_green = reapeated_pattern_green[:height, :width]

    #Reapat for blue channel sequence
    reapeated_pattern_blue = np.tile(pattern[2],(rows_repeat,cols_repeat))
    result_blue = reapeated_pattern_blue[:height, :width]

    Mask = np.dstack((result_red, result_green, result_blue))
    
    return Mask * image



def demosaic_Bayer_interpolation(image,type="average"):
    height, width, c = image.shape

    red_channel = np.zeros((height,width))
    green_channel = np.zeros((height,width))
    blue_channel = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            if i % 2 == 0: # wiersz nieparzysty     Wszystko odwrotnie, ponieważ iterację zaczynamy od 0 a nie 1
                if j % 2 == 0: #kolumna nieparzysta
                    green_channel[i,j] = image[i,j,1]
                else: #kolumna parzysta
                    red_channel[i,j] = image[i,j,0] 
            else: # wiersz parzysty
                if j % 2 == 0: # kolumna nieparzysta
                    blue_channel[i,j] = image[i,j,2]
                else: # kolumna parzysta
                    green_channel[i,j] = image[i,j,1]

    # Interpolacja koloru czerwonego w wierszach nieparzystych jako średnia sumy dwóch sąsiadujących pikseli (wysokość)
    for i in range(1,height,2):
        for j in range(1,width,2):
            neighbors = [
            (i - 1, j),  # Górny sąsiad
            (i + 1, j)   # Dolny sąsiad
            ]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    red_channel[i, j] = np.mean([red_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    red_channel[i, j] = np.max([red_channel[x, y] for x, y in valid_neighbors])
    # Interpolacja koloru czerwonego w kolumnach jako średnia sumy dwóch sąsiadujących pixeli (szerokość)
    for i in range(height):
        for j in range(0,width,2):
            neighbors = [
            (i, j - 1),  # Lewy sąsiad
            (i, j + 1)   # Prawy sąsiad
            ]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    red_channel[i, j] = np.mean([red_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    red_channel[i, j] = np.max([red_channel[x, y] for x, y in valid_neighbors])
    # Interpolacja koloru zielonego w kolumnach jako średnia sumy czterech sąsiadujących pixeli albo 3 albo 2, jeżeli są to piksele na brzegu
    for i in range(0, height, 2):
        for j in range(1, width, 2):
            neighbors = [
                (i + 1, j),  # Dolny sąsiad
                (i - 1, j),  # Górny sąsiad
                (i, j + 1),  # Prawy sąsiad
                (i, j - 1)   # Lewy sąsiad
            ]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    green_channel[i, j] = np.mean([green_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    green_channel[i, j] = np.max([green_channel[x, y] for x, y in valid_neighbors])

    for i in range(1, height, 2):
        for j in range(0, width, 2):
            neighbors = [
                (i + 1, j),  # Dolny sąsiad
                (i - 1, j),  # Górny sąsiad
                (i, j + 1),  # Prawy sąsiad
                (i, j - 1)   # Lewy sąsiad
            ]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    green_channel[i, j] = np.mean([green_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    green_channel[i, j] = np.max([green_channel[x, y] for x, y in valid_neighbors])
    # Interpolacja koloru niebieskiego w wierszach parzystych
    for i in range(0,height,2):
        for j in range(0,width,2):
            neighbors = [
            (i - 1, j),  # Górny sąsiad
            (i + 1, j)   # Dolny sąsiad
        ]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    blue_channel[i, j] = np.mean([blue_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    blue_channel[i, j] = np.max([blue_channel[x, y] for x, y in valid_neighbors])
    # Interpolacja koloru niebieskiego w kolumnach
    for i in range(height):
        for j in range(1,width,2):
            neighbors = [
            (i, j - 1),  # Lewy sąsiad
            (i, j + 1)   # Prawy sąsiad
        ]

            valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]

            if valid_neighbors:
                if type == "average":
                    blue_channel[i, j] = np.mean([blue_channel[x, y] for x, y in valid_neighbors])
                elif type == "max":
                    blue_channel[i, j] = np.max([blue_channel[x, y] for x, y in valid_neighbors])

    result = np.dstack((red_channel,green_channel,blue_channel))
    return result

def demosaic_Bayer_convolution(image):
    bayer_mask = np.array([[[1, 1],
                            [1, 1]],

                           [[0.5, 0.5],
                            [0.5, 0.5]],

                           [[1, 1],
                            [1, 1]]])
    result = np.dstack([cv2.filter2D(image[:, :, i], -1, bayer_mask[i]) for i in range(3)])
    return result

def Fuji(image): # R - 8 / 36   G - 20 / 36 8/36
    fuji_mask = np.array([[[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]],

                          [[1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20],
                           [1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20],
                           [1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20],
                           [1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20],
                           [1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20],
                           [1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20, 1 / 20]],

                          [[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]]])

    result = np.dstack([cv2.filter2D(image[:, :, i], -1, fuji_mask[i]) for i in range(3)])
    return result

def Bayer_test(method): # 1 - interpolacja average 2 - konwolucja 3 - interpolacja maxpool 
    paths = [r"Bayer/circle.npy",r"Bayer/milky-way.npy",r"Bayer/mond.npy",r"Bayer/namib.npy",r"Bayer/pandas.npy"]
    for path in paths:
        image = np.load(path)
        
        plt.figure()
        if method == 1:
            final = demosaic_Bayer_interpolation(image,type="average")
            plt.suptitle(f"Demozaikowanie poprzez interpolację 2D (average pooling) \n {path}")
        elif method == 2:
            final = demosaic_Bayer_convolution(image)
            plt.suptitle(f"Demozaikowanie poprzez konwolucję. \n {path}")
        elif method == 3:
            final = demosaic_Bayer_interpolation(image,type="max")
            plt.suptitle(f"Demozaikowanie poprzez interpolację 2D (max pooling). \n {path}")  
        else:
            raise ValueError("Inappropriate argument")
        
       
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')  
        plt.title("Maska Bayera")

        plt.subplot(1, 2, 2)
        plt.imshow(final, cmap="gray")  
        plt.title("Zdemozaikowany obraz")
        
        plt.show()
def Fuji_test():
    paths = [r"Fuji/circle.npy",r"Fuji/milky-way.npy",r"Fuji/mond.npy",r"Fuji/namib.npy",r"Fuji/panda.npy"]
    for path in paths:
        image = np.load(path)
        image1 = MosaicingFuji(image)
        plt.figure()
        final = Fuji(image1)
        mse = round(calculate_mse(image,final),5)
        plt.suptitle(f"Demozaikowanie poprzez konwolucję. \n {path} \n MSE: {mse}")
        
        plt.subplot(1, 2, 1)
        plt.imshow(image1, cmap='gray')  
        plt.title("Maska Fuji")

        plt.subplot(1, 2, 2)
        plt.imshow(final, cmap="gray")  
        plt.title("Zdemozaikowany obraz")
        plt.show()
def comparision_avg_max():
    paths = [r"Bayer/circle.npy",r"Bayer/milky-way.npy",r"Bayer/mond.npy",r"Bayer/namib.npy",r"Bayer/pandas.npy"]
    for path in paths:
        image = np.load(path)
        max = demosaic_Bayer_interpolation(image,"max")
        avg = demosaic_Bayer_interpolation(image,"average")
        mse = calculate_mse(max,avg)
        mse = round(mse,10)
        plt.figure()    
        plt.subplot(1, 2, 1)
        plt.imshow(max, cmap='gray')  
        plt.title("Max Pooling")
        plt.suptitle(f"Porównanie interpolacji max pooling vs average pooling \n MSE: {mse}")
        plt.subplot(1, 2, 2)
        plt.imshow(avg, cmap="gray")  
        plt.title("Average Pooling")
        plt.show()
def comparision_original_avg_max():
    paths = [r"Fuji/circle.npy",r"Fuji/milky-way.npy",r"Fuji/mond.npy",r"Fuji/namib.npy",r"Fuji/panda.npy"]
    for path in paths:
        image = np.load(path)
        max = demosaic_Bayer_interpolation(image,"max")
        avg = demosaic_Bayer_interpolation(image,"average")
        mse_avg = calculate_mse(image,avg)
        mse_avg = round(mse_avg,10)
        mse_max = calculate_mse(image,max)
        mse_max = round(mse_max,10)
        print(f"{path} dla Max Pooling: {mse_max} dla Average: {mse_avg}")
def comparision_avg_convolution():
    paths = [r"Fuji/circle.npy",r"Fuji/milky-way.npy",r"Fuji/mond.npy",r"Fuji/namib.npy",r"Fuji/panda.npy"]
    for path in paths:
        image = np.load(path)
        image1 = MosaicingBayer(image)
        avg = demosaic_Bayer_interpolation(image1,"average")
        conv = demosaic_Bayer_convolution(image1)
        
        mse_avg = calculate_mse(image,avg)
        mse_avg = round(mse_avg,10)
        mse_conv = calculate_mse(image,conv)
        mse_conv= round(mse_conv,10)
        print(f"{path} dla Convolution: {mse_conv} dla Average: {mse_avg}")




if __name__ == "__main__":
    comparision_avg_convolution()
