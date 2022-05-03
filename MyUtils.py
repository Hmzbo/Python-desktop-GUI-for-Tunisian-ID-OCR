import cv2
import numpy as np
from PIL import Image

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU )[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#invert edge detection
def invert(image):
    return 255-image

#skew correction
def deskew(img, limit, delta):
    white = (255,255,255)
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for ang in angles:
        temp_RGBA = Image.fromarray(img).convert('RGBA')
        temp_img = temp_RGBA.rotate(ang, Image.NEAREST, expand = 1, fillcolor = white)
        temp_img = np.array(temp_img.convert('L'))
        hight,width = temp_img.shape
        a = int(hight*0.47)
        b = int(hight*0.53)
        upper=0
        lower =0
        for p in range(width):
            if temp_img[a,p]!=255:
                upper+=1
            if temp_img[b,p]!=255:
                lower+=1
        scores.append(upper+lower)
        
    best_angle = angles[scores.index(np.max(scores))]
    rotated_img = temp_RGBA.rotate(best_angle, Image.NEAREST, expand = 1, fillcolor = white)
    rotated_img = np.array(rotated_img.convert('L'))

    h,w = rotated_img.shape
    top_crop=0
    bot_crop=0
    for p in range(5,h//2):
        a=(h//2)-p
        b=(h//2)+p
        if (np.sum(rotated_img[a,:]==255)==w) & (top_crop==0):
            top_crop=a
        if (np.sum(rotated_img[b,:]==255)==w) & (bot_crop==0):
            bot_crop=b
    cropped_rotated_img = rotated_img[top_crop:bot_crop,:]            
    return cropped_rotated_img

def white_pad(image, pad_size):
    return cv2.copyMakeBorder(image.copy(),pad_size,pad_size,pad_size,pad_size,cv2.BORDER_CONSTANT,value=(255,255,255))

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpen_image(image, alpha=1):
    kernel = np.array([[0, -1, 0],
                    [-1, alpha+4,-1],
                    [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp

def scale(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    return img
