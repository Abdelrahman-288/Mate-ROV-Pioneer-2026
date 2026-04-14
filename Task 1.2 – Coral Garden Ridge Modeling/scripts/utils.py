import cv2
import numpy as np

def blur_score(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def image_diff(img1, img2):
    return np.mean(cv2.absdiff(img1, img2))

def load_image(path):
    return cv2.imread(path)