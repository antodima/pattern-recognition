#!/usr/bin/env python3
import matplotlib.pyplot as plt


def normalized_cut(image_path):
    """
    +--------------+
    | Assignment 4 |
    +--------------+
    
    Perform image segmentation on one of the eight image thematic subsets. 
    Note that each file has a name starting with a number from 1 to 8, 
    which indicates the thematic subset, followed by the rest of the file name. 
    I suggest to use image subsets “1_*” or “2_*”.  
    Use the normalized cut algorithm to perform image segmentation. 
    You are welcome to confront the result with kmeans segmentation algorithm if you wish.
    """
    from skimage import segmentation, color
    from skimage.io import imread
    from skimage.future import graph

    image = imread(image_path)
    labels1 = segmentation.slic(image, compactness=30, n_segments=400)
    segmented_image = color.label2rgb(labels1, image, kind='avg')
    g = graph.rag_mean_color(image, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    cut_normalized_image = color.label2rgb(labels2, image, kind='avg')
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
    ax[0].imshow(segmented_image)
    ax[1].imshow(cut_normalized_image)
    return fig
    

def sift_descriptor(image_path):
    """
    +--------------+
    | Assignment 5 |
    +--------------+
    
    Select one image from each of the eight thematic subsets (see previous assignment), for a total of 8 images. 
    Extract the SIFT descriptors for the 8 images using the visual feature detector embedded in SIFT 
    to identify the points of interest. Show the resulting points of interest overlapped  on the image. 
    Then provide a confrontation between two SIFT descriptors showing completely different information 
    (e.g. a SIFT descriptor from a face portion Vs a SIFT descriptor from a tree image). 
    The confrontation can be simply visual: for instance you can plot the two SIFT descriptors 
    closeby as barplots (remember that SIFTs are histograms). But you are free to pick-up 
    any reasonable means of confronting the descriptors (even quantitatively, if you whish).
    
    >> "pip3 install opencv-python opencv-contrib-python"
    """
    import cv2 as cv
    
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des
    

def sobel_filter(image_path):
    import cv2 as cv
    
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_sobelx = cv.Sobel(gray, cv.CV_8U, 1, 0, ksize=3)
    img_sobely = cv.Sobel(gray, cv.CV_8U, 0, 1, ksize=3)
    return img_sobelx, img_sobely
