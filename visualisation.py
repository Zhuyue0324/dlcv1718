"""Data utility functions."""
"""Taken from DL4CV Homework3 and modified the Label list"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature

SEG_LABELS_LIST = [
    {"id": 0,  "name": "unlabeled",   "rgb_values": [0, 0,    0]},
    {"id": 1,  "name": "ego vehicle",      "rgb_values": [0,   0,  0]},
    {"id": 2,  "name": "rectification border",       "rgb_values": [0, 0,  0]},
    {"id": 3,  "name": "out of roi",        "rgb_values": [0,   0,    0]},
    {"id": 4,  "name": "static",      "rgb_values": [0, 0,    0]},
    {"id": 5,  "name": "dynamic",      "rgb_values": [111,   74,  0]},
    {"id": 6,  "name": "ground",        "rgb_values": [81, 0,  81]},
    {"id": 7,  "name": "road",   "rgb_values": [128,  64,   128]},
    {"id": 8,  "name": "sidewalk",   "rgb_values": [244, 35,   232]},
    {"id": 9,  "name": "parking",      "rgb_values": [250,  170,  160]},
    {"id": 10, "name": "rail track",       "rgb_values": [230, 150,  140]},
    {"id": 11, "name": "building",        "rgb_values": [70,  70,    70]},
    {"id": 12, "name": "wall",    "rgb_values": [102, 102,    156]},
    {"id": 13, "name": "fence",     "rgb_values": [190,  153,  153]},
    {"id": 14, "name": "guard rail",       "rgb_values": [180, 165,  180]},
    {"id": 15, "name": "bridge",       "rgb_values": [150,   100,   100]},
    {"id": 16, "name": "tunnel",       "rgb_values": [150, 120,   90]},
    {"id": 17, "name": "pole",      "rgb_values": [153,   153,  153]},
    {"id": 18, "name": "polegroup",       "rgb_values": [153, 153,   153]},
    {"id": 19, "name": "traffic light",        "rgb_values": [250,   170,  30]},
    {"id": 20, "name": "traffic sign",        "rgb_values": [220, 220,  0]},
    {"id": 21, "name": "vegetation",       "rgb_values": [107,  142,   35]},
    {"id": 22, "name": "terrain",       "rgb_values": [152,  251,   152]},
    {"id": 23, "name": "sky",       "rgb_values": [70,  130,   180]},
    {"id": 24, "name": "person",       "rgb_values": [220,  20,   60]},
    {"id": 25, "name": "rider",       "rgb_values": [255,  0,   0]},
    {"id": 26, "name": "car",       "rgb_values": [0,  0,  142]},
    {"id": 27, "name": "truck",       "rgb_values": [0,  0,   70]},
    {"id": 28, "name": "bus",       "rgb_values": [0,  60,   100]},
    {"id": 29, "name": "caravan",       "rgb_values": [0,  0,   90]},
    {"id": 30, "name": "trailer",       "rgb_values": [0,  0,   110]},
    {"id": 31, "name": "train",       "rgb_values": [0,  80,   100]},
    {"id": 32, "name": "motorcycle",       "rgb_values": [0,  0,   230]},
    {"id": 33, "name": "bicycle",       "rgb_values": [119,  11,   32]},
    {"id": -1, "name": "license plate",       "rgb_values": [0, 0,   142]}]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


def lowest_non_road(label_img):
    label_img = np.squeeze(label_img)

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    mask = np.where(label_img > 10, 1, 0)
    mask = np.diag(range(1024)).dot(mask)
    am = np.argmax(mask,0)
    for i in range(5):
        label_img_rgb[np.minimum(am+i-2,1023),range(2048),:] = [255,0,0]

    return label_img_rgb.astype(np.uint8)

def lowest_non_road_color(color_img, label_img):
    label_img = np.squeeze(label_img)
    color_img_rgb = color_img.transpose(1,2,0)
    mask = np.where(label_img > 10, 1, 0)
    mask = np.diag(range(1024)).dot(mask)
    am = np.argmax(mask,0)
    for i in range(9):
        color_img_rgb[np.minimum(am+i-4,1023),range(2048),:] = [127,0,0]
    return color_img_rgb



def reduce(label_img, road_labels):
    # reduces an image to just two classes, one of them given as input
    mask = np.where(label_img == 0, 1, 0)
    for l in road_labels:
        mask += np.where(label_img == l, 1, 0)
    return mask # is already of shape (1024,2048)

def lnr_basic(label_img):
    label_img = np.squeeze(label_img)
    label_img[1000:,:] = 1 # to avoid the outside_ROI

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    mask = np.where(label_img > 0, 0, 1)
    
    mask = np.diag(range(1024)).dot(mask)
    am = np.argmax(mask,0)
    for i in range(5):
        label_img_rgb[np.minimum(am+i-2,1023),range(2048),:] = [255,0,0]

    return label_img_rgb.astype(np.uint8)

def proximity(x,y,alpha=0.001,beta=0.4):
    # proximity of point (x,y) to (1024,1024), in [0,255]
    dx = np.abs(x-1023)
    dy = np.abs(y-1023)
    return np.sqrt(dy**2 + beta*(dx*(1-alpha*dy))**2)/9

def compare(label_img1, label_img2):
    # WARNING img1 should be target, img2 prediction
    label_img1 = np.squeeze(label_img1)
    label_img2 = np.squeeze(label_img2)
    label_img1[1000:,:] = 1
    label_img2[1000:,:] = 1
    
    res_img = np.array([label_img1,
                              label_img1,
                              label_img1]).transpose(1,2,0)
    mask1 = np.where(label_img1[40:980,:] > 0, 0, 1)
    mask2 = np.where(label_img2[40:980,:] > 0, 1, 0)
    mask1 = np.diag(range(940)).dot(mask1)
    mask2 = np.diag(range(940)).dot(mask2)
    am1 = np.argmax(mask1,0)
    am2 = np.argmax(mask2,0)
    #for i in range(5):
    #    res_img[np.minimum(am1+i-2,1021),range(2048),:] = [255,0,0]
    #    for j in range(2048):
    #        d = proximity(am2[j]+i-2,j)
    #        res_img[np.minimum(am2[j]+i-2,1021),j,:] = [255-d,d,0]
    thresh = 400 * np.ones(2048)
    mse = np.sum(np.minimum((am1-am2)**2,thresh))
    return (mse // 2048)
    #return res_img.astype(np.uint8)

def edge_smoother(img):
    img[0:30,:] = 0
    img[1000:,:] = 1
    mask=feature.canny(img, low_threshold=0, high_threshold=1)
    am = np.argmax(mask,0)
    mask[am,range(2048)] = 1
    return mask


def stixel_bar(color_img, label_img, stride, offset):
    label_img = np.squeeze(label_img)
    color_img_rgb = color_img.transpose(1,2,0)
    mask =np.where(label_img[40:980,offset:-offset] > 0, 1, 0)
    mask = np.diag(range(940)).dot(mask)
    am = np.argmax(mask,0)
    distances = np.zeros(2048-2*offset)
    for i in range(2048-2*offset):
        distances[i]=proximity(i+offset, am[i],alpha=0.00001, beta=0.1)
    distances=((distances-np.min(distances))/(np.max(distances)-np.min(distances))*250).astype(int)
    for i in range(2048-2*offset):
        if i%stride > 5:
            d=distances[i]
            if (i%stride < 13):
                for j in range(am[i]+44):
                    color_img_rgb[j,i+offset,:] = [2+d, 253-d,0]
            elif (i%stride >stride-8):
                for j in range(am[i]+44):
                    color_img_rgb[j,i+offset,:] = [2+d, 253-d,0]
            else:
                for j in range(7):
                    color_img_rgb[am[i]+43-j,i+offset,:] = [2+d, 253-d,0]
                
    return color_img_rgb
    
