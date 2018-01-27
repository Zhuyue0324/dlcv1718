"""Data utility functions."""
"""Taken from DL4CV Homework3 and modified the Label list"""
import os
import numpy as np

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

