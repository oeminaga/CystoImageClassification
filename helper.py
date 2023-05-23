'''
Copyright by Okyaz Eminaga 2023
'''
import cv2
import numpy as np
from PIL import Image,ImageOps
import pandas as pd
from GUI import get_ROI

from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
def EminagaAlgorthimPreprocess(frame):
    frm=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img=Image.fromarray(frm)
    #frm=ImageOps.autocontrast(img, cutoff=110)
    #img=ImageOps.autocontrast(img, cutoff=110)
    #img = img.filter(SHARPEN)
    #frm = np.array(img.filter(EDGE_ENHANCE))
    frm = np.array(img)
    x,y,w,h=get_ROI(np.array(frm).astype(np.uint8))
    x_clip =w//4
    y_clip = h//6
    x_start = x + x_clip
    y_start = y + y_clip
    h = w - 2* w//4 #5
    w = w - 2*w//4  #5
    frm=frm[y_start:h+y_start,x_start:w+x_start]
    #frm=applications.mobilenet_v3.preprocess_input(frm)
    return frm, (x_start,y_start,w,h)