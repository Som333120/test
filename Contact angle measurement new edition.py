import cv2
import numpy as np 
import matplotlib.pyplot as plt
import csv
import pandas as pd 
from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave


def nothing(x):
    pass

def selectPoint(point1,point2) :
    mark_1 = result.loc[[point1],['conner_axis-X','conner_axis-Y']]
    mark_2 = result.loc[[point2],['conner_axis-X','conner_axis-Y']]


    pt1 = (int(mark_1.iloc[0,0]),int(mark_1.iloc[0,1]))
    pt2 = (int(mark_2.iloc[0,0]),int(mark_2.iloc[0,1]))
    locat1 = cv2.circle(im,pt1,2,(155,40,90),2,-1)
    locat2 = cv2.circle(im,pt2,2,(155,40,90),2,-1)
    drae_line = cv2.line(im,pt2,pt1,(0,0,255),1,0)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(im)
    ax2.imshow(edge)
    plt.show()
    print(pt1,pt2)
    return pt2,pt1


img =cv2.imread('21.jpg',0)
r = cv2.selectROI('Select Area ',img)
bright = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #crop image 
yen_threshold = threshold_yen(img)
bright = rescale_intensity(bright, (0, yen_threshold), (0, 255)) # AUTO CONTRAST IMAGE 
high_thresh, thresh_im = cv2.threshold(bright, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #calculate hight and low  threshold by OTSU method
lowThresh = 0.5*high_thresh


num = 1 
contact_pointx = []
contact_pointy = []
contact_point = pd.DataFrame([])
data = pd.DataFrame([])
coor = pd.DataFrame([])
while(1) :

    edge = cv2.Canny(bright,high_thresh,lowThresh) #edge detection 
    th, im_th = cv2.threshold(bright, 220, 225, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv

    yen_threshold_ = threshold_yen(im_out)

    lines = cv2.HoughLinesP(
        edge,
        rho = 1.0,
        theta = np.pi/180,
        threshold = 20,
        minLineLength= 2,
        maxLineGap= 0
    )
    line_img = np.zeros((im_out.shape[0],im_out.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    dot_color = [0,255,0]
    
    #resize image 
    scale_percent = 60 # percent of original size
    width = int(im_out.shape[1] * scale_percent / 100)
    height = int(im_out.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(im_out, dim, interpolation = cv2.INTER_AREA)

    #drawing line on edge
    for line in lines :
        for x1, y1, x2, y2 in line :
            #cv2.line(im_out, (x1,y1),(x2,y2),line_color,1)
            #cv2.circle(im_out,(x1,y1),3,line_color,-1)
            #cv2.circle(im_out,(x2,y2),3,line_color,-1)
            data = data.append(pd.DataFrame({'X1': x1, 'Y1': y1, 'X2' :x2, 'Y2':y2}, index=[0]), ignore_index=True)
    
    im = cv2.cvtColor(im_out,cv2.COLOR_GRAY2BGR )
    corners = cv2.goodFeaturesToTrack(im_out, 6, 0.01, 10)
    corners = np.int0(corners)
    for i in corners: 
        x, y = i.ravel() 
        cv2.circle(im, (x, y), 5, (0, 255, 0), -1)
        #cv2.putText(im, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA, True)
        contact_point = contact_point.append(pd.DataFrame({'conner_axis-X': x, 'conner_axis-Y': y,}, index=[0]), ignore_index=True)


    indices = np.where(edge != [0])
    coordinates = zip(indices[0], indices[1])
    k =cv2.waitKey(1) &0xff 

    coor = coor.append(pd.DataFrame({'edge_axis-x': indices[0], 'edge_axis-y': indices[1]}), ignore_index=True)
    column = coor["edge_axis-x"]
    max_index = column.min() #find min value on edge 
    cv2.line(im,(0,max_index),(im.shape[1],max_index),color=(255,0,0),thickness=1,lineType=1)

    frames = [coor, data, contact_point]
    result = pd.concat(frames,axis =1 )
    result.to_csv('result.csv', encoding='utf-8', index=False)
    break
    #cv2.imshow('test',im)
    if k == 27 :
        break
fig,(ax1,ax2) = plt.subplots(1,2)
ax1.imshow(im)
ax2.imshow(edge)

plt.show()
method = str(input("Enter_method"))
if method == str('circle') :
    print("NO")
elif method == str('half'):
    point1=  (int(input("Enter_Point1 ")))
    point2 = (int(input("Enter_Point1 ")))
    print(point1,point2)
    selectPoint(point1,point2)



