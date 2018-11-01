#decoding: utf-8
#encoding: utf-8

#openCV transform C++ code to python
#python2.7  openCV3.4

import numpy as np
import cv2 as cv
import os
import logging
import time 

def ini():  
    
    TM=time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())
    LOG_FORMAT = "%(levelname)s -[:%(lineno)d]- %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.info('**************************Mason(%s)*******************'%(TM,))
    
def show(image,name):
    cv.imshow('%s'%name,image)
    cv.waitKey(0)
    logging.info('show image....')
    
def save_image(image,name):
    path_out=path+name+'.jpg'
    cv.imwrite(path_out,image)
    logging.info('save image:%s ,done'%name)
    

#####################
#高斯模糊，降噪
def Gauss(image):
    logging.info('Gauss...')
    image_Guass=cv.GaussianBlur(image, (5, 5), 1.5)
    return image_Guass

#图像灰度化, 三种方式：最大，平均，加权平均
def Gary(image):
    logging.info('Gary...')
    iamge_Gary=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return iamge_Gary

#图像增强/边缘检测，sobel算子（选定水平边缘为检测对象）
def Enhencement(image):
    logging.info('Enhencement...')

    Sobel_x=0 # x horizontal，
    Sobel_y= cv.Sobel(image, cv.CV_16S,dx=0,dy=1,ksize=3)
    #dy=1 means vertical, ddepth=CV_16S;,ksize use default;ksize must be 1,3,5,7, what is default?=3
    image_Sobel=Sobel_y
    return image_Sobel

#二值化， 两种方式：全局阈值和 自适应阈值cv.adaptiveThreshold
def Binary(image):
    logging.info('Binary...')
    # for binary type, donot need :MaxvalMaxval=conf.getint('Binary', 'value')# set to be that value, 
    return_value,image_Binary=cv.threshold(image,127,255,cv.THRESH_BINARY)
    return image_Binary

#闭操作，区域连接
def Close(image):# should input binary image
    logging.info('Close...')
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(20,20))  #must be int 方阵
    #func explaination   https://blog.csdn.net/qq_31186123/article/details/78770141
    image_Close=cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)

    return image_Close


#计算边界/轮廓
#func explaination https://blog.csdn.net/hjxu2016/article/details/77833336
def Edge(path):#should input binary image
    logging.info('Edge...')
    #calculation
    image=cv.imread(path,cv.CV_8UC1) 
    binary,contours, hierarchy = cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) 
    #in open CV3， findContours return 3 value; so my version is openCV3

    #draw boundary on the image
    image=cv.drawContours(image,contours,-1,(0,0,255),3)#-1:draw all edge;  3:line width；（0，0，255):RGB color scaler
    image_Edge=image
    return image_Edge,contours #outpur contours is a array, 


#不相连的轮廓求最小外界矩形,并添加到原图中
def draw_Min_Rect(image,contours):
    logging.info('draw_Min_Rect...')
    N= np.array(contours).shape[0]
    for i in range (0,N):
        min_=cv.minAreaRect(contours[i])
        box = cv.boxPoints(min_) 
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (0,0,255), 3)
    #cv.imshow('0', image)
    #cv.waitKey(0)# the number of windows
    image_with_MinRect=image
    return image_with_MinRect



#角度筛选;角度判断与旋转。把倾斜角度大于阈值（如正负30度）的矩形舍弃
def Angle_filter(contours):
    logging.info('Angle_filter...')
    N= np.array(contours).shape[0]
    Box_update=[]
    contours_update=[]
    for i in range (0,N):
        Rect=cv.minAreaRect(contours[i])
        A=Rect[2]#anglle\
        
        if abs(A)<=20:
            contours_update.append(contours[i])
            Box_update.append(Rect)
    return contours_update,Box_update

#Area筛选box
def Rect_filter(contours):
    logging.info('Rect_filter...')
    N= np.array(contours).shape[0]
    contours_update=[]
    Box_update=[]
    for i in range (0,N):
        Rect=cv.minAreaRect(contours[i])
        Area_=Rect[1][0]*Rect[1][1]
        if Area_ >=3000 and Area_<=5000 :
            print Area_            
            contours_update.append(contours[i])
            Box_update.append(Rect)
    return contours_update,Box_update

#######################

#######################
#截取车牌box，
def CutImage(image_orig,Box_update):#Box_update is minAreaRect
    logging.info('CutImage...')
    points=cv.boxPoints(Box_update)
    left_min=min(points[0][0],points[1][0],points[2][0],points[3][0])
    right_max=max(points[0][0],points[1][0],points[2][0],points[3][0])
    up_max=max(points[0][1],points[1][1],points[2][1],points[3][1])
    down_min=min(points[0][1],points[1][1],points[2][1],points[3][1])
    CutImage=image_orig[down_min:up_max,left_min:right_max]

    return CutImage

#统一尺寸，classical 136*36， 但是LeNet 是默认36*36
def Unisize(CutImage):
    logging.info('Unisize...')
    width=conf.getint('Unisize','width')
    hight=conf.getint('Unisize','hight')
    Inter_Method=conf.get('Unisize',Inter_Method)
    cv.resize(CutImage,(width,hight),Inter_Method)
    return image_Unisize

def segment_box(image_Edge,contours,type_):
    logging.info('segment_box...')
    image_with_MinRect=draw_Min_Rect(image_Edge,contours)
    contours_Ang,box_ang=Angle_filter(contours)
    contours_Rect,box_rect=Rect_filter(contours_Ang)

    #save cpa
    for i in box_rect():
        CutImage(image,box_rect)
        image_Unisize=Unisize(CutImage)
        path_tmp=args["tmp_path"]+name
        cv.imwrite(path_tmp,image_Unisize)
    #SVM classification
    svm= SVM_(type_)
    for i in box_rect():
        CutImage(image,box_rect)
        image_Unisize=Unisize(CutImage)
        svm.predict(Unisize_image)
        if result ==Has:
            image_located=Unisize_image
            break
    return iamge_located #only one image should be output

'''def sobel_locate(iamge,name):
    logging.info('sobel_locate...')
    image_Gauss=Gauss(image)
    iamge_Gary=Gary(image_Gauss)
    image_Sobel=Enhencement(iamge_Gary)
    image_Binary=Binary(image_Sobel)
    image_Close=Close(image_Binary)
    image_Edge,contours=Edge(image_Close)
    
    image_located=segment_box(image,contours,'sobel_')

    return image_located'''


#########################

if __name__ == '__main__':
    ini()
    path='./image_proprocess/'
    
    path_='./image_proprocess/orig3.jpg'
    
    image_orig=np.array(cv.imread(path_))
    
    
    image_Gauss=Gauss(image_orig)
    save_image(image_Gauss,"image_Gauss")
    
    iamge_Gary=Gary(image_Gauss)
    save_image(iamge_Gary,"image_Gary")
    
    image_Sobel=Enhencement(iamge_Gary)
    save_image(image_Sobel,"image_Sobel")
    
    image_Binary=Binary(image_Sobel)
    save_image(image_Binary,"image_Binary")
    
    image_Close=Close(image_Binary)
    save_image(image_Close,"image_Close")
    for i in range(0,1):
        image_Close=Close(image_Close)
        save_image(image_Close,"image_Close")
    

   
    image_Edge,contours=Edge('./image_proprocess/image_Close.jpg')
    save_image(image_Edge,"image_Edge")
    #print np.array(contours).shape
    filter=1 #1 means open the two filters 
    if filter!=1:
        image_with_MinRect=draw_Min_Rect(image_orig,contours)
        save_image(image_with_MinRect,"image_with_MinRect")
    else:
        contours_update,Box_update=Angle_filter(contours)
        contours_update,Box_update=Rect_filter(contours_update)
        image_with_MinRect_update=draw_Min_Rect(image_orig,contours_update)
        save_image(image_with_MinRect_update,"image_with_MinRect_update")
        
    
    
    
    
    
    
    
    

