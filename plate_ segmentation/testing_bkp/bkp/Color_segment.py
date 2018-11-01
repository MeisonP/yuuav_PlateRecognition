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




#将RGB模型图像映射到HVS模型
def HSV(image):#image is a equliaed image
    logging.info('HSV...')
    image_HSV= cv.cvtColor(image, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(image_HSV)
    save_image(H,'H')
    save_image(S,'S')
    save_image(V,'V')
    return image_HSV, H, S, V

def Gary(image):
    logging.info('Gary...')
    iamge_Gary=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return iamge_Gary

#直方图均衡/灰度均衡，使得输入图像在每一级灰度上都有相同的像素点输出,
#输入为单通道，所以对于多通道的RGB要3个通道单独输入
def Equalization(image_RGB):
    logging.info('RGB_Equalization...')
    B, G, R = cv.split(image_RGB)
    print np.array(R).shape
    save_image(R,'R')
    save_image(G,'G')
    save_image(B,'B')
    
    R_Equal=cv.equalizeHist(R)
    save_image(R_Equal,'R_Equal')
    G_Equal=cv.equalizeHist(G)
    save_image(G_Equal,'G_Equal')
    B_Equal=cv.equalizeHist(B)
    save_image(B_Equal,'B_Equal')
    
    RGB_merged = cv.merge([B,G,R])
    image_Equal_RGB=RGB_merged
    return image_Equal_RGB

#二值化分割
def Segment_Blue(image_orig,image_HSV,H,S,V):#must be HSV 
    logging.info('Segment_Blue...')
    H_Blue_Min=90
    H_Blue_Max=130
    S_Blue_Min=80
    S_Blue_Max=255
    V_Blue_Min=90
    V_Blue_Max=255
    LowerBlue = np.array([H_Blue_Min, S_Blue_Min,V_Blue_Min])
    UpperBlue = np.array([H_Blue_Max, S_Blue_Max,V_Blue_Max])
    mask= cv.inRange(image_HSV, LowerBlue, UpperBlue)#in range then set to 1, 白色； 0=背景黑色
    Things= cv2.bitwise_and(image_orig,image_orig,mask=mask)#图像与运算，类似于显示原图中的区域
    image_Segment_Blue=Things
    return image_Segment_Blue


def Segment_Yellow(image_orig,image_HSV,H,S,V):#must be HSV 
    logging.info('Segment_Yellow...')
    H_Yellow_Min=15
    H_Yellow_Max=26
    S_Yellow_Min=43
    S_Yellow_Max=255
    V_Yellow_Min=46
    V_Yellow_Max=255
    LowerYellow = np.array([H_Yellow_Min, S_Yellow_Min,V_Yellow_Min])
    UpperYellow = np.array([H_Yellow_Max, S_Yellow_Max,V_Yellow_Max])
    mask= cv.inRange(image_HSV, LowerYellow, UpperYellow)#in range then set to 1, 白色； 0=背景黑色
    Things= cv2.bitwise_and(image_orig,image_orig,mask=mask)#图像与运算，类似于显示原图中的区域
    image_Segment_Yellow=Things
    
    return image_Segment_Yellow
    



#计算边界/轮廓
#func explaination https://blog.csdn.net/hjxu2016/article/details/77833336
def Edge(path):#should input binary image
    logging.info('Edge...')
    #calculation
    image=cv.imread(path,cv.CV_8UC1) 
    binary,contours, hierarchy = cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) 
    #in open CV3， findContours return 3 value; so my version is openCV3

    #draw boundary on the image
    #image=cv.drawContours(image,contours,-1,(0,0,255),3)#-1:draw all edge;  3:line width；（0，0，255):RGB color scaler
    #image_Edge=image
    return contours #outpur contours is a array, 

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


#Area筛选box
def Rect_filter(contours):
    logging.info('Rect_filter...')
    N= np.array(contours).shape[0]
    contours_update=[]
    Box_update=[]
    for i in range (0,N):
        Rect=cv.minAreaRect(contours[i])
        Area_=Rect[1][0]*Rect[1][1]
        if int(Area_) >=3000:
            print Area_
            if int(Area_)<=12000 :           
                contours_update.append(contours[i])
                Box_update.append(Rect)
    return contours_update,Box_update

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

def color_locate(image):#yellow or blue
    image_HSV=HSV(image)
    image_Equal=Equalization(image_HSV)
    
    #Blue
    image_Segment_Blue=Segment_Blue(image_Equal)
    save_image(image_Segment_Blue,'image_Segment_Blue')
    path=path+"image_Segment_Blue"
    image_Segment_Blue=cv.imread(path)
    image_Edge_b,contours_b=Edge(image_Segment_Blue)
    k_b= np.array(contours_b).shape[0]
    image_located_b=segment_box(image,contours_b,'color_')

    #Yellow
    image_Segment_Yellow=Segment_Yellow(image_Equal)
    image_Edge_y,contours_y=Edge(image_Segment_Yellow)
    k_y=np.array(contours_y).shape[0]
    image_located_y=segment_box(image,contours_y,'color_')

    if k_b==1 or k_b==2:
        iamge_located=image_located_b
        k=k_b
    else:
        iamge_located=image_located_y
        k=k_y
    return  k,iamge_located #k is the number of rectangle





def locate_mechanism(image,name):
    #do color locate first, and judge if need to do sobel locate
    k,result_=color_locate(image)

    if k==1 or k==2:
        image_located=result_
    else:
        image_located=sobel_locate(iamge,name)

    return image_located




#########################

if __name__ == '__main__':
    ini()
    path='./image_proprocess/'
    path_='./image_proprocess/orig4.jpg'
    image_orig=cv.imread(path_)
    
    image_Equalized=Equalization(image_orig)
    save_image(image_Equalized,'image_Equalized')
    
    image_HSV,H,S,V=HSV(image_Equalized)
    save_image(image_HSV,'image_HSV')
    #print np.array(image_HSV).shape #is h, [1] is w, [2] is ch
    
    #################
    image_Segment_Blue=Segment_Blue(image_orig,image_HSV,H,S,V)
    save_image(image_Segment_Blue,'image_Segment_Blue') 
    
    contours=Edge(path+'image_Segment_Blue.jpg')
    os.remove(path+'image_Segment_Blue.jpg')
    
    filter=1
    if filter!=1: #raw model
        image_with_MinRect=draw_Min_Rect(image_orig,contours)
        save_image(image_with_MinRect,'image_with_MinRect')      
    else:
        # filter
        contours_update,Box_update=Angle_filter(contours)
        contours_update,Box_update=Rect_filter(contours_update)
        
        #color check
        if np.array(contours_update).shape[0]<=3: #Blue if 0 or >3 then the car and plate have same color  
            image_with_MinRect_update=draw_Min_Rect(image_orig,contours_update)# draw on the image_orig object  
            save_image(image_with_MinRect_update,'image_with_MinRect_update')
        else:#Yellow
                image_Segment_Yellow=Segment_Yellow(image_orig,image_HSV,H,S,V)
                save_image(image_Segment_Yellow,'image_Segment_Yellow') 
                contours=Edge(path+'image_Segment_Yellow.jpg')
                os.remove(path+'image_Segment_Yellow.jpg')
                
                contours_update,Box_update=Rect_filter(contours)
                image_with_MinRect_update=draw_Min_Rect(image_orig,contours_update)# draw on the image_orig object  
                save_image(image_with_MinRect_update,'image_with_MinRect_update')

