#decoding: utf-8
#encoding: utf-8

#openCV transform C++ code to python
#python2.7  openCV3.3
import cv as cv
import argparse
import configparser
import os

def pre(): 
    ##
    class myconf(ConfigParser.ConfigParser):
        def __init__(self, defaults=None):
            ConfigParser.ConfigParser.__init__(self, defaults=defaults)
        def optionxform(self, optionstr):
            return optionstr

    conf=myconf()
    ##

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_path", required=True,
        help="path of the original image")
    ap.add_argument("-o", "--output_path", required=True,
        help="path of the output image")
    ap.add_argument("-t", "--tmp_path", required=True,
        help="path of the tmporary image")
    args = vars(ap.parse_args())

    return args， conf

#####################
#高斯模糊，降噪
def Gauss(image):
    kernel_size=conf.getint('Gaussian_Blur', 'kernel_size')
    sigma=conf.getint('Gaussian_Blur', 'sigma')
    image_Guass=cv.GaussianBlur(imge, kernel_size, sigma)
    return image_Guass

#图像灰度化, 三种方式：最大，平均，加权平均
def Gary(image):
    alpha=conf.getint('Gary_Transform', 'alpha')
    beta=conf.getint('Gary_Transform', 'bate')
    gamma=conf.getint('Gary_Transform', 'gamma')
    #平均
    #

    #加权平均
    for i in range(image.height):  
        for j in range(image.width):  
        iamge_Gary[i,j] = alpha * image[i,j][0] + beta * image[i,j][1] +  gamma * image[i,j][2]
    return iamge_Gary

#图像增强/边缘检测，sobel算子（选定水平边缘为检测对象）
def Enhencement(iamge):
    depth=conf.getint('Soble_Enhencement', 'depth')
    ksize=conf.getint('Soble_Enhencement', 'ksize') 

    soble_x=0 # x horizontal，
    soble_y= cv.sobel(imae, depth,ksize)
    image_Soble=soble_y
    return image_Soble

#二值化， 两种方式：全局阈值和 自适应阈值cv.adaptiveThreshold
def Binary (image):
    threshold=conf.getint('Binary', 'threshold')
    # for binary type, donot need :MaxvalMaxval=conf.getint('Binary', 'value')# set to be that value, 
    threshold_type=conf.get('Binary', 'threshold_type')
    return_value,image_Binary=cv.threshold（iamge,threshold,threshold_type）
    return image_Binary

#闭操作，区域连接
def Close(image):# should input binary image
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(5,5))  
    #func explaination   https://blog.csdn.net/qq_31186123/article/details/78770141
    image_Close=cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)

    return image_Close


#计算边界/轮廓
#func explaination https://blog.csdn.net/hjxu2016/article/details/77833336
def Edge(image):#should input binary image
    model_type=conf.get('Edge','type')
    approx_method=conf.get('Edge','approx_method')
    #calculation
    binary,contours, hierarchy = cv.findContours(binary,model_type,approx_method) 
    #in open CV3， findContours return 3 value; so my version is openCV3

    #draw boundary on the image
    draw_type=conf.getint('Edge','draw_type')
    R=conf.getint('Edge','color_R')
    G=conf.getint('Edge','color_G')
    B=conf.getint('Edge','color_B')
    line_width=conf.getint('Edge','line_width')
    cv.drawContours(image,contours,draw_type,(B,G,R),line_width)#-1:draw all edge;  3:line width；（0，0，255):RGB color scaler
    image_Edge=image
    return image_Edge， contours=#outpur contours is a array, 


#不相连的轮廓求最小外界矩形,并添加到原图中
def draw_Min_Rect(image,contours):
    draw_type=conf.getint('draw_Min_Rect','draw_type')
    R=conf.getint('draw_Min_Rect','color_R')
    G=conf.getint('draw_Min_Rect','color_G')
    B=conf.getint('draw_Min_Rect','color_B')
    line_width=conf.getint('draw_Min_Rect','line_width')

    N= np.array(contours).shape[0]
    for i in range (0,N):
        min_=cv.minAreaRect(contours[i])
        box = cv.boxPoints(min_) 
        box = np.int0(box)
        cv.drawContours(image, [box], draw_type, (B,G,R), line_width)
    #cv.imshow('0', image)
    #cv.waitKey(0)# the number of windows
    image_with_MinRect=image
    return image_with_MinRect



#角度筛选;角度判断与旋转。把倾斜角度大于阈值（如正负30度）的矩形舍弃
def Angle_filter(contours):
    angle_th=conf.getint('Angle','angle_th')
    N= np.array(contours).shape[0]
    for i in range (0,N):
        Rect=cv.minAreaRect(contours[i])
        A=Rect[2]#anglle
        Box_update=[]
        contours_update=[]
        if abs(A)<=angle_th:
            contours_update.append(contours[i])
            Box_update.append(Rect)
    return contours_update,Box_update

#Area筛选box
def Rect_filter(contours):
    DL=conf.getint('Rect_filter','DL')
    UL=conf.getint('Rect_filter','UL')
    N= np.array(contours).shape[0]
    for i in range (0,N):
        Rect=cv.minAreaRect(contours[i])
        Area_=Rect[1][0]*Rect[1][1]
        Box_update=[]
        contours_update=[]
        if Area_ >= DL and Area_<=UL :
            contours_update.append(contours[i])
            Box_update.append(Rect)
    return contours_update,Box_update

#截取车牌box，
def CutImage(image_orig,Box_update):#Box_update is minAreaRect
    points=cv.boxPoints(Box_update)
    left_min=min(points[0][0],points[1][0],points[2][0],points[3][0])
    right_max=max(points[0][0],points[1][0],points[2][0],points[3][0])
    up_max=max(points[0][1],points[1][1],points[2][1],points[3][1])
    down_min=min(points[0][1],points[1][1],points[2][1],points[3][1])
    CutImage=image_orig[down_min:up_max,left_min:right_max]

    return CutImage

#统一尺寸，classical 136*36， 但是LeNet 是默认36*36
def Unisize(CutImage):
    width=conf.getint('Unisize','width')
    hight=conf.getint('Unisize','hight')
    Inter_Method=conf.get('Unisize',Inter_Method)
    cv.resize(CutImage,(width,hight),Inter_Method)
    return image_Unisize

'''#PCA 特征提取
def PCA_feature():

    return feature_vector
'''

'''def SVM_soble(feature_vector):
    model_path=conf.get('model','model_path')
    svm= cv.svm.load('model_path')
    result=svm.predict(feature_vector)
    return result'''

def SVM_(type_):
    if type_='sobel_':
        svm_path_sobel=conf.get('model','svm_path_sobel')
        svm= cv.svm.load('svm_path_sobel')
    else:
        svm_path_color=conf.get('model','svm_path_color')
        svm= cv.svm.load('svm_path_color')
    return svm

def segment_box(image_Edge,contours,type_):
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

def sobel_locate(iamge,name):
    image_Gauss=Gauss(image)
    iamge_Gary=Gary(image_Gauss)
    image_Soble=Enhencement(iamge_Gary)
    image_Binary=Binary(image_Soble)
    image_Close=Close(image_Binary)
    image_Edge,contours=Edge(image_Close)
    
    image_located=segment_box(image,contours,'sobel_')

    return image_located


#########################












#将RGB模型图像映射到HVS模型
def HSV(image):
    image_HSV= cv.cvtColor(image, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(image_HSV)
    return image_HSV

#直方图均衡/灰度均衡，使得输入图像在每一级灰度上都有相同的像素点输出
def Equalization(image_gray):
    img_Equal= cv.equalizeHist(img_gray)
    return image_Equal

#二值化分割
def Segment_Blue(image):#must be HSV image after equalization
    S_Blue_Min=conf.getint('Segment','S_Blue_Min')
    S_Blue_Max=conf.getint('Segment','S_Blue_Man')
    V_Blue_Min=conf.getint('Segment','V_Blue_Min')
    V_Blue_Max=conf.getint('Segment','V_Blue_Man')
    H_Blue_Min=conf.getint('Segment','H_Blue_Min')
    H_Blue_Max=conf.getint('Segment','H_Blue_Man')
    for i in range(image_HSV.height):  
        for j in range(image_HSV.width):     
            if H in range(H_min,H_max) and /
            S in range(S_min,S_max) and /
            V in range(V_min,V_max):
                image_HSV[i,j]=255
            else:
                image_HSV[i,j]=0
    return image_Segment_Blue
def Segment_Yellow(image_HSV):
    S_Yellow_Min=conf.getint('Segment','S_Yellow_Min')
    S_Yellow_Max=conf.getint('Segment','S_Yellow_Man')
    V_Yellow_Min=conf.getint('Segment','V_Yellow_Min')
    V_Yellow_Max=conf.getint('Segment','V_Yellow_Man')
    H_Yellow_Min=conf.getint('Segment','H_Yellow_Min')
    H_Yellow_Max=conf.getint('Segment','H_Yellow_Man')
    for i in range(image_HSV.height):  
        for j in range(image_HSV.width):     
            if H in range(H_min,H_max) and /
            S in range(S_min,S_max) and /
            V in range(V_min,V_max):
                image_HSV[i,j]=255
            else:
                image_HSV[i,j]=0
    return image_Segment_Yellow

def color_locate(image):#yellow or blue
    image_HSV=HSV(image)
    image_Equal=Equalization(image_HSV)
    
    #Blue
    image_Segment_Blue=Segment_Blue(image_Equal)
    image_Edge+b,contours_b=Edge(image_Segment_Blue)
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










if __name__ == '__main__':

    args, conf=pre()

    #load the image
    # grab the image paths and randomly shuffle them
    image_paths = sorted(list(paths.list_images(args['path'])))

    #begin to process
    for i in image_paths: # i is the path of single image
        iamge=cv.imread(i)
        name=os.path.split(i)[1]#split path to be name[1] and direction[0]
        
        image_located=locate_mechanism(image,name)

        #save the results
        path_out=args["output_path"]+name
        cv.imwrite(path_out,image_located)

        




