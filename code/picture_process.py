import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
def claheMethod(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#彩色图要拆分三个通道分别做均衡化，否则像我这里一样转为灰度图
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 自适应均衡化，参数可选
    cl1 = clahe.apply(hsv)
 
    #测试加了滤波没能让边缘清晰
    #         #cl1MedianBlur = cv2.medianBlur(cl1, 1)
        # cl1GaussianBlur = cv2.GaussianBlur(cl1, (1, 1), 0)
    return cl1    
 
def zhifangtu(img):
    mpl.rcParams["font.sans-serif"]=["SimHei"]
    mpl.rcParams["axes.unicode_minus"]=False
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH)) 
    return result

