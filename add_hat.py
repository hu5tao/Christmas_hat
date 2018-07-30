# -*- coding: utf-8 -*-
import numpy as np 
import cv2
#import dlib#********
import scipy.io as sio #matlab
# 给img中的人头像加上圣诞帽，人脸最好为正脸
def add_hat(img,hat_img,dets_results):
    # 分离rgba通道，合成rgb三通道帽子图，a通道后面做mask用
    r,g,b,a = cv2.split(hat_img) 
    rgb_hat = cv2.merge((r,g,b))

#    cv2.imwrite("hat_alpha.jpg",a)
    len_num=len(dets_results)
    if len_num>0:
        # for d in dets:
        for i1 in range(len_num):
            x,y,w,h = int(dets_results[i1][0]),int(dets_results[i1][1]),int(dets_results[i1][2]-dets_results[i1][0]),int(dets_results[i1][3]-dets_results[i1][1])
            if(x - w / 2<0):
                x=0
            else:
                x=x-w/2
            w, h=int(w+w/3),int(h)

            eyes_center = (int(x+w/2),int(y+h/2))
            bbox1=[eyes_center[0]-w/2-w/2,eyes_center[1]-h/2,eyes_center[0]+w/2+w/2,eyes_center[1]+h/2]

            #  根据人脸大小调整帽子大小
            factor = 1.2
            factor_h = 1
            resized_hat_h = int(round(rgb_hat.shape[0]*w/rgb_hat.shape[1]*factor_h))
            resized_hat_w = int(round(rgb_hat.shape[1]*w/rgb_hat.shape[1]*factor))

            if resized_hat_h > y:
                resized_hat_h = y-1

            # 根据人脸大小调整帽子大小
            resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))

            # 用alpha通道作为mask
            mask = cv2.resize(a,(resized_hat_w,resized_hat_h))
            mask_inv =  cv2.bitwise_not(mask)

            # 帽子相对与人脸框上线的偏移量
            dh = 0
            dw = 0
            # 原图ROI
            # bg_roi = img[y+dh-resized_hat_h:y+dh, x+dw:x+dw+resized_hat_w]
            bg_roi = img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)]

            # 原图ROI中提取放帽子的区域
            bg_roi = bg_roi.astype(float)
            mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
            alpha = mask_inv.astype(float)/255

            # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
            alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
            # print("alpha size: ",alpha.shape)
            # print("bg_roi size: ",bg_roi.shape)
            bg = cv2.multiply(alpha, bg_roi)
            bg = bg.astype('uint8')

#            cv2.imwrite("bg.jpg",bg)
            # cv2.imshow("image",img)
            # cv2.waitKey()

            # 提取帽子区域
            hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
#            cv2.imwrite("hat.jpg",hat)
            

            hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))
            # 两个ROI区域相加
            add_hat = cv2.add(bg,hat)
            # cv2.imshow("add_hat",add_hat) 

            # 把添加好帽子的区域放回原图
            img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat

            # 展示效果
            # cv2.imshow("img",img )  
            # cv2.waitKey(0)  

    return img

   
# 读取帽子图，第二个参数-1表示读取为rgba通道，否则为rgb通道
hat_img = cv2.imread("hat2.png",-1)

# 读取头像图
img = cv2.imread("timg.jpg")
load_fn = 'detect_results.mat'
load_data = sio.loadmat(load_fn)
load_matrix = load_data['bboxes'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当

output = add_hat(img,hat_img,load_matrix)

# 展示效果
#cv2.imshow("output",output )  
# cv2.waitKey(0)
cv2.imwrite("output.jpg",output)
