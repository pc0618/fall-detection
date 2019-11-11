# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:55:30 2019

@author: Shiru
"""

# resize to 224 X 224 then 3 channel RGB
import os
import cv2
import numpy as np

work_folder = '/home/shirui/Fall_Dataset'

avi_folder = work_folder + '/dataset_avi'

image_folder = work_folder + '/images'
numpy_folder = work_folder + '/numpys'
optical_folder = work_folder + '/optical'

# os.mkdir(image_folder)
# os.mkdir(numpy_folder)
# os.mkdir(optical_folder)

#for scenario in os.listdir(avi_folder):
for scenario in  ['chute11','chute12','chute13','chute14','chute15','chute16','chute17','chute18',
 'chute19','chute20','chute21','chute22','chute23','chute24']:
   
    print(scenario)
    os.mkdir(image_folder+'/'+scenario)
    os.mkdir(numpy_folder+'/'+scenario)
    os.mkdir(optical_folder+'/'+scenario)
    
    for camera in os.listdir(avi_folder+'/'+scenario):
        print(camera)
        os.mkdir(image_folder+'/'+scenario+'/'+camera)
        os.mkdir(numpy_folder+'/'+scenario+'/'+camera)
        os.mkdir(optical_folder+'/'+scenario+'/'+camera)
        
        vidcap = cv2.VideoCapture(avi_folder+'/' + scenario + '/' +camera)
    
        success,image = vidcap.read()
        image = cv2.resize(image,(224,224),interpolation =cv2.INTER_AREA)
          
        prvs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(image)
        hsv[...,1] = 255
        
        count = 0
        while success:
          success,image = vidcap.read()
          if success:
            image = cv2.resize(image,(224,224),interpolation =cv2.INTER_AREA)  
            #cv2.imwrite(image_folder+'/'+scenario+'/' + camera + '/frame%d.jpg' % count, image)   
            np.save(numpy_folder+'/'+scenario+'/' + camera + '/frame%d.npy' % count, np.asarray(image))
              
          if success:
            nex = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,nex, None, 0.5, 3, 15, 3, 5, 1.2, 0)
              
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            
            np.save(optical_folder+'/'+scenario+'/'+ camera+'/opticalhsv%d.npy'% count,np.asarray(rgb))
            prvs = nex
            count += 1
            
        vidcap.release()
        #cv2.destroyAllWindows()

"""
for scenario in os.listdir(read_folderName):
    print(scenario)
    os.mkdir(label_folderName+'/'+scenario)
    if (scenario == 'chute01'):
        size = 1562
        label = np.zeros((size,),dtype=int)
        left = 1080
        right = 1108
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)

    elif (scenario == 'chute02'):
        size = 831
        label = np.zeros((size,),dtype=int)
        left = 375
        right = 399
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
        
    elif (scenario == 'chute03'):
        size = 939
        label = np.zeros((size,),dtype=int)
        left = 591
        right = 625
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
               
    elif (scenario == 'chute04'):
        size = 1049
        label = np.zeros((size,),dtype=int)
        left1 = 288
        right1 = 314
        left2 = 601
        right2 = 638
        label[left1-1:right1] = np.ones((right1-left1+1,),dtype=int)     
        label[left2-1:right2] = np.ones((right2-left2+1,),dtype=int)  
        
    elif (scenario == 'chute05'):
        size = 767
        label = np.zeros((size,),dtype=int)
        left = 311
        right = 336
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)   
        
        
    elif (scenario == 'chute06'):
        size = 1256
        label = np.zeros((size,),dtype=int)
        left = 583
        right = 629
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
        
    elif (scenario == 'chute07'):
        size = 926
        label = np.zeros((size,),dtype=int)
        left = 476
        right = 507
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)   
        
    elif (scenario == 'chute08'):
        size = 713
        label = np.zeros((size,),dtype=int)
        left = 271
        right = 298
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
       
    elif (scenario == 'chute09'):
        size = 943
        label = np.zeros((size,),dtype=int)
        left = 628
        right = 651
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
        
    elif (scenario == 'chute10'):
        size = 933
        label = np.zeros((size,),dtype=int)
        left = 512
        right = 530
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)  
        
    elif (scenario == 'chute11'):
        size = 822
        label = np.zeros((size,),dtype=int)
        left = 464
        right = 489
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)  
        
        
    elif (scenario == 'chute12'):
        size = 937
        label = np.zeros((size,),dtype=int)
        left = 605
        right = 653
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)   
        
    elif (scenario == 'chute13'):
        size = 1206
        label = np.zeros((size,),dtype=int)
        left = 823
        right = 863
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)  
        
    elif (scenario == 'chute14'):
        size = 1499
        label = np.zeros((size,),dtype=int)
        left = 989
        right = 1023
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)

    elif (scenario == 'chute15'):
        size = 1086
        label = np.zeros((size,),dtype=int)
        left = 755
        right = 787
        label[left-1:right,] = np.ones((right-left+1,),dtype=int) 
        
    elif (scenario == 'chute16'):
        size = 1234
        label = np.zeros((size,),dtype=int)
        left = 891
        right = 940
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
        
    elif (scenario == 'chute17'):
        size = 1539
        label = np.zeros((size,),dtype=int)
        left = 730
        right = 770
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
        
    elif (scenario == 'chute18'):
        size = 1001
        label = np.zeros((size,),dtype=int)
        left = 571
        right = 601
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)
        
    elif (scenario == 'chute19'):
        size = 1016
        label = np.zeros((size,),dtype=int)
        left = 499
        right = 600
        label[left-1:right,] = np.ones((right-left+1,),dtype=int) 
        
    elif (scenario == 'chute20'):
        size = 1008
        label = np.zeros((size,),dtype=int)
        left = 545
        right = 672
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)

    elif (scenario == 'chute21'):
        size = 1207
        label = np.zeros((size,),dtype=int)
        left = 864
        right = 901
        label[left-1:right,] = np.ones((right-left+1,),dtype=int) 

    elif (scenario == 'chute22'):
        size = 1109
        label = np.zeros((size,),dtype=int)
        left = 767
        right = 808
        label[left-1:right,] = np.ones((right-left+1,),dtype=int)  
        
    elif (scenario == 'chute23'):
        size = 5381
        label = np.zeros((size,),dtype=int)
        left1 = 1520
        right1 = 1595
        left2 = 3574
        right2 = 3614
        label[left1-1:right1] = np.ones((right1-left1+1,),dtype=int)     
        label[left2-1:right2] = np.ones((right2-left2+1,),dtype=int)  
        
    elif (scenario == 'chute24'):
        size = 1008
        label = np.zeros((size,),dtype=int)

    np.save(label_folderName+'/'+scenario+'/label.npy' , label)
"""        
      