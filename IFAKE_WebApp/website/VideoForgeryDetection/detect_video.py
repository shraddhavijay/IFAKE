import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
from keras.models import load_model

#vid_name = input("\nEnter the name of video: ")
#vid_src = "G:/Video_Forgery_Detection_Using_Machine_Learning/Input_Videos/" + vid_name + ".mp4"

def detect_video_forgery(vid_src):
    vid = []

    sumFrames =0
    cap= cv2.VideoCapture(vid_src)
    #cap.set(3,240)
    #cap.set(4,320)

    fps = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        #compressImage = cv2.resize(frame, (0, 0), fx = 0.1, fy = 0.1)
        if ret == False:
            fps = cap.get(cv2.CAP_PROP_FPS)
            break
        b = cv2.resize(frame,(320,240),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        sumFrames +=1
        vid.append(b)
    cap.release()
        
    print("\nNo. Of Frames in the Video: ",sumFrames)

    Xtest = np.array(vid)

    print("\nPredicting !! ")
    model = load_model('C://Users//User//ML//Video_Forgery_Detection//ResNet50_Model//forgery_model_me.hdf5')
    output = model.predict(Xtest)

    output = output.reshape((-1))
    results = []
    for i in output:
        if i>0.5:
            results.append(1)
        else:
            results.append(0)


    no_of_forged = sum(results)

    print('No of forged----no_of_forged:',no_of_forged)
            
    if no_of_forged <=0:
        print("\nThe video is not forged")
        return {'result':'Authentic','f_frames':0}
        
    else:
        print("\nThe video is forged")
        print("\nNumber of Forged Frames in the video: ",no_of_forged)
        return {'result':'Forged','f_frames':no_of_forged}
