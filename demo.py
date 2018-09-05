#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import argparse
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo,read_type):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    #writeVideo_flag = True

    #geneate a video object
    video_dir='/home/liuyp/liu/mot/deep_sort_yolov3/model_data/demo2.wmv'
    video=video_open(read_type,video_dir)
    video_capture = video.generate_video()
    fps=0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        t1 = time.time()

        image = Image.fromarray(frame)
        time3=time.time()
        boxs = yolo.detect_image(image)
        time4=time.time()
        print('detect cost is',time4-time3)
       # print("box_num",len(boxs))
        time3=time.time()
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        time4=time.time()
        print('features extract is',time4-time3)
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('', frame)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()

    cv2.destroyAllWindows()

class video_open:
    def __init__(self,read_type,video_dir):
        #self.readtype=read_type
        if read_type=='video':
            self.readtype=0
        else:
            self.readtype=video_dir

    def generate_video(self):
        video_capture=cv2.VideoCapture(self.readtype)
        return video_capture

######################paraters######################
def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--read_type", help="camera or video",
        default='camera', required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(YOLO(),args.read_type)
