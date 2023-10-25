import numpy as np 
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import supervision as sv
import time


#frames per second function
def get_fps(frame_number):
    if frame_number == 0:
        get_fps.start_time = time.time()
        return 0.0
    end_time = time.time()
    fps = float(frame_number) / float(end_time - get_fps.start_time)
    return fps

#intersection distance
def intersection_distance(rect1, rect2):
    (x11, y11, x12, y12) = rect1
    (x21, y21, x22, y22) = rect2
    
    #coordinates of intersection
    if x12<x21 or x11>x22 or y12<y21 or y11>y22:
        intersect = 0.0
    else:
        #find coordinates of intersection
        x1 = max(x11, x21)
        y1 = max(y11, y21)
        x2 = min(x12, x22)
        y2 = min(y12, y22)
        
                
        intersect = float((x2-x1)*(y2-y1))
        
    return intersect  

#side distance
def side_distance(rect1, rect2):
    
    (x11, y11, x12, y12) = rect1
    (x21, y21, x22, y22) = rect2
    #get width and height of the second rect
    width2 = x22-x21
    height2 = y22-y21  
             
    
    if y21>=y11-height2 and y21<=y12: #horizontal area
        d = min([np.abs(x11-x21),
               np.abs(x11-x22),
               np.abs(x12-x21),
               np.abs(x12-x22)])
        
    elif x21>=x11-width2 and x21<=x12: #vertical area
        d = min([np.abs(y11-y21),
               np.abs(y11-y22),
               np.abs(y12-y21),
               np.abs(y12-y22)]) 
        
    elif x21>=x12 and y22<=y11: #upper right area
        d = np.sqrt((x21-x12)**2+(y22-y11)**2)
                
    elif x22<=x11 and y22<=y11: #upper left area
        d = np.sqrt((x22-x11)**2+(y22-y11)**2)
        
        
    elif x22<=x11 and y21>=y12: #bottom left area
        d = np.sqrt((x22-x11)**2+(y21-y12)**2)
        
        
    elif x21>=x12 and y21>=y12: #bottom right area
        d = np.sqrt((x21-x12)**2+(y21-y12)**2)   
    
    return d  


def main(args):
    
    model = YOLO(args.model)
    byte_tracker = sv.ByteTrack()
    
    cap = cv2.VideoCapture(args.video_in)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    cap_writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*'MJPG'),
                            25, (frame_width, frame_height))

    success, frame = cap.read()
    
    frame_num=0

    while success:   
        
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)          
        detections = byte_tracker.update_with_detections(detections)
        bboxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id
        results = []
        machinery_boxes = []
        people_boxes = []
        for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
            
            if class_id == 0: #machinery
                machinery_boxes.append(
                bbox.astype("int")
                )
            elif class_id == 1: #person
                people_boxes.append(
                bbox.astype("int")
                )               
                            
            # print(machinery_boxes)
            bbox = bbox.astype("int")
            cv2.rectangle(frame, bbox[:2], bbox[2:], color = (255,0,0), thickness=2)
            cv2.rectangle(frame, (frame_width-100, 20), (frame_width-10, 60), (0,0,0), -1)
            cv2.putText(frame, "%.2f FPS" % get_fps(frame_num), (frame_width-100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX , fontScale = 0.5, color=(255, 255, 255), thickness=1) 
                    
                            
        for person in people_boxes:
            for machine in machinery_boxes:
                intersection_d = intersection_distance(machine, person)
                side_d = side_distance(machine, person)
                if side_d < 20 or intersection_d > 0:
                    cv2.rectangle(frame, person[:2], person[2:], color = (0, 0, 255), thickness=2) 
                    cv2.rectangle(frame, machine[:2], machine[2:], color = (0, 0, 255), thickness=2)
                    cv2.putText(frame, "ALARM!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX , fontScale = 1, color=(0, 0, 255), thickness=5) 
                                
        imS = cv2.resize(frame, (960, 540))  
        cap_writer.write(frame)      
        cv2.imshow('YOLO V8 Detection', imS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        success, frame = cap.read()
        frame_num+=1
        

    cap.release()

    cv2.destroyAllWindows()
    