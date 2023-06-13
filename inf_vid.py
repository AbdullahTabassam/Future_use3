import math
from ultralytics import YOLO
import cv2
import time


def main():
    video_path = '/home/msc1/Desktop/Tensorflow-Birds_sem1/Base/v1/object_detection/Birds_Sem1.mp4'  # Path to your input video
    output_path = 'YOLO.mp4'  # Path to save the output video
    model_path = 'runs/detect/train2/weights/best.pt'  # Path to model
    cls_name = ['European_Robin', 'Coal_Tit', 'Eurasian_Magpie', 'Common_Blackbird']  # Add class names
    threshold = 0.7  # Score threshold

    model = YOLO(model_path)  # Load model
    cap = cv2.VideoCapture(video_path)  # Open video capture object

    # Get video frame dimensions and create a VideoWriter object for saving the output
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    # FPS calculataion
    start_time = 0
    current_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break

        results = model(frame)  # Perform inference on the frame

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                score = math.ceil((box.conf[0] * 100)) / 100
                if score > threshold:
                    
                    # Draw B-Box on object:
                    color = [(102, 0, 204), (153, 0, 102), (51, 153, 0), (0, 204, 153)]
                    if int(box.cls) == 0:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[int(box.cls)], 2)
                        
                    elif int(box.cls) == 1:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[int(box.cls)], 2)
                        
                    elif int(box.cls) == 2:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[int(box.cls)], 2)
                    
                    elif int(box.cls) == 3:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[int(box.cls)], 2)
                    
                    # Write name of class on top:
                    text_size, _ = cv2.getTextSize(f'{cls_name[int(box.cls)]}  {score}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_w, text_h = text_size
                    cv2.rectangle(frame, (int(x1), int(y1 - 30)), (int(x1) + text_w, int(y1)), color[int(box.cls)], -1)
                  
                    cv2.putText(frame, f'{cls_name[int(box.cls)]}  {score}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = 1 / elapsed_time
        start_time = current_time
                            

        # Display FPS
        cv2.putText(frame, f'FPS: ', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, f'    {int(fps)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 150), 5, cv2.LINE_AA)
        cv2.putText(frame, 'YOLOv8 (L)', (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 150), 5, cv2.LINE_AA)
                            
        out.write(frame)  # Write the processed frame to the output video
        cv2.imshow("Inference", cv2.resize(frame, (640, 360)))  # Display processed frame
        if cv2.waitKey(1) == 27:  # Press escape key to close the window
            break

    cap.release()  # Release video capture object
    out.release()  # Release VideoWriter object
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 
