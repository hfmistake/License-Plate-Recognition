from ultralytics import YOLO
from predictions import Prediction

if __name__ == '__main__':
    my_data = {
        "video_source": "videos/2.mp4",
        "line": [100, 300, 1800, 300],
        "object_indices": [2, 3, 5, 7],
        "directory": "results/results22"
    }
    my_model = YOLO('models/yolov8n.pt')
    prediction = Prediction(my_model, my_data)
    prediction.simple_track_and_capture()
