import cv2
import torch
import uuid as uid
from PIL import Image


class Prediction:
    def __init__(self, model, data: dict):
        self.model = model
        self.data = data
        self.captured_ids = set()
        Prediction.check_gpu()

    @staticmethod
    def check_gpu():
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"Starting with GPU: {gpu_name}")
        else:
            print("GPU not available. Starting with CPU.")

    def is_point_inside_line(self, center_x, center_y, track_id):
        line_x1, line_y1, line_x2, line_y2 = self.data["line"]
        is_inside_line = line_x1 < center_x < line_x2 and line_y1 - 15 < center_y < line_y1 + 15

        if is_inside_line and track_id not in self.captured_ids:
            self.captured_ids.add(track_id)
            return True

        return False

    def draw_line(self, frame):
        cv2.line(frame, (self.data["line"][0], self.data["line"][1]),
                 (self.data["line"][2], self.data["line"][3]),
                 (88, 85, 232), 4)

    def _process_frame(self, frame):
        results = self.model.track(frame, classes=self.data["object_indices"], conf=0.6, iou=0.5)
        return self.draw_visualization_elements(results[0], results[0].plot())

    def check_and_draw_collision(self, result, original_frame):
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2 - x1, y2 - y1
            center_x, center_y = x1 + width // 2, y1 + height // 2
            center_point = (center_x, center_y)
            cv2.circle(original_frame, center_point, 7, (88, 85, 232), -1)
            if box.id is not None and self.is_point_inside_line(center_x, center_y, int(box.id.item())):
                print("teste")
                cv2.putText(original_frame, "Captura realizada", (0, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (88, 85, 232), 3, cv2.LINE_AA)
                track_id = int(box.id.item())
                msg = f"Captured vehicle id: {track_id}"
                cv2.imshow(msg, original_frame)

        return original_frame

    def draw_visualization_elements(self, result, original_frame):
        self.draw_line(original_frame)
        self.check_and_draw_collision(result, original_frame)
        return original_frame

    def visualize(self):
        cap = cv2.VideoCapture(self.data["video_source"])
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            annotated_frame = self._process_frame(frame)
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def simple_track_and_capture(self):
        directory = self.data["directory"]
        object_indices = self.data["object_indices"]
        results = self.model.track(self.data["video_source"], stream=True, iou=0.5, classes=object_indices, conf=0.6)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                center_x, center_y = x1 + width // 2, y1 + height // 2
                if box.id is not None and self.is_point_inside_line(center_x, center_y, int(box.id.item())):
                    im = Image.fromarray(result.orig_img)
                    im.save(f'{directory}/{uid.uuid4()}.jpg')
                    print("captura realizada")
