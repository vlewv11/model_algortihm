import cv2
import numpy as np
from typing import List, Tuple
from model import Model
import cvzone


class VideoProcessor:
    def __init__(self, model, tracker):
        self.video = None
        self.inside_polygons = None
        self.outside_polygons = None
        self.width = None
        self.height = None
        self.model = model
        self.tracker = tracker

        self.inside = [set()]
        self.entry_times = {}
        self.count = 0
        self.visitors = []
        self.time_spent = []
        self.n_inside_polygons = 0
        self.n_outside_polygons = 0

    def init_model(self):
        self.model = Model()
        self.model.gpu_init()

    def set_size(self, width, height):
        self.width = width
        self.height = height

    def set_polygons(self, in_polygon: List[List[Tuple[int, int]]], out_polygon: List[List[Tuple[int, int]]]):
        try:
            self.n_inside_polygons = len(in_polygon)
            self.n_outside_polygons = len(out_polygon)
            self.visitors = [0 for _ in range(self.n_inside_polygons)]
            self.entry_times = [{} for _ in range(self.n_inside_polygons)]
            self.time_spent = [set() for _ in range(self.n_inside_polygons)]
            self.inside = [set() for _ in range(self.n_inside_polygons)]
            self.inside_polygons = np.int32(in_polygon)
            self.outside_polygons = np.int32(out_polygon)
            print(f"Set polygons of shape: {self.outside_polygons.shape}, {self.inside_polygons.shape}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}\nPolygons shape (n_polygons, n_coordinates:4, xy:2)")

    def set_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("No video")
            if self.width is None or self.height is None:
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Width and height set to {self.width}, {self.height}")
            self.video = cap
        except Exception as e:
            print(f"Wrong video path and error: {e}")

    def visualize_frame(self, frame):
        cvzone.putTextRect(frame, f"VISITED: {self.visitors}", (50, 60), 2, 2)
        # cvzone.putTextRect(frame, f"CURRENTLY INSIDE: {len(self.inside)}", (50, 160), 2, 2)
        cvzone.putTextRect(frame, f"TIME SPENT: {self.time_spent}", (50, 260), 2, 2)
        for polygon in self.inside_polygons:
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
        for polygon in self.outside_polygons:
            cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)
        cv2.imshow("RGB", frame)

    def draw_bboxes(self, frame, x3, y3, x4, y4, id):
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cv2.circle(frame, (x3, y4), 7, (255, 0, 255), -1)
        cv2.circle(frame, (x4, y4), 7, (255, 0, 255), -1)
        cvzone.putTextRect(frame, f"Person-{id}", (x3, y3), 1, 1)

    def detect_oclisions(self, x3, y4, id):
        for i in range(self.n_inside_polygons):
            inside_check = cv2.pointPolygonTest(
                self.inside_polygons[i], (x3, y4), False
            )
            if inside_check >= 0 and (id not in self.inside[i]):
                self.visitors[i] += 1
                self.inside[i].add(id)
                self.entry_times[i][id] = self.count
        for i in range(self.n_outside_polygons):
            outside_check = cv2.pointPolygonTest(
                self.outside_polygons[i], (x3, y4), False
            )
            if outside_check >= 0 and (id in self.inside[i]):
                self.time_spent[i].add((self.count - self.entry_times[i][id]) / 25)
                self.inside[i].discard(id)
                self.entry_times[i].pop(id)

    def process_video(self, visual=False):
        if self.video is None:
            raise Exception("No video")
        if self.inside_polygons is None or self.outside_polygons is None:
            raise Exception("No polygons")
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break
                self.count += 1
                if self.count % 10 != 0:
                    continue
                results = self.model.predict(frame)
                if results is None:
                    continue
                for result in results:
                    detections = []
                    for r in result.boxes.data.tolist():
                        x1, y1, x2, y2, score, _ = r
                        detections.append([x1, y1, x2, y2, score])
                    self.tracker.update(frame, detections)

                for track in self.tracker.tracks:
                    x3, y3, x4, y4 = track.bbox
                    x3 = int(x3)
                    y3 = int(y3)
                    x4 = int(x4)
                    y4 = int(y4)
                    id = track.track_id
                    if visual:
                        self.draw_bboxes(frame, x3, y3, x4, y4, id)
                    self.detect_oclisions(x3, y4, id)
                if visual:
                    self.visualize_frame(frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Random occurences of new ids substracted from the total number of visitors
            for i in range(len(self.visitors)):
                self.visitors[i] - len(self.inside[i])
            self.video.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Something with process: {e}")
