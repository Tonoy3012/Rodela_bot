from typing import List
import cv2
from queue import Queue
from kthread import KThread
import mediapipe as mp
from utils import Rectangle
from face_recognizer import Face, FaceNameType
import numpy as np

def check_camera(id):
    cap = cv2.VideoCapture(id)
    if cap.isOpened():
        print("Camera is working")
    else:
        print("Camera is not working")
        
known_face_encodings = []
known_face_names = []

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection #type:ignore
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence)

    def detect(self, frame) -> List[Rectangle]:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.detections:
            face_rects = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                face_rects.append(Rectangle(bbox.xmin * w, bbox.ymin * h, bbox.width * w, bbox.height * h))
            return face_rects
        return []
    
    def fancy_draw(self, frame, face_rects):
        for face_rect in face_rects:
            cv2.rectangle(frame, (int(face_rect.x), int(face_rect.y)), (int(face_rect.x + face_rect.width), int(face_rect.y + face_rect.height)), (0, 255, 0), 2)
        return frame


class Vision:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.frame_queue = Queue()
        self.frame = None
        self.running = False
        self.thread = None
        
        self.viewing = False
        self.viewing_thread = None
        
        # objects
        
        # face
        self.face_detector = FaceDetector()
        self.face_queue = Queue()
        self.faces:list[Face] = []
        self.max_diff = 10

    
    def check_for_faces(self,frame):
        face_rects = self.face_detector.detect(frame)
        if face_rects:
            self.face_queue.put((face_rects, frame))
            self.start_face_recognizer()
        else:
            self.faces:list[Face] = []
        
            
    def __get_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_queue.put(frame)
                t = KThread(target=self.check_for_faces, args=(frame,))
                t.start()
            else:
                print("Frame not found")

    def start(self):
        self.running = True
        self.thread = KThread(target=self.__get_frame)
        self.thread.start()
    
    def start_face_recognizer(self):
        if not self.face_queue.empty():
            face_rects, frame = self.face_queue.get()
            if self.faces:
                face_rects_vals = np.array([face.rect.val for face in self.faces])
                keep_faces = []
                for face_rect in face_rects:
                    diff = np.abs(face_rects_vals - face_rect.val)
                    
                    min_diff = None
                    min_idx = None
                    
                    for idx, d in enumerate(diff):
                        d = abs(d)
                        if min_diff is None:
                            min_diff = d
                            min_idx = idx
                        else:
                            if d < min_diff:
                                min_diff = d
                                min_idx = idx
                    
                    if min_diff <= self.max_diff:
                        self.faces[min_idx].update(face_rect, frame)
                        keep_faces.append(self.faces[min_idx])
                    else:
                        self.faces.append(Face(face_rect, frame))
                        keep_faces.append(idx)
                new_faces = []
                for idx, face in enumerate(self.faces):
                    if idx in keep_faces:
                        new_faces.append(face)
                    else:
                        face.terminate_process()
                self.faces = new_faces
            
            else:
                for face_rect in face_rects:
                    self.faces.append(Face(face_rect, frame))

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.kill()
            self.thread = None

    def get_frame(self):
        if not self.frame_queue.empty():
            self.frame = self.frame_queue.get()
        return self.frame
    
    def __view(self):
        while self.viewing:
            frame = self.get_frame()
            if frame is not None:
                frame = self.check_for_faces(frame)

                for face in self.faces:
                    frame = face.draw_face(frame)
            
                cv2.imshow("frame", frame) if frame.size > 0 else None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop()
        cv2.destroyAllWindows()
    
    def view(self):
        self.viewing = True
        t = KThread(target=self.__view)
        t.start()
        self.viewing_thread = t
        return t
        
    def stop_view(self):
        self.viewing = False
        if self.viewing_thread:
            self.viewing_thread.kill()
            self.viewing_thread = None



if __name__ == "__main__":
    v = Vision()
    v.start()
    v.view()
    
    for face in v.faces:
        if face.waiting_for_recognize and face.nametype == FaceNameType.unknown:
            face.recognize_process.start()
    