import speech_recognition as sr
import cv2
import mediapipe as mp
import time
import kthread
import numpy as np
import face_recognition as fr
import os
import platform
import uuid
import random
from mimetypes import guess_type
import datetime
import requests
import serial
from utils import app_dir, temp_dir
from voice import Speaker

all_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
img_dir = "imgs"

speaker = Speaker()
host = "http://localhost:5000"

# Helper functions
def random_file_name(ex=".jpg"):
    return "".join(random.choices(all_chars, k=10)) + ex

recent_names = {}

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def val(self):
        return (self.x + self.y + self.w + self.h) / 4

    @property
    def p1(self):
        return (self.x, self.y)

    @property
    def p2(self):
        return (self.x + self.w, self.y + self.h)

    @property
    def p3(self):
        return (self.x, self.y + self.h)

    @property
    def p4(self):
        return (self.x + self.w, self.y)

    def diff(self, other):
        p1 = (self.x - other.x, self.y - other.y)
        p2 = (self.w - other.w, self.h - other.h)
        p3 = (self.x - other.x, self.y + self.h - other.y - other.h)
        p4 = (self.x + self.w - other.x - other.w, self.y - other.y)

        return max(p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1])

    @property
    def area(self):
        return self.w * self.h

    def __repr__(self):
        return f"Rect({self.x}, {self.y}, {self.w}, {self.h})"

    # eq check int value
    def __eq__(self, other):
        return self.val == other.val

    def to_json(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def move_hand():
    ard = serial.Serial('com8' ,9600)
    time.sleep(2)
    var = 'a'
    c=var.encode()
    ard.write(c)
    time.sleep(4)

def say_name(name):
    speaker.speak(f"হাই {name}")    


class Face:
    def __init__(self, img, bbox: Rect, name: str = "Unknown"):
        self.img = img
        self.bbox = bbox
        self.name = name
        self.confidence = 0
        self.encoding = None
        self.process = None
        self.processing = False
        self.loading = 0  # percentage of loading
        self.loaded = False
        self.clear_img = True
        self.id = str(uuid.uuid4())
        self.clock = time.time()
        self.interval = 1.5
        self.screentime = 0
        self.fresh_face = None
        self.clock2 = time.time()

    def update(self, img, bbox: Rect):

        e = time.time() - self.clock
        if e > self.interval:
            if not self.clear_img and abs(self.bbox.val - bbox.val) > 0.74:
                self.recognize()
            self.clock = time.time()
            # print("diff", abs(self.bbox.val - bbox.val))

        if self.loading < 101:
            self.loading += 1

        if self.screentime > 40:
            if "Load" in self.name:
                self.recognize()

        self.img = img
        self.bbox = bbox
        self.screentime += 1

    def recognize(self):
        if not self.processing:
            self.process = kthread.KThread(target=self.recognize_process)
            self.process.daemon = True
            self.process.start()

    def terminate_process(self):
        # check if process is running
        if self.process:
            if self.process.is_alive():
                try:
                    self.process.terminate()
                except RuntimeError:
                    pass

    def recognize_process(self, zoomout=0.25):
        # process with server
        self.processing = True
        print("Recognizing...")

        file_path = os.path.join(temp_dir, random_file_name())
        cv2.imwrite(file_path, self.img)
        bbox_json = self.bbox.to_json()
        data = {'img': file_path, 'bbox': bbox_json, 'zoomout': zoomout}

        r = requests.post('http://localhost:5000/recognize', json=data)
        if r.status_code == 200:
            self.loaded = True
            self.processing = False
            result = r.json()
            name = result['name']
            confidence = result['confidence']
            encoding_exists = result['encoding_exists']
            encoding = result['encoding']
            face = result['face']  # face path
            if encoding_exists:
                encoding = np.array(encoding)
                self.encoding = encoding
                if confidence < 50:
                    self.name = "Unknown"
                    self.confidence = confidence
                    self.clear_img = False
                else:
                    self.confidence = confidence
                    self.name = name
                    self.clear_img = True
                    self.fresh_face = cv2.imread(face)
                    

                    if self.name != "Unknown":
                        t = kthread.KThread(target=move_hand)
                        t.start()
                        t2 = kthread.KThread(target=say_name, args=(self.name,))
                        t2.start()

            else:
                self.name = "Unknown"
                self.confidence = 0
                self.clear_img = False
        else:
            self.name = "Unknown"
            self.confidence = 0
            self.loaded = True
            self.processing = False
            self.clear_img = False

        # delete temp file
        os.remove(file_path)
        os.remove(face)
            
        
        print("Recognized:", self.name, self.confidence, self.clear_img)

    # def save_face(self,name):
    #     path = os.path.join(img_dir, name) + ".jpg"
    #     self.fresh_face = cv2.cvtColor(self.fresh_face, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(path, self.fresh_face)
    #     known_faces.append(self.encoding)
    #     known_names.append(name)

    #     # save_known_faces()
    #     t = kthread.KThread(target=save_known_faces)
    #     t.daemon = True
    #     t.start()

    #     self.name = name
    #     self.confidence = 100

    def save_face(self, name):
        path = os.path.join(img_dir, name) + ".jpg"
        # self.fresh_face = cv2.cvtColor(self.fresh_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, self.fresh_face)

        data = {'encoding': self.encoding.tolist(), 'name': name}
        t = kthread.KThread(target=requests.post, args=(
            'http://localhost:5000/add_face',), kwargs={'json': data})
        t.daemon = True
        t.start()
        
        self.name = name
        self.confidence = 100
        

    def draw_face(self, img):
        # fancy draw outer box and add name and confidence or loading percentage bottom of the box

        # BOX
        cv2.rectangle(img, (self.bbox.x, self.bbox.y),
                      (self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h), (0, 255, 0), 2)

        # lINES
        cv2.line(img, (self.bbox.x, self.bbox.y),
                 (self.bbox.x + self.bbox.w, self.bbox.y), (0, 255, 0), 2)
        cv2.line(img, (self.bbox.x, self.bbox.y), (self.bbox.x,
                 self.bbox.y + self.bbox.h), (0, 255, 0), 2)
        cv2.line(img, (self.bbox.x + self.bbox.w, self.bbox.y),
                 (self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h), (0, 255, 0), 2)
        cv2.line(img, (self.bbox.x, self.bbox.y + self.bbox.h),
                 (self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h), (0, 255, 0), 2)

        name = self.name
        if self.loading < 99:
            name = "Loading..."
        if "Load" not in self.name:
            name = self.name
            self.loading += 10

        # NAME
        # Background box
        text = name
        if self.loaded and self.loading > 99:
            text += " "+str(int(self.confidence)) + "%"

        cv2.rectangle(img, (self.bbox.x, self.bbox.y - 20), (self.bbox.x +
                      len(text) * 10, self.bbox.y), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, text, (self.bbox.x, self.bbox.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # CONFIDENCE
        if self.loaded and self.loading > 99:
            pass
        else:
            # LOADING
            cv2.rectangle(img, (self.bbox.x, self.bbox.y + self.bbox.h + 20),
                          (self.bbox.x + self.bbox.w, self.bbox.y + self.bbox.h + 30), (76, 187, 23), 2)

            cv2.rectangle(img, (self.bbox.x, self.bbox.y + self.bbox.h + 20), (self.bbox.x + int(
                self.bbox.w * self.loading / 100), self.bbox.y + self.bbox.h + 30), (76, 187, 23), cv2.FILLED)
            # cv2.putText(img, str(self.loading) + "%", (self.bbox.x, self.bbox.y +
            #             self.bbox.h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)

        # show clock2 time at bottom of the rectangle
        time_screen = time.time() - self.clock2
        time_screen = str(datetime.timedelta(seconds=time_screen))
        # only minutes and seconds 00:00
        time_screen = time_screen[2:7]
        # if loading then pos is down of the loading bar
        if self.loading < 100:
            # down
            pos = (self.bbox.x+self.bbox.w-len(time_screen)
                   * 7, self.bbox.y + self.bbox.h + 50)
        else:
            # up
            pos = (self.bbox.x, self.bbox.y + self.bbox.h + 20)
        cv2.putText(img, time_screen, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


class FaceDetector:
    def __init__(self):
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpdraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            min_detection_confidence=0.5)
        self.faces = []
        self.results = None
        # self.face_detector_queue = FaceDetectorQueue()
        # self.face_detector_queue.start()

    def detect(self, img) -> list[Rect]:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = Rect(int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                            int(bboxC.width * iw), int(bboxC.height * ih))
                bboxs.append(bbox)
        return bboxs

    def fancy_draw(self, img):
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = Rect(int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                            int(bboxC.width * iw), int(bboxC.height * ih))
                cv2.rectangle(img, (bbox.x, bbox.y),
                              (bbox.x+bbox.w, bbox.y+bbox.h), (255, 0, 255), 2)
                cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox.x, bbox.y-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img



def main():
    detector = FaceDetector()
    cap = cv2.VideoCapture(1)
    # player = MediaPlayer(file)
    faces: list[Face] = []
    max_diff = 10
    cid = 1
    i = -1
    ptime = time.time()
    emptyTime = None
    while True:
        i += 1
        success, img = cap.read()


        copy_img = img.copy()
        bboxs = detector.detect(img)


        # print("BBOXS", len(bboxs), "FACES", len(
        #     faces)) if i % 100 == 0 else None

        if faces:
            face_rect_vals = np.array([face.bbox.val for face in faces])
            keep_faces: int = []  # index of faces to keep
            for bbox in bboxs:
                diff = face_rect_vals - bbox.val
                # diff2 = [face.bbox.diff(bbox) for face in faces]
                # print(min(diff2))

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

                if min_diff <= max_diff:
                    faces[min_idx].update(copy_img, bbox)
                    keep_faces.append(min_idx)

                else:
                    n = len(faces)
                    keep_faces.append(n)
                    faces.append(Face(copy_img, bbox, f"Loading {cid}"))
                    cid += 1

            new_faces = []
            for idx, face in enumerate(faces):
                if idx in keep_faces:
                    new_faces.append(face)
                else:

                    face.terminate_process()
            faces = new_faces
        else:
            for bbox in bboxs:
                faces.append(Face(copy_img, bbox, f"Loading {cid}"))
                cid += 1

        if not bboxs:
            emptyTime = time.time() if emptyTime is None else emptyTime
            diff = time.time() - emptyTime
            if diff >= 4:
                faces = []
                emptyTime = None
                cid = 1

        for face in faces:
            face.draw_face(img)

            if face.name == "Unknown" and face.clear_img:
                speaker.speak("কে আপনি? আপনার নাম বলুন")
                print("Unknown face detected")

                r = sr.Recognizer()

                with sr.Microphone() as source:
                    
                    name = r.listen(source)
                    print("সময় শেষ,ধন্যবাদ")


                    try:
                    
                        print("Name : "+r.recognize_google(name))
                        face.save_face(name)
                    except:
                        print("দুঃক্ষিত,আপনার নামটি আমি শুনতে পাই নি")
                                    


        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        # if val != 'eof' and audio_frame is not None:
        #     img, t = audio_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()
