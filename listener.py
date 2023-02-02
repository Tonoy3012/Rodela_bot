from typing import Dict, List
import speech_recognition as sr
from utils import logger, env, temp_dir, data_dir, cmd_parser,app_dir
import threading
import time
from recognizer import Recognizer
from multiprocessing import Process
import sys
from voice import Speaker
import os
import json
import random
import requests
import cv2
import pytesseract
import os
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = os.path.join(app_dir,'ocr','tesseract.exe')

# check if microphone is available
def check_microphone():
    if not sr.Microphone.list_microphone_names():
        logger.error('No microphone found')
        env.set('microphone', False)
        return False
    return True


class Listener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # don't recognize anything below 1000
        self.recognizer.energy_threshold = 1000

        self.voice_recognizer = Recognizer()
        self.recognizer.energy_threshold = 1000

        self.audios: List[sr.AudioData] = []
        self.texts: Dict[int, str] = {}

        self.listening = True
        self.processing = True

        # listen in a thread, process in another
        logger.info('Starting listener')
        self.listen_thread = self.recognizer.listen_in_background(
            self.microphone, self.add_audio, phrase_time_limit=5)
        logger.info('Listining...')

        # logger.info('Starting processor')
        # self.process_thread = threading.Thread(target=self.process)
        # self.process_thread.daemon = True
        # self.process_thread.start()

        self.index = 0
        self.total_processes = 0

    def add_audio(self, _, audio: sr.AudioData):
        self.single_recognize(self.index, audio)
        self.index += 1

    def listen(self):
        while self.listening:
            logger.debug('Listening...')
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source)
                self.audios.append(audio)
                logger.debug(f'Audio {self.index} recorded')
                self.index += 1
                time.sleep(0.1)

    def single_recognize(self, idx, audio: sr.AudioData) -> None:
        audio_path = self.voice_recognizer.convert_to_wav(audio, idx)
        t = threading.Thread(target=self.voice_recognizer.recognize, args=(
            audio_path, idx, self.texts))
        t.start()

        # remove audio file
        # self.voice_recognizer.remove_audio(audio_path)

    # def process(self):
    #     while self.processing:
    #         if self.audios:
    #             audio = self.audios.pop(0)
    #             print("Recognizing audio", self.index)
    #             self.single_recognize(self.index, audio)
    #         time.sleep(0.1)

    def get_text(self):
        if self.texts:
            soreted_keys = sorted(self.texts.keys())
            text = self.texts[soreted_keys[0]]
            del self.texts[soreted_keys[0]]
            return text

    def stop(self):
        self.listening = False
        self.processing = False
        # self.listen_thread.join()
        # self.process_thread.join()


def get_current_faces():
    end_point = "http://localhost:5000/current_faces"
    data = requests.get(end_point)
    return data.json()['current_faces']

def save_face_name(name,face_id):
    end_point = "http://localhost:5000/save_faces"
    data = {
        'save_faces': [(name,face_id)]
    }
    return requests.post(end_point,data)

def capture_img():
    vid = cv2.VideoCapture(1)
    ret , image = vid.read()
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    file_name = 'img_cap.jpg'
    path = os.path.join(temp_dir,file_name)
    cv2.imwrite(path,image)
    return path

def recognize_text_img(file_name,speaker:Speaker):
    img = Image.open(file_name)
    text = pytesseract.image_to_string(img,lang= 'bn')
    if text:
        speaker.speak(text)
    else:
        speaker.speak("বই এর লিখা অস্পষ্ট")
    

if __name__ == '__main__':
    if check_microphone():
        listener = Listener()
        speaker = Speaker()

        joke_path = os.path.join(data_dir, 'jokes.json')
        with open(joke_path, 'r', encoding='utf-8') as f:
            jokes = json.load(f)['joke']

        jkeys = list(jokes.keys())

        person_path = os.path.join(data_dir, 'person.json')
        with open(person_path, 'r', encoding='utf-8') as f:
            person = json.load(f)['info']

        names = list(person.keys())
        names = [name.strip() for name in names]

        persons_info = ["সারমর্ম","বিস্তারিত"]

        song_names_en = ["Aguner Poroshmoni", "Tui Phele Esechis Kare"]
        song_names = ["রবীন্দ্র সঙ্গীত", "রবীন্দ্র"]

        license_path = os.path.join(data_dir, 'licence.json')
        with open(license_path, 'r', encoding='utf-8') as f:
            license = json.load(f)['licence']

        keys = list(license.keys())
        license_keys = ["লাইসেন্স"]
        
        keys = [key.strip() for key in keys]

        news = license_keys + names
        # print(news)

        news = [name.strip() for name in news]
        
        cmd_parser.cmd_dict['News'].extend(news)
        cmd_parser.cmd_dict['Joke'].extend(jkeys)
        cmd_parser.cmd_dict['Play'].extend(song_names)

        wait_for_name = False
        wait_id = None
        faceid = None
        facename = None
        try:
            while True:
                data = listener.get_text()
                if data:
                    txt, cmd, path, widx = data
                    
                    if not txt:
                        if wait_for_name and wait_id == widx:
                            wait_id += 1
                        continue
                    
                    if txt:
                        print(txt)
                        
                        if wait_for_name and wait_id == widx:
                            pass
                            
                    if not cmd:
                        continue

                    if cmd == "Joke":
                        key = random.choice(jkeys)
                        joke = jokes[key]
                        save_key = f"{key}_joke"
                        speaker.speak(joke, save=True, name=save_key)

                    if cmd == "Play":
                        # name = cmd_parser.parse_name(txt, song_names)
                        # idx = song_names.index(name)
                        # en_name = song_names_en[idx]
                        en_name = random.choice(song_names_en)

                        logger.debug("Playing", en_name)
                        path = os.path.join(data_dir, en_name) + '.mp3'
                        t = speaker.speak_file(path, 'mp3')
                        # t.join()

                    if cmd == "News":
                        key = cmd_parser(txt, news)
                        logger.debug("News", key)

                        if key in names:
                            save_key = f"{key}_info"

                            speaker.speak(
                                person[key], save=True, name=save_key)

                        elif key in license_keys:
                            lkey = cmd_parser(txt, keys)
                            save_key = f"{lkey}_licence"
                            speaker.speak(
                                license[lkey], save=True, name=save_key)
                    
                    elif cmd == "Book":
                        speaker.speak("বইটি ক্যামেরার সামনে সঠিক ভাবে ধরুন")
                        try:
                            file_path = capture_img()
                            recognize_text_img(file_path,speaker)
                        except Exception as e:
                            logger.critical("Error reading image",e)



                    if cmd == "Stop":
                        if speaker.speaking:
                            speaker.stop()

                time.sleep(0.1)
        except KeyboardInterrupt:
            listener.stop()
            sys.exit()

        except Exception as e:
            logger.error(e)
            # listener.stop()
            # sys.exit()
