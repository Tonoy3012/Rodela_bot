import torch
from aksharamukha import transliterate
import soundfile as sf
from multiprocessing import Process
from utils import logger, temp_dir, app_dir, data_dir
from uuid import uuid1
import os
from kthread import KThread
from pydub import AudioSegment
from pydub.playback import play


logger.debug('Loading tts model')
tts_model_path = os.path.join(app_dir, 'tts', 'tts_model')

model, _ = torch.hub.load(repo_or_dir=tts_model_path,
                          model='silero_tts',
                          language='indic',
                          speaker='v3_indic',
                          source='local')
logger.debug('Model loaded')


class Speaker:
    def __init__(self, gender: str = 'female'):
        self.sample_rate = 44100

        self.gender = 'bengali_female'
        if gender[0] == 'm':
            self.gender = 'bengali_male'

        self.speaking = False
        self.speak_thread = None
        self.speak_process = None
        self.file_path = None

        self.stop_tiggred = False
        self.idx = 1

    def __prepare_text(self, text: str) -> str:
        return transliterate.process('Bengali', 'ISO', text)  # type: ignore

    def __get_wav_file(self, tensor: torch.Tensor) -> str:
        file_path = os.path.join(temp_dir, f"{self.idx}.wav")
        sf.write(file_path, tensor.numpy(), self.sample_rate)
        self.idx += 1
        return file_path

    def play_file(self, path, ex='wav'):
        if ex == 'mp3':
            ad = AudioSegment.from_mp3(path)
        else:
            ad = AudioSegment.from_wav(path)
        play(ad)
        self.speaking = False

    def __convert_to_audio_tensor(self, text: str):
        text = self.__prepare_text(text)
        audio = model.apply_tts(text, speaker=self.gender)
        return audio

    def __speak(self, text: str, save=False, file_name=None) -> None:
        if save:
            file_path = os.path.join(data_dir, f"{file_name}.wav")
            if os.path.exists(os.path.join(data_dir, f"{file_name}.wav")):
                self.play_file(file_path)
                return logger.debug('File already exists')

        audio_tensor = self.__convert_to_audio_tensor(text)
        audio_file = self.__get_wav_file(audio_tensor)
        self.file_path = audio_file

        self.speaking = True
        self.play_file(audio_file)

        if save:
            file_path = os.path.join(data_dir, f"{file_name}.wav")
            if not os.path.exists(file_path):
                os.rename(audio_file, file_path)
                
        if os.path.exists(audio_file):
            os.remove(audio_file)

    def _speak(self, text: str, save=False, file_name=None) -> KThread:  # only len 1000 char
        thread = KThread(target=self.__speak, args=(text, save, file_name))
        thread.start()
        self.speak_thread = thread
        return thread

    def speak(self, text: str, save=False, name=None):
        if self.speaking:
            return logger.debug('Already speaking')

        if len(text) < 500:
            return self._speak(text, save, name)
        else:
            self.stop_tiggred = False
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            for chunk in chunks:
                if self.stop_tiggred:
                    self.stop_tiggred = False
                    if self.file_path:
                        if os.path.exists(self.file_path):
                            os.remove(self.file_path)
                    break
                t = self._speak(chunk)
                t.join()

    def _speak_file(self, file_path, ex='wav'):
        return self.play_file(file_path, ex)

    def speak_file(self, file_path, ex='wav'):
        if self.speaking:
            return logger.debug('Already speaking')
        
        self.speaking = True
        t = KThread(target=self._speak_file, args=(file_path, ex))
        t.start()
        self.speak_thread = t
        return t

    def stop(self) -> None:
        if self.speak_thread:
            self.stop_tiggred = True
            self.speak_thread.kill()
            self.speak_thread = None
            self.speaking = False
            if self.file_path:
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)


if __name__ == '__main__':
    speaker = Speaker()
    # print("Speak")
    text = """আমি এখনই জানতে চাই!"""

    speaker.speak(text)
    # speaker.speak('আমি এখনই জানতে চাই!')
