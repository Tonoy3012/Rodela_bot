import torch
import os
from uuid import uuid4
from utils import temp_dir, env, app_dir
from speech_recognition import AudioData
from typing import Dict, List, Tuple, Union
from utils import logger, data_store_path, text_store_path
from threading import Thread
import json
from bnbphoneticparser import BengaliToBanglish
from transformers import pipeline
from utils import cmd_parser

logger.info("Initializing recognizer")

vad_loc = os.path.join(app_dir, 'vad', 'vad_model')

# Loading vad model
logger.debug('Loading voice activity detection model')
model, utils = torch.hub.load(repo_or_dir=vad_loc,
                              model='silero_vad', source='local')

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

logger.debug('Model loaded')
# VAD model loaded

logger.debug('Loading asr model')
pipe = pipeline("automatic-speech-recognition",
                model=os.path.join(app_dir, 'asr'))
logger.debug('Model loaded')


def transcribe(audio_path: str) -> str:
    text = pipe(audio_path)["text"]  # type: ignore
    return text


b2b = BengaliToBanglish()


def bangla_to_banglish(text: str) -> str:
    return b2b.parse(text)


class Recognizer:
    def __init__(self):
        self.sample_rate = 16000


    def voice_activity_detection(self, audio_path: str) -> Tuple[str, bool]:
        wav = read_audio(audio_path, sampling_rate=self.sample_rate)
        speech_timestamps = get_speech_timestamps(
            wav, model, sampling_rate=self.sample_rate)

        voice_activity = False
        if speech_timestamps:
            save_audio(audio_path, collect_chunks(
                speech_timestamps, wav), sampling_rate=self.sample_rate)
            voice_activity = True

        return audio_path, voice_activity

    def convert_to_wav(self, audio: AudioData, idx: int) -> str:
        # save audio to temp file
        audio_path = os.path.join(temp_dir, f'{uuid4()}.wav')
        data = audio.get_wav_data()
        with open(audio_path, 'wb') as f:
            f.write(data)
        return audio_path


    def recognize(self, audio_path: str, idx: int, texts: Dict[int, str]) -> Union[str, None]:
        audio_path, voice_activity = self.voice_activity_detection(audio_path)
        if voice_activity:
            logger.info(f"Voice activity detected for {idx}")
            text = transcribe(audio_path)
            cmd = cmd_parser.parse(text)
            texts[idx] = [text, cmd, audio_path,idx]
            logger.info(f"Command: {cmd}")
            
            
        else:
            logger.info(f"No voice activity detected for {idx}")
        # os.remove(audio_path)


# if __name__ == '__main__':

    # data = JsonSave(data_store_path)
    # text_store = JsonSave(text_store_path)
    # max_processes = env.get('max_processes', 4)
    # total_processes = Value(0)
    # env.set('max_processes', max_processes)
    # while True:
    #     if os.path.exists(data_store_path):
    #         data.load()

    #     if data.get('audio_paths'):
    #         audio_paths = data.get('audio_paths')

    #         for idx, audio_path in audio_paths.items(): # type: ignore
    #             if not text_store.get(idx):
    #                 transcribe_and_add(audio_path, idx, text_store,total_processes)
    #                 break

    #     else:
    #         time.sleep(0.1)
