from pydub import AudioSegment
from pydub.playback import play
import os
import argparse


def play_file(path,ex='wav'):
    if os.path.exists(path):
        sound = AudioSegment.from_file(path)
        play(sound)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path',help='file path')
    parser.add_argument('-e','--ex',help='file extension',default='wav')
    args = parser.parse_args()
    play_file(args.path,args.ex)