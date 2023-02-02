from multiprocessing import Process
import time
import os
import subprocess
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
python_env = os.path.join(current_dir, 'rodela', 'Scripts', 'python.exe')
print(python_env)

files = [
    "face_reco_server.py",
    "face_recognizer.py",
    "listener.py",
]


def start_process(file):
    print("Starting", file)
    subprocess.Popen([python_env, file], cwd=current_dir)


if __name__ == "__main__":

    processes = []
    for file in files:
        p = Process(target=start_process, args=(file,))
        processes.append(p)

    for p in processes:
        p.start()
        time.sleep(1)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program.")
        for p in processes:
            p.terminate()
