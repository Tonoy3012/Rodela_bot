from flask import Flask, request, jsonify
import face_recognition as fr
import cv2
import numpy as np
import os
import multiprocessing as mp
import pickle
import random
from mimetypes import guess_type
from utils import app_dir, temp_dir

current_file_path = os.path.dirname(os.path.abspath(__file__))
# imgs\known_faces_encoding.pickle
# imgs\known_faces_names.pickle
known_faces_path =  os.path.join(app_dir, 'imgs', 'known_faces_encoding.pickle')
known_names_path =  os.path.join(app_dir, 'imgs', 'known_faces_names.pickle')
temp_img_dir = temp_dir
img_dir = os.path.join(app_dir, 'imgs')

if not os.path.exists(temp_img_dir):
    os.makedirs(temp_img_dir)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)


def load_known_faces():
    with open(known_faces_path, 'rb') as f:
        known_faces = pickle.load(f)
    with open(known_names_path, 'rb') as f:
        known_names = pickle.load(f)
    return known_faces, known_names

# if not os.path.exists(encoding_file) or not os.path.exists(names_file):
#         for file in os.listdir(img_dir):
#             if "image" in guess_type(file)[0]:
#                 img = fr.load_image_file(os.path.join(img_dir, file))
#                 encoding = fr.face_encodings(img)
#                 if encoding:
#                     encoding = encoding[0]
#                     name = file.split(".")[0]
#                     known_faces.append(encoding)
#                     known_names.append(name)

#         with open(encoding_file, "wb") as f:
#             pickle.dump(known_faces, f)

#         with open(names_file, "wb") as f:
#             pickle.dump(known_names, f)
#     else:
#         with open(encoding_file, "rb") as f:
#             known_faces2 = pickle.load(f)
#         with open(names_file, "rb") as f:
#             known_names2 = pickle.load(f)

#         known_names.extend(known_names2)
#         known_faces.extend(known_faces2)

if os.path.exists(known_faces_path) and os.path.exists(known_names_path):
    known_faces, known_names = load_known_faces()
else:
    known_faces = []
    known_names = []
    
    for file in os.listdir(img_dir):
        if "image" in guess_type(file)[0]:
            img = fr.load_image_file(os.path.join(img_dir, file))
            encoding = fr.face_encodings(img)
            if encoding:
                encoding = encoding[0]
                name = file.split(".")[0]
                known_faces.append(encoding)
                known_names.append(name)
    
    with open(known_faces_path, 'wb') as f:
        pickle.dump(known_faces, f)
    
    with open(known_names_path, 'wb') as f:
        pickle.dump(known_names, f)

    
print('Loaded known faces')
print('Known faces:', len(known_faces))
print('Known names:', *known_names, sep=' | ')
print('---------------------')

def save_known_faces():
    with open(known_faces_path, 'wb') as f:
        pickle.dump(known_faces, f)
    with open(known_names_path, 'wb') as f:
        pickle.dump(known_names, f)

def random_string(length=8):
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(letters) for i in range(length))

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize():
    # get img path from request
    data = request.get_json()
    img = data['img'] # img path
    bbox = data['bbox']
    zoomout = data['zoomout']
    
    img = cv2.imread(img)
    bbox = data['bbox']
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

    # load image with zoomout
    x = x - int(w * zoomout)
    y = y - int(h * zoomout)
    w = w + int(w * zoomout * 2)
    h = h + int(h * zoomout * 2)

    # check if x, y, w, h are in range
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > img.shape[1]:
        w = img.shape[1] - x
    if y + h > img.shape[0]:
        h = img.shape[0] - y
    
    img = img[y:y+h, x:x+w]
    # check if image is empty
    if img.size == 0:
        return jsonify({'name': 'Unknown', 'confidence': 0, 'encoding_exists': False, 'encoding': [],'face': ''})
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_path = os.path.join(temp_img_dir, random_string() + '.jpg')
    cv2.imwrite(img_path, img)

    # load encodings
    encodings = fr.face_encodings(img)
    data = {'name': 'Unknown', 'confidence': 0, 'encoding_exists': False, 'encoding': [],'face': img_path}
    if encodings:
        encoding = encodings[0]
        matched = fr.compare_faces(known_faces, encoding)
        face_distances = fr.face_distance(known_faces, encoding)
        face_matched = False
        if face_distances.size > 0:
            best_match_index = np.argmin(face_distances)
            face_matched = matched[best_match_index]
        
        if face_matched:
            name = known_names[best_match_index]
            confidence = (1 - face_distances[best_match_index]) * 100
            if confidence < 50:
                data['name'] = 'Unknown'
                data['confidence'] = confidence
                data['encoding_exists'] = True
                data['encoding'] = encoding.tolist()
            else:
                data['name'] = name
                data['confidence'] = confidence
                data['encoding_exists'] = True
                data['encoding'] = encoding.tolist()
        else:
            data['name'] = 'Unknown'
            data['confidence'] = 100
            data['encoding_exists'] = True
            data['encoding'] = encoding.tolist()
    else:
        data['name'] = 'Unknown'
        data['confidence'] = 0
        data['encoding_exists'] = False
        data['encoding'] = []
    
    print('Recognized:' f"{data['name']} {data['confidence']} % Encoding exists: {data['encoding_exists']}")
    print('---------------------')
    return jsonify(data)


@app.route('/add_face', methods=['POST'])
def add_face():
    data = request.get_json()
    encoding = data['encoding']
    encoding = np.array(encoding)
    name = data['name']
    
    print('Adding face:', name)
    
    known_faces.append(encoding)
    known_names.append(name)
    save_known_faces()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run()