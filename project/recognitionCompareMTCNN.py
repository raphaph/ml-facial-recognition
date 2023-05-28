import cv2
import os
import face_recognition
import threading
import tensorflow as tf
from mtcnn import MTCNN

save_folder = 'projeto/frameCaptured'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Carregar as imagens de pessoas definidas e extrair recursos faciais
def load_known_faces():
    known_face_encodings = []
    known_names = []
    for person_folder in os.listdir('projeto/persons'):
        if os.path.isdir(os.path.join('projeto/persons', person_folder)):
            for filename in os.listdir(os.path.join('projeto/persons', person_folder)):
                if filename.endswith('.jpg'):
                    image = face_recognition.load_image_file(os.path.join('projeto/persons', person_folder, filename))
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_names.append(person_folder)
                    
    print(f'peoples: {set(known_names)}')
    return known_face_encodings, known_names


known_face_encodings, known_names = load_known_faces()

detector = MTCNN()

print('faces e algoritmo carregado, iniciando captura de frames...')

video_captured = cv2.VideoCapture(0)

# Inicializa variáveis para cache de recursos faciais
# cache = {}
# lock = threading.Lock()

print('iniciando reconhecimento facial... camera ligada')

while True:
    ret, frame = video_captured.read()

    # Detectar rostos no quadro
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img)
    print(result)
    faces = [result[i]['box'] for i in range(len(result))]
    
    # Comparar o rosto detectado com as imagens de pessoas definidas
    face_encodings = face_recognition.face_encodings(frame, faces)
    
    # top = x right = y bottom = w left = h
    for (x, y, w, h), face_encoding in zip(faces, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.65)
        name = "Unknown"
        print(matches)

        # Se uma correspondência for encontrada, definir o nome correspondente
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Desenhar um retângulo no rosto detectado e escrever o nome correspondente
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (h, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # # Salvar o quadro se o rosto for identificado pela primeira vez
        if name != "Desconhecido":
            if not os.path.exists(os.path.join(save_folder, name)):
                os.makedirs(os.path.join(save_folder, name))
            cv2.imwrite(os.path.join(save_folder, name, f"{name}_{str(len(os.listdir(os.path.join(save_folder, name))))}.jpg"), frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_captured.release()
cv2.destroyAllWindows()
print('camera desligada!')
