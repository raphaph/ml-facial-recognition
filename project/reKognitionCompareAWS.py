import cv2
import os
import boto3
import threading
import functools

# Configuração do cliente Rekognition
session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)

rekognition_client = session.client('rekognition')

faces = os.listdir('project/personsRekognition')
face_cache = {}

for face in faces:
    face_name = face.replace('.jpg', '')
    face_image_path = f'project/personsRekognition/{face_name}.jpg'
    face_image_content = open(face_image_path, 'rb').read()
    face_cache[face_name] = face_image_content

video_captured = cv2.VideoCapture(0)
print('Recognition started!')

@functools.lru_cache(maxsize=128)
def read_image(path):
    return cv2.imread(path)


def detect_face(user_image, frame, name):
    try:
        matches = rekognition_client.compare_faces(
            SourceImage={'Bytes': user_image},
            TargetImage={'Bytes': cv2.imencode('.jpg', frame)[1].tobytes()},
            SimilarityThreshold=80
        )

    except rekognition_client.exceptions.InvalidParameterException:
        # Trata exceção gerada quando não há rosto na imagem
        matches = {'FaceMatches': []}

    # Se uma correspondência for encontrada, definir o nome correspondente
    if len(matches['FaceMatches']) > 0:

        # Coleta dados do match encontrado para desenhar o retângulo e nome no frame
        bounding_box = matches['FaceMatches'][0]['Face']['BoundingBox']
        x = int(bounding_box['Left'] * frame.shape[1])
        y = int(bounding_box['Top'] * frame.shape[0])
        w = int(bounding_box['Width'] * frame.shape[1])
        h = int(bounding_box['Height'] * frame.shape[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        print(f'Recognized face: {name}')
        # print(matches['FaceMatches'])

while True:
    ret, frame = video_captured.read()
    if not ret:
        continue

    found_faces = [] # Lista que armazena os rostos encontrados na imagem
    threads = []

    # top = x right = y bottom = w left = h
    for face in faces:
        if face.endswith('.jpg'):
            face_name = face.replace('.jpg', '')
            user_image = face_cache[face_name]
            image = read_image(f'project/personsRekognition/{face_name}.jpg')
            thread = threading.Thread(target=detect_face, args=(user_image, frame, face_name))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_captured.release()
cv2.destroyAllWindows()
print('Recognition stopped!')