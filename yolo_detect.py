import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Définir et analyser les arguments fournis par l'utilisateur
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Chemin vers le fichier du modèle YOLO (exemple : "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Source d’image : fichier image ("test.jpg"), dossier ("test_dir"), fichier vidéo ("testvid.mp4"), index de caméra USB ("usb0"), ou index de Picamera ("picamera0")', required=True)
parser.add_argument('--thresh', help='Seuil minimal de confiance pour afficher les objets détectés (exemple : "0.4")', default=0.5)
parser.add_argument('--resolution', help='Résolution d’affichage au format LxH (exemple : "640x480"), sinon la résolution source est utilisée', default=None)
parser.add_argument('--record', help='Enregistrer les résultats d’une vidéo ou webcam sous le nom "demo1.avi". Nécessite l’argument --resolution.', action='store_true')

args = parser.parse_args()

# Analyse des arguments
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Vérification du modèle
if not os.path.exists(model_path):
    print('ERREUR : Le chemin du modèle est invalide ou le fichier est introuvable.')
    sys.exit(0)

# Charger le modèle YOLO
model = YOLO(model_path, task='detect')
labels = model.names

# Déterminer le type de source (image, dossier, vidéo, caméra USB, Picamera)
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP','.webp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Extension de fichier {ext} non prise en charge.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Source {img_source} invalide.')
    sys.exit(0)

# Vérifier et appliquer la résolution utilisateur
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Préparer l’enregistrement si demandé
if record:
    if source_type not in ['video','usb']:
        print('L’enregistrement fonctionne uniquement avec une vidéo ou une caméra.')
        sys.exit(0)
    if not user_res:
        print('Veuillez spécifier une résolution pour l’enregistrement.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Charger la source (image, dossier, vidéo ou caméra)
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Couleurs des boîtes englobantes (palette Tableau 10)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Variables de contrôle
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Boucle principale d’inférence
while True:
    t_start = time.perf_counter()

    # Charger une image ou une trame selon la source
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Toutes les images ont été traitées.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Fin du fichier vidéo.')
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Impossible de lire les images de la caméra USB.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Impossible de lire les images de la Picamera.')
            break

    # Redimensionner si nécessaire
    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # Lancer l’inférence
    results = model(frame, verbose=False)

    # Extraire les résultats
    detections = results[0].boxes
    object_count = 0

    # Traiter chaque détection
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        # Dessiner la boîte si la confiance dépasse le seuil
        if conf > 0.5:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # Afficher le nombre d’objets et la fréquence d’images
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.putText(frame, f'Nombre objets: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('Résultats de la détection YOLO', frame)
    if record:
        recorder.write(frame)

    # Gérer les touches clavier
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png',frame)

    # Calculer la fréquence d’images (FPS)
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Nettoyage final
print(f'Fréquence moyenne du pipeline : {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
