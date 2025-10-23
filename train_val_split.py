# Division du jeu de données en dossiers d'entraînement (train) et de validation (val)

from pathlib import Path
import random
import os
import sys
import shutil
import argparse

# Définir et analyser les arguments fournis par l'utilisateur
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Chemin du dossier contenant les images et les annotations', required=True)
parser.add_argument('--train_pct', help='Proportion d’images à placer dans le dossier train ; le reste ira dans validation (exemple : ".8")', default=.8)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

# Vérification des entrées fournies
if not os.path.isdir(data_path):
   print('Le dossier spécifié par --datapath est introuvable. Vérifiez le chemin et réessayez.')
   sys.exit(0)
if train_percent < .01 or train_percent > 0.99:
   print('Entrée invalide pour train_pct. Veuillez entrer un nombre entre .01 et .99.')
   sys.exit(0)
val_percent = 1 - train_percent

# Définir les chemins d'entrée du dataset (images et labels)
input_image_path = os.path.join(data_path,'images')
input_label_path = os.path.join(data_path,'labels')

# Définir les chemins des dossiers de sortie (train et validation)
cwd = os.getcwd()
train_img_path = os.path.join(cwd,'data/train/images')
train_txt_path = os.path.join(cwd,'data/train/labels')
val_img_path = os.path.join(cwd,'data/validation/images')
val_txt_path = os.path.join(cwd,'data/validation/labels')

# Créer les dossiers s'ils n'existent pas déjà
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'Dossier créé : {dir_path}')

# Obtenir la liste de toutes les images et annotations
img_file_list = [path for path in Path(input_image_path).rglob('*')]
txt_file_list = [path for path in Path(input_label_path).rglob('*')]

print(f'Nombre d’images : {len(img_file_list)}')
print(f'Nombre d’annotations : {len(txt_file_list)}')

# Déterminer le nombre de fichiers à déplacer dans chaque dossier
file_num = len(img_file_list)
train_num = int(file_num*train_percent)
val_num = file_num - train_num
print('Images déplacées vers train : %d' % train_num)
print('Images déplacées vers validation : %d' % val_num)

# Sélectionner aléatoirement les fichiers et les copier dans les dossiers correspondants
for i, set_num in enumerate([train_num, val_num]):
  for ii in range(set_num):
    img_path = random.choice(img_file_list)
    img_fn = img_path.name
    base_fn = img_path.stem
    txt_fn = base_fn + '.txt'
    txt_path = os.path.join(input_label_path,txt_fn)

    if i == 0: # Copier la première série de fichiers dans train
      new_img_path, new_txt_path = train_img_path, train_txt_path
    elif i == 1: # Copier la seconde série dans validation
      new_img_path, new_txt_path = val_img_path, val_txt_path

    shutil.copy(img_path, os.path.join(new_img_path,img_fn))
    # Si le fichier d'annotation existe, le copier aussi
    if os.path.exists(txt_path):
      shutil.copy(txt_path,os.path.join(new_txt_path,txt_fn))

    img_file_list.remove(img_path)
