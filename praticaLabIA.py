import os
import numpy as np
import cv2

path = '/tmp/guest-apin3n/base'

num_treino = 0
num_teste = 0

#numero de arquivos no folder de treino e no folder de teste
for directory in os.listdir(path):
	if directory == 'treino':
		for dir_treino in os.listdir(path + '/' + directory):
			num_treino = num_treino + len(os.listdir(path + '/' + directory + '/' + dir_treino))
	else:
		num_teste = num_teste + len(os.listdir(path + '/' + directory))

treino = np.empty([num_treino,64,64,3], dtype=np.uint8)
teste = np.empty([num_teste,64,64,3], dtype=np.uint8)
labels = np.empty([num_treino], dtype = np.uint8)
validation = np.empty([num_treino/4], dtype = np.uint8)

#preenchendo os vetores de treino e de teste
for directory in os.listdir(path):
	if directory == 'treino':
		k = 0
		l = 0
		for dir_treino in os.listdir(path + '/' + directory):
			for img in os.listdir(path + '/' + directory + '/' + dir_treino):

				classe = 0
				if dir_treino == 'sinuca':
					classe = 0
				elif dir_treino == 'futebol':
					classe = 1
				elif dir_treino == 'tenis':
					classe = 2
				elif dir_treino == 'golfe':
					classe = 3					
					
				img = cv2.imread(path + '/' + directory + '/' + dir_treino + '/' + img,cv2.IMREAD_COLOR)
				treino[k] = img
				labels[l] = classe
				k = k + 1
				l = l + 1
	else:
		k = 0
		for img in os.listdir(path + '/' + directory):
			img = cv2.imread(path + '/' + directory + '/' + img,cv2.IMREAD_COLOR)
			teste[k] = img
			k = k + 1




