import os
import numpy as np
import cv2

def processa_base(path,heigth,width,dimension,size_of_validation):

	num_treino = 0
	num_teste = 0

	#numero de arquivos no folder de treino e no folder de teste
	for directory in os.listdir(path):
		if directory == 'treino':
			for dir_treino in os.listdir(path + '/' + directory):
				num_treino = num_treino + len(os.listdir(path + '/' + directory + '/' + dir_treino))
		else:
			num_teste = num_teste + len(os.listdir(path + '/' + directory))

	#criando os vetores
	treino = np.empty([num_treino,heigth,width,dimension], dtype=np.uint8)
	teste = np.empty([num_teste,heigth,width,dimension], dtype=np.uint8)
	labels_treino = np.empty([num_treino], dtype = np.uint8)

	tam_validacao = int(size_of_validation*num_treino)
	tam_treino_oficial = num_treino - tam_validacao

	validacao = np.empty([tam_validacao,heigth,width,dimension], dtype=np.uint8)
	labels_validacao = np.empty([tam_validacao],dtype=np.uint8)

	treino_oficial = np.empty([tam_treino_oficial,heigth,width,dimension], dtype=np.uint8)
	labels_treino_oficial = np.empty([tam_treino_oficial], dtype = np.uint8)

	tam_classes = []
	
	#preenchendo os vetores de treino e de teste
	for directory in os.listdir(path):
		if directory == 'treino':
			k = 0
			l = 0
			for classe,dir_treino in enumerate(os.listdir(path + '/' + directory)):
				lista_imagens_classe = os.listdir(path + '/' + directory + '/' + dir_treino)
				tam_classes.append(len(lista_imagens_classe))
				for img in lista_imagens_classe:
	
					imagem = cv2.imread(path + '/' + directory + '/' + dir_treino + '/' + img,cv2.IMREAD_COLOR)
					treino[k] = imagem
					labels_treino[l] = classe
					k = k + 1
					l = l + 1
		else:
			k = 0
			for img in os.listdir(path + '/' + directory):
				imagem = cv2.imread(path + '/' + directory + '/' + img,cv2.IMREAD_COLOR)
				teste[k] = imagem
				k = k + 1

	#aplicando os pesos para cada classe
	tam_classes = map(lambda i: int(i*size_of_validation),tam_classes)
	
	#construindo validacao
	classe_atual = 0
	tam = 0
	classe_de_insercao = 0
	cont = 0
	v = 0
	j = 0
	for k in range(len(treino)):
		if labels_treino[k] != classe_atual:
			classe_atual = classe_atual + 1
		if cont < len(tam_classes) and tam > tam_classes[cont] - 1:
			classe_de_insercao = classe_de_insercao + 1
			cont = cont + 1	
			tam = 0
		if classe_atual == classe_de_insercao:
			validacao[v] = treino[k]
			labels_validacao[v] = labels_treino[k]
			v = v + 1
			tam = tam + 1
		else:
			treino_oficial[j] = treino[k]
			labels_treino_oficial[j] = labels_treino[k]
			j = j + 1 

path = '/home/victor/base'
bla = processa_base(path,64,64,3,0.2)