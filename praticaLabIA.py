import os
import numpy as np
import cv2

def aplica_peso(n,peso):
	n = n * peso
	return int(n)


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
	num_exemplos_por_classe = []
	treino = np.empty([num_treino,heigth,width,dimension], dtype=np.uint8)
	teste = np.empty([num_teste,heigth,width,dimension], dtype=np.uint8)
	labels_treino = np.empty([num_treino], dtype = np.uint8)
	labels_validation = np.empty([int(num_treino*size_of_validation)], dtype = np.uint8)
	validation = np.empty([int(num_treino*size_of_validation),heigth,width,dimension],dtype=np.uint8)

	#preenchendo os vetores de treino e de teste
	for directory in os.listdir(path):
		if directory == 'treino':
			k = 0
			l = 0
			for classe,dir_treino in enumerate(os.listdir(path + '/' + directory)):
				lista_imagens_classe = os.listdir(path + '/' + directory + '/' + dir_treino)
				num_exemplos_por_classe.append(len(lista_imagens_classe))
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

	
	#definindo o numero de imagens para cada classe no vetor de validacao
	num_exemplos_por_classe = list(map(lambda i:int(i*size_of_validation),num_exemplos_por_classe))

	#construindo conjunto de validacao
	#nao estou conseguindo deletar os elementos
	cont = 0
	num = 1
	v = 0
	for k in range(0,len(treino)):
		if cont == len(num_exemplos_por_classe):
			break
		if num <= num_exemplos_por_classe[cont]:
			validation[v] = treino[k]
			labels_validation[v] = cont
			elementos_a_retirar.append(k)
			v = v + 1
			num = num + 1
		else:
			while labels_treino[k] == cont:
				k = k + 1
			k = k - 1
			num = 1
			cont = cont + 1

	

path = '/home/victor/base'
bla = processa_base(path,64,64,3,0.2)
