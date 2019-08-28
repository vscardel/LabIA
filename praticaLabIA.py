import os
import numpy as np
import cv2

def calcula_tamanho_dos_diretorios():
	global num_treino,num_teste
	for directory in os.listdir(path):
		if directory == 'treino':
			for dir_treino in os.listdir(path + '/' + directory):
				num_treino = num_treino + len(os.listdir(path + '/' + directory + '/' + dir_treino))
		else:
			num_teste = num_teste + len(os.listdir(path + '/' + directory))

def preenche_vetores_de_treino_e_teste():
	global treino,teste,labels_treino,tam_classes
	for directory in os.listdir(path):
		if directory == 'treino':
			k = 0
			for classe,dir_treino in enumerate(os.listdir(path + '/' + directory)):
				lista_imagens_classe = os.listdir(path + '/' + directory + '/' + dir_treino)
				tam_classes.append(len(lista_imagens_classe))
				for img in lista_imagens_classe:
					imagem = cv2.imread(path + '/' + directory + '/' + dir_treino + '/' + img,cv2.IMREAD_COLOR)
					treino[k] = imagem
					labels_treino[k] = classe
					k = k + 1
		else:
			k = 0
			for img in os.listdir(path + '/' + directory):
				imagem = cv2.imread(path + '/' + directory + '/' + img,cv2.IMREAD_COLOR)
				teste[k] = imagem
				k = k + 1

def constroi_validacao():
	global labels_treino,tam_classes,validacao,treino,labels_validacao,labels_treino,treino_oficial,labels_treino_oficial

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

def processa_base():
	global tam_classes
	preenche_vetores_de_treino_e_teste()
	#aplicando os pesos para cada classe
	tam_classes = map(lambda i: int(i*size_of_validation),tam_classes)
	constroi_validacao()

def gradient_descent_step(imagens,labels,pesos,learning_rate,bias):
	P = len(imagens)
	N = len(pesos)
	der_par_atual = 0

	bias_g = 0
	gradiente = np.empty([N])

	for j in range(N):
		for i in range(P):
			features = np.reshape(imagens[i],-1)
			der_par_atual += (2.0/P)*((np.dot(features,pesos) + bias) - labels[i])*features[j]
			if j == 0:
				bias_g += (2.0/P)*((np.dot(features,pesos) + bias) - labels[i])
		gradiente[j] = der_par_atual

	bias_update = bias - (learning_rate*bias_g)
	pesos_update = pesos - (learning_rate*gradiente)

	return bias_update,pesos_update

def MSE(batch,labels,pesos,bias):
	N = len(batch)
	erro = 0
	for cont,img in enumerate(batch):
		features = np.reshape(img,-1)
		y_ = np.dot(features,pesos) + bias
		erro += (labels[cont]-y_)**2
	erro /= N
	return erro

def acc(batch,labels,pesos,bias):
	N = len(batch)
	classe,acertos = 0,0
	for cont,img in enumerate(batch):
		features = np.reshape(img,-1)
		y_ = np.dot(features,pesos) + bias
		if y_ <= 0.7:
			classe = 0 
		elif y_ > 0.7 and y_ <= 1.4:
			classe = 1
		elif y_ > 1.4 and y_ <= 2.1:
			classe = 2
		elif y_ > 2.1:
			classe = 3
		if classe == labels[cont]:
			acertos += 1

	acc = float(acertos)/N
	return acc
###############GLOBALS#################

path = '/home/victor/base'
heigth = 64
width = 64
dimension = 3
size_of_validation = 0.2
num_treino,num_teste = 0,0

#######################################

calcula_tamanho_dos_diretorios()

treino = np.empty([num_treino,heigth,width,dimension], dtype=np.uint8)
teste = np.empty([num_teste,heigth,width,dimension], dtype=np.uint8)
labels_treino = np.empty([num_treino], dtype = np.uint8)
tam_validacao = int(size_of_validation*num_treino)
tam_treino_oficial = num_treino - tam_validacao
tam_classes = []

validacao = np.empty([tam_validacao,heigth,width,dimension], dtype=np.uint8)
labels_validacao = np.empty([tam_validacao],dtype=np.uint8)
treino_oficial = np.empty([tam_treino_oficial,heigth,width,dimension], dtype=np.uint8)
labels_treino_oficial = np.empty([tam_treino_oficial], dtype = np.uint8)

#############PARAMETROS#################

#1) learning_rate1 = 0.000000000000001 pior q o 2
#2) learning_rate2 = 0.00000000000001 pior q o 3
#3) learning_rate = 0.0000000000001

pesos = np.empty([heigth*width*dimension])
learning_rate = 0.0000000000001
bias = 0
batch_size = 20
num_iteracoes_treino = 20

########################################

processa_base()
for i in range(num_iteracoes_treino):
	print("iteracao" + str(i+1))
	print
	lista_indices = np.random.permutation(len(treino_oficial))
	batch = np.take(treino_oficial,lista_indices[:batch_size],axis=0)
	labels_batch = np.take(labels_treino_oficial,lista_indices[:batch_size],axis=0)
	bias,pesos = gradient_descent_step(batch,labels_batch,pesos,learning_rate,bias)
	print(MSE(validacao,labels_validacao,pesos,bias),acc(validacao,labels_validacao,pesos,bias))
