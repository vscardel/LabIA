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
	tam_classes = list(tam_classes)
	constroi_validacao()

def sigmoid(x):
	return 1/(1+np.exp(-x))

def gradient_descent_step(imagem,label,pesos,learning_rate,bias):
	global num_features,num_classes
	tam_gradiente = num_features*num_classes
	gradiente = np.empty([tam_gradiente])
	cont = 0
	der_parcial_atual = 0.0
	for i in range(num_features):
		for j in range(num_classes):
			features = np.reshape(imagem,-1)
			y_ = sigmoid(np.dot(features,pesos[j][0])+bias[j])
			if cont == 0:
				bias_g = (y_-label)*y_*(1-y_)
			der_parcial_atual = (y_-label)*y_*(1-y_)*features[i]
			gradiente[cont] = der_parcial_atual
			cont = cont + 1
	bias_update = bias - (learning_rate*bias_g)
	pesos_update = pesos - (learning_rate*gradiente)

	return bias_update,pesos_update


###############GLOBALS#################

path = '/home/tomas/base'
heigth = 64
width = 64
dimension = 3
size_of_validation = 0.2
num_treino,num_teste = 0,0
num_classes = 4
num_features = heigth*width*dimension

#######################################

calcula_tamanho_dos_diretorios()

treino = np.empty([num_treino,heigth,width,dimension], dtype=np.float64)
teste = np.empty([num_teste,heigth,width,dimension], dtype=np.float64)
labels_treino = np.empty([num_treino], dtype = np.uint8)
tam_validacao = int(size_of_validation*num_treino)
tam_treino_oficial = num_treino - tam_validacao
tam_classes = []

validacao = np.empty([tam_validacao,heigth,width,dimension], dtype=np.float64)
labels_validacao = np.empty([tam_validacao],dtype=np.uint8)
treino_oficial = np.empty([tam_treino_oficial,heigth,width,dimension], dtype=np.float64)
labels_treino_oficial = np.empty([tam_treino_oficial], dtype = np.uint8)

#############PARAMETROS#################

pesos = [ [np.empty([num_features])] for i in range(num_features)]
bias = [0.0 for i in range(num_features)]
learning_rate = 0.0000000000001
batch_size = 20
num_iteracoes_treino = 20

########################################

print("processando a base de dados")
print() 
processa_base()
for i in range(num_iteracoes_treino):
	print("iteracao" + str(i+1))
	print()
	lista_indices = np.random.permutation(len(treino_oficial))
	batch = np.take(treino_oficial,lista_indices[:batch_size],axis=0)
	labels_batch = np.take(labels_treino_oficial,lista_indices[:batch_size],axis=0)
	bias,pesos = gradient_descent_step(batch,labels_batch,pesos,learning_rate,bias)
	print(MSE(validacao,labels_validacao,pesos,bias),acc(validacao,labels_validacao,pesos,bias))
	print()
