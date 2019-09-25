import os
import sys
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
	global treino,teste,labels_treino,tam_classes,num_classes
	for directory in os.listdir(path):
		if directory == 'treino':
			k = 0
			diretorios = os.listdir(path + '/' + directory)
			diretorios.sort()
			for classe,dir_treino in enumerate(diretorios):
				lista_imagens_classe = os.listdir(path + '/' + directory + '/' + dir_treino)
				tam_classes.append(len(lista_imagens_classe))
				for img in lista_imagens_classe:
					imagem = cv2.imread(path + '/' + directory + '/' + dir_treino + '/' + img,cv2.IMREAD_COLOR)
					treino[k] = imagem
					one_hot_encoding = np.zeros([num_classes])
					one_hot_encoding[classe] = 1
					labels_treino[k] = one_hot_encoding
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
		indice = np.where(labels_treino[k] == 1)
		if indice[0][0] != classe_atual:
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

def aplica_modelo(img,pesos,bias):
	features  = np.reshape(img,-1)
	features = (features * 1/255)
	y_ = (features @ pesos) + bias
	for i in range(len(y_)):
		y_[i] = sigmoid(y_[i])
	return y_

def one_hot_encode(y_):
	maior = -1000
	pos = 0
	for i in range (len(y_)):
		if y_[i] > maior:
			maior = y_[i]
			pos = i
	one_hot_encoding = np.zeros([num_classes],dtype = np.uint8)
	one_hot_encoding[pos] = 1
	return one_hot_encoding

#recebe 'p' se inicializar a primeira camada e 's' se incializar a segunda,
#bp se inicializar o bias da primeira camada e bs caso inicialize a segunda
def inicializa_pesos(camada = ''):
	global num_features, num_classes, num_features_segunda_camada

	#numero de linhas da matriz de pesos
	num_linhas,num_colunas = 0,0

	if camada == 'p':
		pesos = np.random.rand(num_features*num_features_segunda_camada)
		num_linhas = num_features
		num_colunas = num_features_segunda_camada
	elif camada == 's':
		pesos = np.random.rand(num_features_segunda_camada*num_classes)
		num_linhas = num_features_segunda_camada
		num_colunas = num_classes
	elif camada == 'bp':
		pesos = np.random.rand(num_features_segunda_camada)
		num_linhas = num_features_segunda_camada
		num_colunas = 0
	else:
		pesos = np.random.rand(num_classes)
		num_linhas = num_classes
		num_colunas = 0

	pesos = pesos * 0.01
	if not num_colunas:
		return pesos
	else:
		return np.reshape(pesos,(num_linhas,num_colunas))
	return pesos

def gradient_descent_step(imagens,labels,pesos,learning_rate,bias):
	global num_features,num_classes
	flag_bias = 0
	gradiente = np.empty([num_features*num_classes])
	bias_gradiente = np.empty([num_classes])
	N = len(imagens) + 1
	gradiente_atual = 0.0
	bias_atual = 0.0
	i,j = 0,0
	modelos = np.empty([len(imagens),num_classes],dtype = np.float64)
	vetor_features = np.empty([len(imagens),num_features])

	for cont,img in enumerate(imagens):
		y_imagem = aplica_modelo(img,pesos,bias)
		f = np.reshape(img,-1)
		modelos[cont] = y_imagem
		vetor_features[cont] = f

	for k in range(num_features*num_classes):
		for cont,img in enumerate(imagens):
		 		label = labels[cont]
		 		features = vetor_features[cont]
		 		y_ = modelos[cont]
		 		gradiente_atual += (1/N)*((y_[j]-label[j])*y_[j]*(1-y_[j])*features[i])
		 		if not flag_bias:
		 			bias_atual += (1/N)*((y_[j]-label[j])*y_[j]*(1-y_[j]))
		j = j + 1
		if j == num_classes:
			j = 0
			i = i + 1
			flag_bias = 1
		gradiente[k] = gradiente_atual
		if not flag_bias:
			bias_gradiente[j] = bias_atual
		gradiente_atual = 0.0
		bias_atual = 0.0
	novos_pesos = np.reshape(pesos,-1) - (learning_rate*gradiente)
	novo_bias = bias - (learning_rate*bias_gradiente)
	novos_pesos = np.reshape(novos_pesos,(num_features,num_classes))
	return novos_pesos,novo_bias

def acc(imagens,labels,pesos,bias):
	global num_classes
	acertos = 0
	for cont,img in enumerate(imagens):
		y_ = aplica_modelo(img,pesos,bias)
		one_hot_encoding = one_hot_encode(y_)
		if (np.array_equal(one_hot_encoding,labels[cont])):
			acertos += 1
	return (acertos/len(imagens))

def salva_modelo(nome_modelo,pesos,bias):
	global num_classes,num_features
	f = open(nome_modelo,'w')
	for i in range(num_features):
		for j in range(num_classes):
			if j < num_classes-1:
				f.write(str(pesos[i][j]) + ' ')
			else:
				f.write(str(pesos[i][j]) + '\n')
	f.write('b')
	for i in range(num_classes):
		if i < num_classes - 1:
			f.write(str(bias[i]) + ' ')
		else:
			f.write(str(bias[i]) + '\n')
	f.close()

###############GLOBALS#################
path = '/home/victor/base'
heigth = 64
width = 64
dimension = 3
size_of_validation = 0.2
num_treino,num_teste = 0,0
num_classes = 4
num_features = heigth*width*dimension
num_features_segunda_camada = 100
#######################################
calcula_tamanho_dos_diretorios()
treino = np.empty([num_treino,heigth,width,dimension], dtype=np.uint8)
teste = np.empty([num_teste,heigth,width,dimension], dtype=np.uint8)
labels_treino = np.empty([num_treino,num_classes], dtype = np.uint8)
tam_validacao = int(size_of_validation*num_treino)
tam_treino_oficial = num_treino - tam_validacao
tam_classes = []
validacao = np.empty([tam_validacao,heigth,width,dimension], dtype=np.uint8)
labels_validacao = np.empty([tam_validacao,num_classes],dtype=np.uint8)
treino_oficial = np.empty([tam_treino_oficial,heigth,width,dimension], dtype=np.uint8)
labels_treino_oficial = np.empty([tam_treino_oficial,num_classes], dtype = np.uint8)
#############PARAMETROS#################
pesos_primeira_camada = inicializa_pesos('p')
pesos_segunda_camada = inicializa_pesos('s')
bias_primeira_camada = inicializa_pesos('bp')
bias_segunda_camada = inicializa_pesos('bs')
learning_rate = 0.00001
batch_size = 50
num_iteracoes_treino = 150
########################################

print("processando a base de dados")

print() 

processa_base()

# for i in range(num_iteracoes_treino):
# 	print("iteracao " + str(i+1))
# 	print()
# 	lista_indices = np.random.permutation(len(treino_oficial))
# 	batch = np.take(treino_oficial,lista_indices[:batch_size],axis=0)
# 	labels_batch = np.take(labels_treino_oficial,lista_indices[:batch_size],axis=0)
# 	pesos,bias = gradient_descent_step(batch,labels_batch,pesos,learning_rate,bias)
# 	print("acuracia da iteracao " + str(i+1) + ': ', end="")
# 	print(acc(validacao,labels_validacao,pesos,bias))
# 	print()
	
# print('salvando modelo')
# salva_modelo('modelo0',pesos,bias)
# print('modelo salvo')