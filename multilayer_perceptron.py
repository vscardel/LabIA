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

def aplica_sigmoide(y_):
	for i in range(len(y_)):
		y_[i] = sigmoid(y_[i])
	return y_

#funcao de debug
def ve_maior_index(vet):
	vet = np.reshape(vet,-1)
	maior = -100000000000000
	index = 0
	for cont,i in enumerate(vet):
		if i > maior:
			maior = i
			index = cont
	return index

def aplica_modelo(img,ppc,psc,bpc,bsc):
	features  = np.reshape(img,-1)
	features = (features * 1/255)
	
	h_ = (features @ ppc) + bpc
	h_ = aplica_sigmoide(h_)

	y_ = (h_ @ psc) + bsc
	y_ = aplica_sigmoide(y_)
	return y_,h_

def one_hot_encode(y_):
	index = ve_maior_index(y_)
	one_hot_encoding = np.zeros([len(y_)],dtype = np.uint8)
	one_hot_encoding[index] = 1
	return one_hot_encoding

#recebe 'p' se inicializar a primeira camada e 's' se incializar a segunda,
#bp se inicializar o bias da primeira camada e bs caso inicialize a segunda
def inicializa_pesos(camada = ''):
	global num_features, num_classes, num_features_segunda_camada

	#numero de linhas da matriz de pesos
	num_linhas,num_colunas = 0,0

	if camada == 'p':
		pesos = np.random.normal(loc=0.0,scale=0.01,size = num_features*num_features_segunda_camada)
		num_linhas = num_features
		num_colunas = num_features_segunda_camada
	elif camada == 's':
		pesos = np.random.normal(loc=0.0,scale=0.01,size = num_features_segunda_camada*num_classes)
		num_linhas = num_features_segunda_camada
		num_colunas = num_classes
	elif camada == 'bp':
		pesos = np.random.normal(loc=0.0,scale=0.01,size = num_features_segunda_camada)
		num_linhas = num_features_segunda_camada
		num_colunas = 0
	else:
		pesos = np.random.normal(loc=0.0,scale=0.01,size = num_classes)
		num_linhas = num_classes
		num_colunas = 0

	if not num_colunas:
		return pesos
	else:
		return np.reshape(pesos,(num_linhas,num_colunas))
	return pesos

def gradient_descent_step(imagens,labels,ppc,psc,learning_rate,bpc,bsc):

	global num_features,num_features_segunda_camada,num_classes

	N = len(imagens)

	gradiente_primeira_camada = np.zeros([num_features,num_features_segunda_camada],dtype = np.float64)
	gradiente_segunda_camada = np.zeros([num_features_segunda_camada,num_classes],dtype = np.float64)
	gradiente_bias_primeira_camada = np.zeros([num_features_segunda_camada],dtype = np.float64)
	gradiente_bias_segunda_camada = np.zeros([num_classes],dtype = np.float64)

	
	for i in range(len(imagens)):

		img = imagens[i]
		y_imagem,cam_meio_img = aplica_modelo(img,ppc,psc,bpc,bsc)
		
		features = np.reshape(img,-1)

		delta_kas_atual = np.empty([num_classes],dtype = np.float64)

		#calcula os delta_kas da imagem atual
		for j in range(num_classes):

			y = labels[i]
			delta_k = (y_imagem[j]-y[j])*y_imagem[j]*(1-y_imagem[j])
			gradiente_bias_segunda_camada[j] += (1/N) * delta_k
			delta_kas_atual[j] = delta_k

		#calcula os gradientes da segunda camada
		for j in range(len(cam_meio_img)):
			for k in range(num_classes):
				gradiente_segunda_camada[j][k] += (1/N)*(delta_kas_atual[k]*cam_meio_img[j])
		#calcula os gradientes da primeira camada
		for f in range(len(features)):
			delta_j = 0.0
			for j in range(len(cam_meio_img)):
				summ = 0.0
				#calcula delta_j
				for k in range(num_classes):
					summ += delta_kas_atual[k]*pesos_segunda_camada[j][k]

				delta_j = cam_meio_img[j]*(1-cam_meio_img[j])*summ
				gradiente_bias_primeira_camada[j] += (1/N) * delta_j
				gradiente_primeira_camada[f][j] += (1/N)* (delta_j*features[f])


	np.reshape(gradiente_primeira_camada,-1)
	np.reshape(gradiente_segunda_camada,-1)
	np.reshape(pesos_primeira_camada,-1)
	np.reshape(pesos_segunda_camada,-1)
	novos_pesos_primeira_camada = pesos_primeira_camada - (learning_rate*gradiente_primeira_camada)
	print(learning_rate*gradiente_primeira_camada)
	np.reshape(novos_pesos_primeira_camada,(num_features,num_features_segunda_camada))
	novos_pesos_segunda_camada = pesos_segunda_camada - (learning_rate*gradiente_segunda_camada)
	print(learning_rate*gradiente_segunda_camada)
	np.reshape(novos_pesos_segunda_camada,(num_features_segunda_camada,num_classes))
	novo_bias_primeira_camada = bias_primeira_camada - (learning_rate*gradiente_bias_primeira_camada)
	novo_bias_segunda_camada = bias_segunda_camada - (learning_rate*gradiente_bias_segunda_camada)

	return novos_pesos_primeira_camada,novos_pesos_segunda_camada,novo_bias_primeira_camada,novo_bias_segunda_camada

def acc(imagens,labels,ppc,psc,bpc,bsc):
	global num_classes
	acertos = 0
	for cont,img in enumerate(imagens):
		y_,h_ = aplica_modelo(img,ppc,psc,bpc,bsc)
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
path = '/tmp/guest-synwlu/base'
heigth = 64
width = 64
dimension = 3
size_of_validation = 0.2
num_treino,num_teste = 0,0
num_classes = 4
num_features = heigth*width*dimension
num_features_segunda_camada = 64
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
learning_rate = 0.001
batch_size = 20
num_iteracoes_treino = 150
########################################

print("processando a base de dados")

print() 

processa_base()

for i in range(num_iteracoes_treino):
	print("iteracao " + str(i+1))
	print()
	lista_indices = np.random.permutation(len(treino_oficial))
	batch = np.take(treino_oficial,lista_indices[:batch_size],axis=0)
	labels_batch = np.take(labels_treino_oficial,lista_indices[:batch_size],axis=0)
	pesos_primeira_camada,pesos_segunda_camada,bias_primeira_camada,bias_segunda_camada = gradient_descent_step(batch,labels_batch,
	pesos_primeira_camada,pesos_segunda_camada,
	learning_rate,bias_primeira_camada,bias_segunda_camada)
	print("acuracia da iteracao " + str(i+1) + ': ', end="")
	print(acc(validacao,labels_validacao,pesos_primeira_camada,
	pesos_segunda_camada,bias_primeira_camada,bias_segunda_camada))
	print()
	
# print('salvando modelo')
# salva_modelo('modelo0',pesos,bias)
# print('modelo salvo')
