# ---------------------------------------------------------------------------------------------------------- #
# Author: Victor Cardel                                                                                              #
# ---------------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import sys
import numpy as np
import cv2
import os

# ---------------------------------------------------------------------------------------------------------- #
# Linha de comando:                                                                                          #
#         python3 train_and_test.py /path/to/train_folder /path/to/test_folder /path/to/output.txt           #
# Descrição:                                                                                                 #
#         * carrega todas as imagens nos subdiretórios de /path/to/train_folder e cria modelo de             #
#           classificação                                                                                    #
#         * carrega todas as imagens a serem classificadas no diretório /path/to/test_folder                 #
#         * para cada imagem de teste carregada, imprime uma linha no arquivo /path/to/output.txt com dois   #
#           valores separados por um espaço, sendo eles o nome do arquivo da imagem seguido do nome da sua   #
#           respectiva classe ('futebol', 'golfe', 'sinuca' ou 'tenis')                                      #
# ---------------------------------------------------------------------------------------------------------- #

def calcula_tamanho_dos_diretorios():
	global num_treino,num_teste

	for dir_treino in os.listdir(path):
		num_treino += len(os.listdir(path + '/'+ dir_treino))

	num_teste = len(os.listdir(sys.argv[2]))

def preenche_vetores_de_treino_e_teste():
	global treino,teste,labels_treino,tam_classes,num_classes

	k = 0
	diretorios = os.listdir(path)
	diretorios.sort()
	for classe,dir_treino in enumerate(diretorios):
		lista_imagens_classe = os.listdir(path + '/' + dir_treino)
		tam_classes.append(len(lista_imagens_classe))
		for img in lista_imagens_classe:
			imagem = cv2.imread(path + '/' + dir_treino + '/' + img,cv2.IMREAD_COLOR)
			treino[k] = imagem
			one_hot_encoding = np.zeros([num_classes])
			one_hot_encoding[classe] = 1
			labels_treino[k] = one_hot_encoding
			k = k + 1
	k = 0
	for img in os.listdir(sys.argv[2]):
		imagem = cv2.imread(str(sys.argv[2]) + '/' +  img,cv2.IMREAD_COLOR)
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


###############GLOBALS#################
path = sys.argv[1]
heigth = 64
width = 64
channels = 3
num_filters = 64
size_of_validation = 0.2
num_treino,num_teste = 0,0
num_classes = 4
num_features = heigth*width*channels
batch_size = 32
num_iteracoes_treino = 10000
lr = 0.001
#######################################
calcula_tamanho_dos_diretorios()
treino = np.empty([num_treino,heigth,width,channels], dtype=np.uint8)
teste = np.empty([num_teste,heigth,width,channels], dtype=np.uint8)
labels_treino = np.empty([num_treino,num_classes], dtype = np.uint8)
tam_validacao = int(size_of_validation*num_treino)
tam_treino_oficial = num_treino - tam_validacao
tam_classes = []
validacao = np.empty([tam_validacao,heigth,width,channels], dtype=np.uint8)
labels_validacao = np.empty([tam_validacao,num_classes],dtype=np.uint8)
treino_oficial = np.empty([tam_treino_oficial,heigth,width,channels], dtype=np.uint8)
labels_treino_oficial = np.empty([tam_treino_oficial,num_classes], dtype = np.uint8)
#######################################

processa_base()

#creating vector of labels
labels_treino_escalar = np.empty([tam_treino_oficial], dtype = np.uint8)
labels_val = np.empty([tam_validacao], dtype = np.uint8)

for index,one_hot in enumerate(labels_treino_oficial):
	for classe,value in enumerate(one_hot):
		if value == 1:
			labels_treino_escalar[index] = classe
			break

for index,one_hot in enumerate(labels_validacao):
	for classe,value in enumerate(one_hot):
		if value == 1:
			labels_val[index] = classe
			break

#inicializa seu grafo de operacoes
graph = tf.Graph()
with graph.as_default():
	#imagens, labels e learning rate
	img = tf.compat.v1.placeholder(tf.float32,shape=(None,heigth,width,channels))
	labels = tf.compat.v1.placeholder(tf.int64, shape=(None,))
	learning_rate = tf.compat.v1.placeholder(tf.float32)
	training = tf.compat.v1.placeholder(tf.bool)
	#camadas de convolução
	cv1 = tf.compat.v1.layers.conv2d(img,
				     	num_filters,
				     	(3,3),
				     	(1,1),
				     	padding = 'valid',
				     	activation = tf.nn.relu)

	
	cv2 = tf.compat.v1.layers.max_pooling2d(cv1,
					       (2,2),
					       (2,2),
					       padding = 'valid')

	cv2 = tf.layers.dropout(cv2,0.15, training = training)
	

	cv3 = tf.compat.v1.layers.conv2d(cv2,
				     	 num_filters,
				     	 (3,3),
				     	 (1,1),
				     	 padding = 'valid',
				     	 activation = tf.nn.relu)

	
	cv4 = tf.compat.v1.layers.max_pooling2d(cv3,
					       (2,2),
					       (2,2),
					       padding = 'valid')

	cv4 = tf.layers.dropout(cv4,0.15, training = training)

	cv5 = tf.compat.v1.layers.conv2d(cv4,
				         num_filters,
				         (3,3),
				         (1,1),
				         padding = 'valid',
				         activation = tf.nn.relu)

	cv6 = tf.compat.v1.layers.max_pooling2d(cv5,
					       (2,2),
					       (2,2),
					       padding = 'valid')

	cv6 = tf.layers.dropout(cv6,0.15, training = training)


	cv7 = tf.compat.v1.layers.conv2d(cv6,
				     	 num_filters,
				        (3,3),
				        (1,1),
				        padding = 'valid',
				        activation = tf.nn.relu)

	cv8 = tf.compat.v1.layers.max_pooling2d(cv7,
					       (2,2),
					       (2,2),
					       padding = 'valid')

	cv8 = tf.layers.dropout(cv8,0.15, training = training)

	#Need to reshape output of cnn for classifier
	shape = cv8.shape
	cv8 = tf.reshape(cv8,[-1,shape[1]*shape[2]*shape[3]])
	#classifier
	d1 = tf.compat.v1.layers.dense(cv8, 1024, activation=tf.nn.relu, name='d1')
	d1 = tf.layers.dropout(d1,0.5, training = training)
	d2 = tf.compat.v1.layers.dense(d1, 64, activation=tf.nn.relu, name='d2')
	d2 = tf.layers.dropout(d2,0.2, training = training)
	
	output = tf.compat.v1.layers.dense(d2, 4, name='output')
	#loss
	loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)

	train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	
	correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, axis=1), labels), dtype=tf.float32))

	print(correct)


def accuracy(session, Xi, yi):
	batch_size = 32
	cont = 0
	for i in range(0, len(Xi), batch_size):
		X_batch = Xi[i:i+batch_size]
		y_batch = yi[i:i+batch_size]
		ret = session.run([correct], feed_dict = {img : X_batch, labels : y_batch, training : False})
		cont += ret[0]
	return 100.0*cont/len(Xi)

with tf.compat.v1.Session(graph = graph) as session:
	#random initialization of the weigths
	session.run(tf.compat.v1.global_variables_initializer())

	for i in range(num_iteracoes_treino):

		indexes = np.random.permutation(len(treino_oficial))[:batch_size]
		imgs_batch = np.take(treino_oficial, indexes, axis=0)
		labels_batch = np.take(labels_treino_escalar, indexes, axis=0)

		ret = session.run([train_op], feed_dict = {img : imgs_batch, labels : labels_batch, learning_rate : lr, training: True})

		if i%100 == 99:
			print("Iteration #%d" % (i))
			print("TRAIN: ACC=%.5f" % (accuracy(session, treino_oficial, labels_treino_escalar)))
			print("VAL: ACC=%.5f\n" % (accuracy(session, validacao, labels_val)))


