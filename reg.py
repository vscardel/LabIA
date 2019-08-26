import numpy as np
import cv2

def gradient_descent_step(b0, w0, batch, learning_rate):
    # compute gradients
    b_grad = 0
    w_grad = 0
    N = len(batch)
    for i in range(N):
        x = batch[i, 0]
        y = batch[i, 1]
        b_grad += (2.0/N)*(w0*x + b0 - y)
        w_grad += (2.0/N)*x*(w0*x + b0 - y)

    # update parameters
    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)

    return b1, w1

def MSE (b,w,batch):
	error = 0
	N = len(batch)
	for i in range(N): 
		x = batch[i, 0]
		y = batch[i, 1]
		y_ = x*w+b
		error += (y-y_)**2
	error /= N
	return error

pontos = np.loadtxt('arquivo.txt',dtype=np.float32)

w = 0
b = 0
learning_rate = 0.001
batch_size = 20
for i in range(10):
	batch = np.take(pontos,np.random.permutation(len(pontos))[:batch_size],axis = 0)
	b,w = gradient_descent_step(b,w,batch,learning_rate)
	print b,w,MSE(b,w,pontos)



