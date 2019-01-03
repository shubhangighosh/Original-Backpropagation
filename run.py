# Importing Header files
from random import seed
from random import randrange
from random import random
from random import choice
from csv import reader
from math import exp
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#softmax
def softmax(arr):
	x = np.exp(arr)/((np.exp(arr)).sum(axis=0))
	return x
#derivative of tanh function	
def deriv_tanh(arr):
	return (1.0-np.tanh(arr)**2.0)
#derivative of softmax
def deriv_softmax(arr):
	return softmax(arr)*(1.0-softmax(arr))	
#sigmoid
def sigmoid(arr):
	return (1.0+np.exp(-1.0*arr))**(-1.0)	
np.random.seed(1)
#no.of i/ps, hidden layer neurons and o/p classes
n_inputs = 96
n_hidden = 50
n_outputs = 4

#weights from i.p to hidden layer
weights_ip_layer = []
for i in range(n_inputs):
	weights_ip_layer.append([random() for i in range(n_hidden)])
bias_ip = random()
weights_ip_layer = np.array(weights_ip_layer)
weights_ip_layer = weights_ip_layer -0.5	

#weights from hidden to o/p layer
weights_hidden_layer = []
for i in range(n_hidden):
	weights_hidden_layer.append([random() for i in range(n_outputs)])
bias_hidden = random()
weights_hidden_layer = np.array(weights_hidden_layer)	
weights_hidden_layer = weights_hidden_layer - 0.5	

weights_hidden_layer = np.array(weights_hidden_layer)
weights_ip_layer = np.array(weights_ip_layer)

#forward prop with current weights to calculate o/p
def forward_prop(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden):
	hidden_ip = np.dot(input_t,weights_ip_layer) + bias_ip
	hidden_op = np.tanh(hidden_ip)
	op_ip = np.dot(hidden_op,weights_hidden_layer) + bias_hidden
	op_op = np.tanh(op_ip)
	return op_op, hidden_op

#Backpropagating error for part a
def back_prop_err(op_exp,op_op,hidden_op,input_t):	
	err_op = (op_exp-op_op)*deriv_tanh(op_op)
	del_hid_weight = np.dot(np.array([hidden_op]).T,np.array([err_op]))
		
	err_hid = del_hid_weight.sum(axis=1)*deriv_tanh(hidden_op)
	del_ip_weight = np.dot(np.array([input_t]).T,np.array([hidden_op]))
	del_bias_hidden = err_op.sum()
	del_bias_ip = err_hid.sum()
	return del_hid_weight, del_ip_weight, del_bias_hidden, del_bias_ip

#Backpropagating error for part b
def back_prop_err2(op_exp,op_op,hidden_op,input_t,weights_hidden_layer,weights_ip_layer,gamma):	
	err_op = (op_exp-op_op)*deriv_tanh(op_op)
	del_hid_weight = np.dot(np.array([hidden_op]).T,np.array([err_op]))+2.0*gamma*weights_hidden_layer
		
	err_hid = del_hid_weight.sum(axis=1)*deriv_tanh(hidden_op)
	del_ip_weight = np.dot(np.array([input_t]).T,np.array([hidden_op]))+2.0*gamma*weights_ip_layer
	del_bias_hidden = err_op.sum()
	del_bias_ip = err_hid.sum()
	return del_hid_weight, del_ip_weight, del_bias_hidden, del_bias_ip	

#Updating weights after calculating backpropagation error
def update_weights(weights_hidden_layer,weights_ip_layer,bias_hidden,bias_ip,del_hid_weight,del_ip_weight,del_bias_hidden,del_bias_ip,l_rate):
	weights_hidden_layer += l_rate*(del_hid_weight)
	weights_ip_layer += l_rate*(del_ip_weight)
	bias_hidden += l_rate*(del_bias_hidden)
	bias_ip += l_rate*(del_bias_ip)
	return weights_hidden_layer,weights_ip_layer,bias_hidden,bias_ip

#Training neural net by repeating the process
def train_net(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden,op_exp,l_rate,epochs):
	#print weights_ip_layer
	for i in range(epochs):

		op_op, hidden_op = forward_prop(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden)
		del_hid_weight, del_ip_weight, del_bias_hidden, del_bias_ip = back_prop_err(op_exp,op_op,hidden_op,input_t)
		weights_hidden_layer, weights_ip_layer,bias_hidden,bias_ip = update_weights(weights_hidden_layer,weights_ip_layer,bias_hidden,bias_ip,del_hid_weight,del_ip_weight,del_bias_hidden,del_bias_ip,l_rate)
		#For checking validation -- no of epochs
		#print "Epoch %d: %f" %(i,((op_exp-op_op)**2.0).sum())
		#print op_exp, op_op
	
	return weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden
vtrain_net = np.vectorize(train_net)	

#Training neural net by repeating the process for part b
def train_net2(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden,op_exp,l_rate,epochs,gamma):
	for i in range(epochs):
		op_op, hidden_op = forward_prop(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden)
		del_hid_weight, del_ip_weight, del_bias_hidden, del_bias_ip = back_prop_err2(op_exp,op_op,hidden_op,input_t,weights_hidden_layer,weights_ip_layer,gamma)
		weights_hidden_layer, weights_ip_layer,bias_hidden,bias_ip = update_weights(weights_hidden_layer,weights_ip_layer,bias_hidden,bias_ip,del_hid_weight,del_ip_weight,del_bias_hidden,del_bias_ip,l_rate)
	return weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden		

#reading training data
df=pd.read_csv('../../Dataset/DS2-train.csv', sep=',',header=None)
dataset = df.values
dataset = np.array(dataset)
np.random.shuffle(dataset)
scaler = StandardScaler()
scaler.fit(dataset[:,:-4])
dataset[:,:-4] = scaler.transform(dataset[:,:-4])



train_ip = dataset[:,:-4]
train_op = dataset[:,-4:]


#sequentially feeding in all training sets:


for j in range(len(train_ip)):
 	input_t = train_ip[j]
 	op_exp = train_op[j]
 	weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden = train_net(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden,op_exp,0.001,500)
	


#reading test data
df=pd.read_csv('../../Dataset/DS2-test.csv', sep=',',header=None)
dataset = df.values
dataset = np.array(dataset)


dataset[:,:-4] = scaler.transform(dataset[:,:-4])
valid_ip, non_valid_ip = train_test_split(dataset, test_size=0.5)


#Validation data
valid_ip = dataset[:,:-4]
valid_op = dataset[:,-4:]
input_t = train_ip[0]
op_exp = train_op[0]
valid_ip_layer,valid_bias_ip,valid_hidden_layer,valid_bias_hidden = train_net(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden,op_exp,0.001,400)

for j in range(len(valid_ip)):
 	input_t = train_ip[j]
 	op_exp = train_op[j]
 	valid_ip_layer,valid_bias_ip,valid_hidden_layer,valid_bias_hidden = train_net(input_t,valid_ip_layer,valid_bias_ip,valid_hidden_layer,valid_bias_hidden,op_exp,0.001,400)
	

test_ip = dataset[:,:-4]
test_op = dataset[:,-4:]

#predicting for test data
pred_op = []
for j in range(len(test_ip)):
	
	op_op, hidden_op = forward_prop(test_ip[j],weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden)
	pred_op.append(np.argmax(op_op))
#print pred_op, test_op	

target_names = ['mountain', 'forest','coast', 'insidecity']

Y_test = np.argmax(test_op, axis=1)
print len(Y_test), len(pred_op)
#printing classification measures
print 100.0*accuracy_score(Y_test, pred_op)
print(classification_report(Y_test, pred_op, target_names=target_names))

#varying gamma - regularisation parameter and doing the same
g_val = np.array([0.01,0.1,1.0,10.0,100.0])
for gamma in g_val:
	pred_op = []
	for j in range(len(train_ip)):
		input_t = train_ip[j]
		op_exp = train_op[j]
		weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden = train_net2(input_t,weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden,op_exp,0.001,400,gamma)
	for j in range(len(test_ip)):
		
		op_op, hidden_op = forward_prop(test_ip[j],weights_ip_layer,bias_ip,weights_hidden_layer,bias_hidden)
		pred_op.append(np.argmax(op_op))
	my_df = pd.DataFrame(np.array([np.array(weights_ip_layer),np.array(weights_hidden_layer)]))
	my_df.to_csv('weights'+str(gamma)+'.csv', index=False, header=False)	
	print len(Y_test), len(pred_op)
	print(classification_report(Y_test, pred_op, target_names=target_names))
