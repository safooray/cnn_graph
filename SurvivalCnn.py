import time
start = time.time()
import sys
import os
sys.path.append('./lib')
import models, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse import csr_matrix
from SurvivalAnalysis import SurvivalAnalysis
print(str(time.time()-start))
LOAD_A = False
LOG = True
CL = 3
def print_log(msg):
	if LOG:
		print(msg)

def prepare_graphs():
	p = os.path.join(os.getcwd(), '../survivalnet/data/Brain_Integ.mat')
	D = sio.loadmat(p)
	T = np.asarray([t[0] for t in D['Survival']])
	O = 1 - np.asarray([c[0] for c in D['Censored']])
	X = D['Integ_X']#[:,1855:]
	X = (X - np.mean(X, axis=0))/np.std(X, axis = 0)
	fold_size = int(10 * len(X) / 100)
	X_train, T_train, O_train = X[2*fold_size:], T[2*fold_size:], O[2*fold_size:]
	X_test, T_test, O_test    = X[:fold_size], T[:fold_size], O[:fold_size]
	X_val, T_val, O_val       = X[fold_size:2*fold_size], T[fold_size:2*fold_size], O[fold_size:2*fold_size]
	print_log('train and test shapes:' + str(X_train.shape) + str(X_test.shape))
	if not LOAD_A:
		start = time.time()
		dist, idx = graph.distance_scipy_spatial(X_train.T, k=4, metric='euclidean')
		print_log('graph constructed:'+ str(dist.shape) + str(idx.shape) + ' in ' + str(time.time()-start))
		A = graph.adjacency(dist, idx).astype(np.float32)
		d = X.shape[1]
		assert A.shape == (d, d)
		np.savez('A_sml', data=A.data, indices=A.indices,
				indptr=A.indptr, shape=A.shape )
		print('d = |V| = {}, k|V| < |E| = {}'.format(d, A.nnz))
	#plt.spy(A, markersize=2, color='black');
	#plt.savefig('tmp.png')
	else:
		start = time.time()
		loader = np.load('A.npz')
		A = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
		print_log('graph loaded:'+ ' in ' + str(time.time()-start))
		print ('adjacency matrix type and shape: ', A.__class__, A.shape)
	start = time.time()
	graphs, perm = coarsening.coarsen(A, levels=CL, self_connections=False)
	print_log('graph coarsened:'+' in ' + str(time.time()-start))
	X_train = coarsening.perm_data(X_train, perm)
	X_val = coarsening.perm_data(X_val, perm)
	X_test = coarsening.perm_data(X_test, perm)
	print_log('train and test shapes:' + str(X_train.shape) + str(X_test.shape))
	L = [graph.laplacian(A, normalized=True) for A in graphs]
	#graph.plot_spectrum(L)

	n_train = len(X_train)
	params = dict()
	params['dir_name']       = 'demo'
	params['num_epochs']     = 2000
	params['batch_size']     = int(len(X_train)/1.0)
	params['eval_frequency'] = 10

	# Building blocks.
	params['filter']         = 'chebyshev5'
	params['brelu']          = 'b1relu'
	params['pool']           = 'apool1'

	# Architecture.
	params['F']              = [8,8,8]  # Number of graph convolutional filters.
	params['K']              = [9,9,9]  # Polynomial orders.
	params['p']              = [2, 2, 2]    # Pooling sizes.
	params['M']              = [128, 1]  # Output dimensionality of fully connected layers.

	# Optimization.
	params['regularization'] = 0 
	params['dropout']        = 1
	params['learning_rate']  = 1e-4
	params['decay_rate']     = 0.999
	params['momentum']       = 0
	params['decay_steps']    = n_train / params['batch_size']
	model = models.cgcnn(L, **params)
	accuracy, loss, t_step = model.cox_fit(X_train, T_train, O_train, X_val, T_val, O_val)
	#res = model.evaluate(X_test, y_test)
	#print(res[0])

if __name__=="__main__":
	prepare_graphs()
