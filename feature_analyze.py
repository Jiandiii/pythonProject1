import numpy as np
import scipy.sparse as sp
import pickle
dataset='Cora'
graph_noise=0.3
label_noise=0.3
feature_noise=0.3
idx_features_labels = np.genfromtxt("{}{}.content".format("./tmp/Cora/", "cora"), dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
datapath='./new/{}/g_{}_l_{}_f_{}/'.format(dataset,graph_noise,label_noise,feature_noise)
with open(datapath + 'Cora_features.pkl', 'rb') as f:
    features_noise = pickle.load(f, encoding='latin1')
noise_nodes=[]
for i in range(features.shape[0]):
    if features[i] !=features_noise[i]:
        noise_nodes.append(i)

