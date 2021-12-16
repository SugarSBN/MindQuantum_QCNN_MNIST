import sys
sys.setrecursionlimit(1000000)

from mindquantum.framework import MQLayer
from mindquantum.core import Z, H, I, RX, RY, RZ, X, UN, Circuit, Hamiltonian, QubitOperator
from mindquantum import ParameterResolver
from mindquantum.simulator import Simulator
import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import struct, scipy
import tools

images, labels = tools.load_mnist("./datasets/")
labels = labels.astype(np.int32)

import math

import matplotlib.pyplot as plt

def encoding(x):
    '''
    intput: x is an array, whose length is 16 * 16, representing the MNIST pixels value
    
    output: c is a parameterizd circuit, and num is a parameter resolver
            
            simulator.apply_circuit(c, num).get_qs(True) will get a qubit state:
                    x0|00000000> + x1|10000000> + x2|01000000> + x3|11000000> + ... + x254|01111111> + x255|11111111>
            which is an "amplitudes encoding" for classic data (x1, ..., x255)

    '''
    c = Circuit()
    tree = []
    for i in range(len(x) - 1):
        tree.append(0)
    for i in range(len(x)):
        tree.append(x[i])
    for i in range(len(x) - 2, -1 ,-1):
        tree[i] += math.sqrt(tree[i * 2 + 1] * tree[i * 2 + 1] + tree[i * 2 + 2] * tree[i * 2 + 2])
    
    path = [[]]
    num = {}
    cnt = 0
    for i in range(1, 2 * len(x) - 1, 2):
        path.append(path[(i - 1) // 2] + [-1])
        path.append(path[(i - 1) // 2] + [1])
        
        tmp = path[(i - 1) // 2]
        controlls = []
        for j in range(len(tmp)):
            controlls.append(tmp[j] * j)
        theta = 0
        if tree[(i - 1) // 2] > 1e-10:
            amp_0 = tree[i] / tree[(i - 1) // 2]
            theta = 2 * math.acos(amp_0)
        num[f'alpha{cnt}'] = theta
        tools.controlled_gate(c, RY(f'alpha{cnt}'), len(tmp), controlls, (0 if len(tmp) > 0 and tmp[0] == -1 else 1))
        cnt += 1

    return c, ParameterResolver(num)


images_params = np.load('MNIST_params.npy')
images_origin = np.load('MNIST_train_formalized.npy')
images_filtered = []
labels_filtered = []

# only work for binary classification. labels_filtered[i] = 0 or 1
for i in range(len(images_params)):
    if labels[i] <= 1:
        images_filtered.append(images_params[i])
        labels_filtered.append(labels[i])

def get_ansatz(n, m):
    '''
    circuit : n qubits
    output : m qubits (reduce because of pooling layer)
    '''
    c = Circuit()
    cnt = 0
    while n > m:
        entan = []
        for i in range(n // 2):
            entan.append(i)
            entan.append(i + (n // 2))
        for i in range(len(entan) - 1):
            q1 = entan[i]
            q2 = entan[i + 1]
            c += RY(f'beta{cnt}').on(q1)
            cnt += 1
            c += RY(f'beta{cnt}').on(q2)
            cnt += 1
            c += X.on(q1, q2)
        for i in range(n // 2):
            c += RZ(f'beta{cnt}').on(i, i + (n // 2))
            cnt += 1
            tools.controlled_gate(c, RZ(f'beta{cnt}'), i, [-(i + (n // 2))], 0)
            cnt += 1
        n = n // 2
    return c

def M(n, l):
    '''
    calculate the coordinate of |n><n| under Pauli basis
    intuitively just polynomial multiplication
    return m is meaningless.

    e.g:
        M(2, 2) = [0.25, 0.25, -0.25, -0.25], ['I0 I1', 'I0 Z1', 'Z0 I1', 'Z0 Z1']
        
    '''
    if l == 1 and n == 0:
        return [0.5, 0.5], ['', 'Z0'], 0
    if l == 1 and n == 1:
        return [0.5, -0.5], ['', 'Z0'], 0
    a, G, m = M(n // 2, l - 1)
    an = [0.5, 0.5] if n % 2 == 0 else [0.5, -0.5]
    Gn = ['', f'Z{m + 1}']
    resa = []
    resG = []
    for i in range(len(a)):
        for j in range(len(an)):
            resa.append(a[i] * an[j])
            resG.append(G[i] + ' ' + Gn[j])
    return resa, resG, m + 1

def Ham(n, l):
    '''
     Ham(2, 2) = |01><01| = (I + Z) / 2 tensor (I - Z) / 2 = (II / 4 - IZ / 4 + ZI / 4 - ZZ / 4)
     Ham(3, 10) = |1100000000><1100000000|
     Ham(1, 3) = |100><100|
     Ham(7, 5) = |11100><11100|
     Ham(5, 3) = |101><101|

     Hamiltonian of computational basis
    
    '''
    a, G, m = M(n, l)
    res = a[0] * QubitOperator(G[0])
    for i in range(1, len(a)):
        res += a[i] * QubitOperator(G[i])
    return Hamiltonian(res)

from mindquantum.algorithm import HardwareEfficientAnsatz
encoder, solver = encoding(images_origin[0])
encoder = encoder.no_grad()
ansatz = get_ansatz(8, 1) 
circuit = encoder + ansatz
#encoder.summary()
#ansatz.summary()
#circuit.summary()
#-----------------------------------------------------------------------------------------------------------------------------------

from mindquantum.framework import MQLayer
from mindspore import ops, Tensor
import mindspore as ms

sim = Simulator('projectq', 8)
circuit.summary()

hams = [Ham(i, 1) for i in range(2)] # hams = [|0><0|, |1><1|]

grad_ops = sim.get_expectation_with_grad(hams, circuit, circuit, encoder.params_name, ansatz.params_name, parallel_worker=5)

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)

QuantumNet = MQLayer(grad_ops)
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Adam, Accuracy
from mindspore import Model
from mindspore.dataset import NumpySlicesDataset
from mindspore.train.callback import Callback, LossMonitor

loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opti = Adam(QuantumNet.trainable_params(), learning_rate=0.01)

model = Model(QuantumNet, loss, opti, metrics={'Acc':Accuracy()})
train_loader = NumpySlicesDataset({'features':images_filtered[:100], 'labels': labels_filtered[:100]}, shuffle=False).batch(5)
test_loader = NumpySlicesDataset({'features':images_filtered[100:110], 'labels':labels_filtered[100:110]}).batch(5)
class StepAcc(Callback):
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []
    def step_end(self, run_context):
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])

monitor = LossMonitor(5)
weights = np.load('weights.npy')   # Already trained parameters' values



# ------------------------------------------!For training:!---------------------------------------------------
#acc = StepAcc(model, test_loader)
#model.train(10, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)

#np.save('weights.npy', QuantumNet.weight.asnumpy()) # to save the trained parameters's values
#-------------------------------------------------------------------------------------------------------------



def predict(data, weight): # data for encoder and weight for ansatz
    dic = {}
    for i in range(len(data)):
        dic[f'alpha{i}'] = data[i]
    for i in range(len(weight)):
        dic[f'beta{i}'] = weight[i]
    sim = Simulator('projectq', 8)
    sim.apply_circuit(circuit, ParameterResolver(dic))
    if np.real(sim.get_expectation(hams[0])) > np.real(sim.get_expectation(hams[1])):
        return 0
    else:
        return 1

predicts = []
for i in range(1000, 1020):
    predicts.append(predict(images_filtered[i], weights))   
print('predicts:', predicts)
print('realnums:', labels_filtered[1000 :1020])
t = 0
for i in range(len(images_origin)):
    if labels[i] <= 1:
        t += 1
        if t >= 1000:
            plt.subplot(4, 5, t - 1000 + 1)
            plt.imshow(images_origin[i].reshape(16, 16), cmap = 'gray')
        if t == 1019:
            break
plt.colorbar()
plt.show()


