import numpy as np
import matplotlib.pyplot as plt
from bokeh.layouts import layout
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
data, target = housing.data, housing.target
print(data.shape,target.shape, housing.feature_names, np.max(target))
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=1)
scaler = StandardScaler()
scaler.fit(data_train)
data_train_scal = scaler.transform(data_train)
data_test_scal = scaler.transform(data_test)
data_train_scal_bias = np.c_[np.ones(len(data_train_scal)), data_train_scal]
data_test_scal_bias = np.c_[np.ones(len(data_test_scal)), data_test_scal]
print(data_train_scal_bias.shape,data_test_scal_bias.shape)

def gd(X, Y, ad=0, eta=0.005, fres=1e-6):
    losshis = []
    whis = []
    gradhis = []
    w = np.zeros(X.shape[1])
    loss = np.mean((X @ w - Y)**2)
    losshis.append(loss)
    whis.append(w)
    grad = (2/len(Y))*(X.T @ (X @ w - Y))
    gradhis.append(grad)
    if ad == 0:
        w = w - eta * grad
    else:
        w = w - eta * grad / (np.sqrt(np.sum(np.array(gradhis)**2)+1e-8))
    loss_new = np.mean((X @ w - Y)**2)
    losshis.append(loss_new)
    whis.append(w)
    while np.abs(loss_new - loss) > fres:
        grad = (2/len(Y))*(X.T @ (X @ w - Y))
        gradhis.append(grad)
        if ad == 0:
            w = w - eta * grad
        else:
            w = w - eta * grad / (np.sqrt(np.sum(np.array(gradhis)**2)+1e-8))
        loss = loss_new
        loss_new = np.mean((X @ w - Y)**2)
        losshis.append(loss_new)
        whis.append(w)
    return w, np.array(losshis), np.array(whis)

def sgd(X, Y, size=1, ad=0, eta=0.005, fres=1e-6):
    losshis = []
    whis = []
    gradhis = []
    w = np.zeros(X.shape[1])
    loss = np.mean((X @ w - Y)**2)
    losshis.append(loss)
    whis.append(w)
    lo = (X @ w - Y)
    index = np.random.choice(np.arange(len(Y)), size=size, replace=False)
    grad = 0
    for i in index:
        grad += (2/len(index))*(X[i] * lo[i])
    gradhis.append(grad)
    if ad == 0:
        w = w - eta * grad
    else:
        w = w - eta * grad / (np.sqrt(np.sum(np.array(gradhis)**2)+1e-8))
    loss_new = np.mean((X @ w - Y) ** 2)
    losshis.append(loss_new)
    whis.append(w)
    while np.abs(loss_new - loss) > fres:
        lo = (X @ w - Y)
        index = np.random.choice(np.arange(len(Y)), size=size, replace=False)
        grad = 0
        for i in index:
            grad += (2 / len(index)) * (X[i] * lo[i])
        gradhis.append(grad)
        if ad == 0:
            w = w - eta * grad
        else:
            w = w - eta * grad / (np.sqrt(np.sum(np.array(gradhis)**2)+1e-8))
        loss = loss_new
        loss_new = np.mean((X @ w - Y) ** 2)
        losshis.append(loss_new)
        whis.append(w)
    return w, np.array(losshis), np.array(whis)

def visual(w, losshis, whis, savename=''):
    fig, axes = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    axes[0].plot(np.arange(len(losshis)), losshis)
    axes[0].set_xlabel('Iteration', fontsize='x-large',weight='bold')
    axes[0].set_ylabel('Loss', fontsize='x-large',weight='bold')
    axes[0].tick_params(labelsize='large',which='major',length=5, width=3)
    def L(w1,w2, X=data_train_scal_bias, Y=target_train):
        w12 = w
        w12[1] = w1
        w12[2] = w2
        return np.mean((X @ w12 - Y)**2)
    W1 = np.linspace(-1, 2.5, 100)
    W2 = np.linspace(-1.5, 1.5, 100)
    Z = np.zeros((len(W2), len(W1)))
    for j in range(len(W2)):
        for i in range(len(W1)):
            Z[j, i] = L(W1[i], W2[j])
    cm = axes[1].contourf(W1, W2, Z, levels=100, cmap='viridis')
    cbar = fig.colorbar(cm)
    cbar.set_label('Loss')
    axes[1].set_xlabel('$w_1$ (weight of MedInc)', fontsize='x-large', weight='bold')
    axes[1].set_ylabel('$w_2$ (weight of HouseAge)', fontsize='x-large', weight='bold')
    axes[1].tick_params(labelsize='large', which='major', length=5, width=3)
    axes[1].scatter(whis.T[1], whis.T[2])
    for i in range(len(whis.T[1]) - 1):
        dx = whis.T[1][i + 1] - whis.T[1][i]
        dy = whis.T[2][i + 1] - whis.T[2][i]
        axes[1].arrow(whis.T[1][i], whis.T[2][i], dx, dy)
    axes[2].scatter(np.arange(len(target_test)), target_test, marker='o', color='black', s=0.2 ,label='Test')
    target_pred = np.zeros_like(target_test)
    for i in range(len(target_pred)):
        target_pred[i] = data_test_scal_bias[i] @ w
    axes[2].scatter(np.arange(len(target_pred)), target_pred, marker='d', color='lime', s=0.2, label='Prediction')
    axes[2].legend(loc='best',fontsize='large')
    axes[2].set_xlabel('Test Sets', fontsize='x-large', weight='bold')
    axes[2].set_ylabel('Median House Value ($100,000)', fontsize='x-large', weight='bold')
    axes[2].tick_params(labelsize='large', which='major', length=5, width=3)
    plt.show()
    if savename != '':
        fig.savefig(savename, format='png', dpi=600, bbox_inches='tight', pad_inches=0.2)

w_gd, losshis_gd, whis_gd = gd(data_train_scal_bias, target_train)
w_sgd, losshis_sgd, whis_sgd = sgd(data_train_scal_bias, target_train, fres=7e-7)
w_msgd, losshis_msgd, whis_msgd = sgd(data_test_scal_bias, target_test, size=10)
w_gd_ad, losshis_gd_ad, whis_gd_ad = gd(data_train_scal_bias, target_train, ad=1, eta=0.1)
w_sgd_ad, losshis_sgd_ad, whis_sgd_ad = sgd(data_train_scal_bias, target_train, ad=1, eta=0.1, fres=5e-9)
w_msgd_ad, losshis_msgd_ad, whis_msgd_ad = sgd(data_test_scal_bias, target_test, size=10, ad=1, eta=0.1, fres=5e-7)

visual(w_gd, losshis_gd, whis_gd, savename='gd.png')
visual(w_sgd, losshis_sgd, whis_sgd, savename='sgd.png')
visual(w_msgd, losshis_msgd, whis_msgd, savename='msgd.png')
visual(w_gd_ad, losshis_gd_ad, whis_gd_ad, savename='gd_ad.png')
visual(w_sgd_ad, losshis_sgd_ad, whis_sgd_ad, savename='sgd_ad.png')
visual(w_msgd_ad, losshis_msgd_ad, whis_msgd_ad, savename='msgd_ad.png')
