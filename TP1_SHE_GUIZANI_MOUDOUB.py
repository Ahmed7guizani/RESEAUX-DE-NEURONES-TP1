# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:14:29 2021

@author: User
"""
# 1 Analyse des données
import numpy as np
data = np.loadtxt('C:/Users/ahmed/Documents/2020-2021/S2/nu/dataset.dat')
print(data)

X = data[:,0:2]
y = data[:,2]
y = y.astype(int)



#visualiser les données
from matplotlib import pyplot
colors= np.array([x for x in "rgbcmyk"])
pyplot.scatter(X[:, 0], X[:, 1], color=colors[y].tolist(), s=10)
pyplot.show()

#partition des données
from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=42)

print('la dimension Dim X = ',X.ndim)
print('la dimension Dim y = ',y.ndim)


print("Le nombre d'exemple d'apprentissage = " ,y_train.size)
print("Le nombre d'exemple test = " ,y_test.size)

# 2 Algorithme du plus-proche-voisin

from sklearn.neighbors import KNeighborsClassifier
one_NN= KNeighborsClassifier(n_neighbors=1, algorithm='brute')
one_NN.fit(X_train, y_train)



print("le taux de reconnaissance sur les bases d’apprentissage : ", one_NN.score(X_train, y_train))

print("le taux de reconnaissance sur les bases de test : ", one_NN.score(X_test, y_test))

# matrice de confusion
from sklearn import metrics
y_pred_test= one_NN.predict(X_test)
metrics.confusion_matrix(y_test, y_pred_test)

print("La matrice de confusion : \n", metrics.confusion_matrix(y_test, y_pred_test))

# les frontières de décision définies par les données d’apprentissage

x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1 
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1 
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h), 
                     np.arange(y_min, y_max, y_h))
Y = one_NN.predict(np.c_[xx.ravel(), yy.ravel()]) 
Y = Y.reshape(xx.shape)


pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_train[:, 0], 
               X_train[:, 1], 
               cmap=pyplot.cm.Paired, 
               color=colors[y_train].tolist()
              ) 
pyplot.xlim(xx.min(), xx.max())
pyplot.ylim(yy.min(), yy.max())
pyplot.show()

# les frontières de décisions et les données de test

x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1 
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1 
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h), 
                     np.arange(y_min, y_max, y_h))
Y = one_NN.predict(np.c_[xx.ravel(), yy.ravel()]) 
Y = Y.reshape(xx.shape)


pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_test[:, 0], 
               X_test[:, 1], 
               cmap=pyplot.cm.Paired, 
               color=colors[y_test].tolist()
              ) 
pyplot.xlim(xx.min(), xx.max())
pyplot.ylim(yy.min(), yy.max())
pyplot.show()

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_test, y_test)
clf.get_params()
result = clf.predict(X_test)

errors = sum(result != y_test) 
print("les erreurs de classification:", errors)





# 3 a Analyse du fonctionnement de l’algorithme
max_size_train = X_train.shape[0] 
acc = []
nb_apprentissage = []


for size in range (1,100):
    X_train1, temp1, y_train1, temp2 = model_selection.train_test_split(X_train,y_train,train_size = size/max_size_train, random_state=42) 
    KNN= KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    KNN.fit(X_train1, y_train1)
    acc.append(KNN.score(temp1, temp2))
    nb_apprentissage.append(max_size_train*size/100)


pyplot.plot(nb_apprentissage, acc)
pyplot.xlabel('Le nombre d’exemples d’apprentissage')
pyplot.ylabel('Le taux de reconnaissance')
pyplot.title("Le graphe du taux de reconnaissance en fonction du nombre d’exemples d’apprentissage")
pyplot.grid()
pyplot.show()


# 3b

max_size_test = X_test.shape[0] 
acc_test = []
nb_test = []

for size in range (1,90):
    X_train1, X_test1, y_train1, y_test1 = model_selection.train_test_split(X_test,y_test,test_size = size/max_size_test, random_state=42) 
    KNN= KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    KNN.fit(X_train, y_train)
    acc_test.append(KNN.score(X_test1, y_test1))
    nb_test.append(max_size_test*size/90)


pyplot.plot(nb_test, acc_test)
pyplot.xlabel('Le nombre d’exemples de test')
pyplot.ylabel('Le taux de reconnaissance')
pyplot.title("Le graphe du taux de reconnaissance en fonction du nombre d’exemples de test")
pyplot.grid()
pyplot.show()


#4a
a = []
nb = []
for k in range (1,11):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
    knn= KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train ,y_train)
    a.append(knn.score(X_test, y_test))
    nb.append(k)
    print('the numéro de neighbors:', k)
    print('taux de reconnaissance en training set:', knn.score(X_test, y_test))
pyplot.plot(nb, a)
pyplot.xlabel('k')
pyplot.ylabel('Le taux de reconnaissance')
pyplot.title("Le graphe du taux de reconnaissance en fonction de k")
pyplot.grid()
pyplot.show()
    
# 4 b

k_range = range (1, 101)
taux_reconnaissance = []

for k in k_range:
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=42)
    knn= KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    taux_reconnaissance.append(score)

pyplot.plot(k_range, taux_reconnaissance)

taux_max=np.argmax(taux_reconnaissance)
print("k optimal pour taux de reconnaissance max en test = ", taux_max + 1)
show_max='['+str(taux_max + 1)+'; '+str(taux_reconnaissance[taux_max].round(2))+']'

pyplot.plot(taux_max + 1,taux_reconnaissance[taux_max],'ko') 
pyplot.annotate(show_max,xy=(taux_max + 1,taux_reconnaissance[taux_max]),xytext=(taux_max + 1, taux_reconnaissance[taux_max]))



pyplot.xlabel('k')
pyplot.ylabel('Le taux de reconnaissance')
pyplot.title("Le graphe du taux de reconnaissance en fonction de k")
pyplot.grid()
pyplot.show()

# 4 c
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=42)
k_etoile = taux_max + 1
knn= KNeighborsClassifier(n_neighbors = k_etoile)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)




x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1 
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1 
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h), 
                     np.arange(y_min, y_max, y_h))
Y = knn.predict(np.c_[xx.ravel(), yy.ravel()]) 
Y = Y.reshape(xx.shape)


pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_test[:, 0], 
               X_test[:, 1], 
               cmap=pyplot.cm.Paired, 
               color=colors[y_test].tolist()
              ) 
pyplot.xlim(xx.min(), xx.max())
pyplot.ylim(yy.min(), yy.max())
pyplot.show()

k_max = 100
knn= KNeighborsClassifier(n_neighbors = k_max)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)

x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1 
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1 
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h), 
                     np.arange(y_min, y_max, y_h))
Y = knn.predict(np.c_[xx.ravel(), yy.ravel()]) 
Y = Y.reshape(xx.shape)


pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_test[:, 0], 
               X_test[:, 1], 
               cmap=pyplot.cm.Paired, 
               color=colors[y_test].tolist()
              ) 
pyplot.xlim(xx.min(), xx.max())
pyplot.ylim(yy.min(), yy.max())
pyplot.show()


k_data = [1, k_etoile, k_max]


k_data = LinearRegression()
#estimate bias and variance
mse,bias, var = bias_variance_decomp(k_data, X_train, y_train, X_test, y_test,loss='mse', num_rounds=200, random_seed=1)
#summarize results
print('MSE:%.3f' % mse)
print('Bias:%.3f' % bias)
print('Variance:%.3f' % var)


# 4 d 

k_range = range (1, 101)
taux_reconnaissance = []

for k in k_range:
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=42)
    knn= KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    score = knn.score(X_train, y_train)
    taux_reconnaissance.append(score)

pyplot.plot(k_range, taux_reconnaissance)

taux_max=np.argmax(taux_reconnaissance)
print("k optimal pour taux de reconnaissance max en apprentissage = ",taux_max + 1)
show_max='['+str(taux_max + 1)+'; '+str(taux_reconnaissance[taux_max].round(2))+']'

pyplot.plot(taux_max + 1,taux_reconnaissance[taux_max],'ko') 
pyplot.annotate(show_max,xy=(taux_max + 1,taux_reconnaissance[taux_max]),xytext=(taux_max + 1, taux_reconnaissance[taux_max]))



pyplot.xlabel('k')
pyplot.ylabel('Le taux de reconnaissance')
pyplot.title("Le graphe du taux de reconnaissance en fonction de k")
pyplot.grid()
pyplot.show()


