# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:06:10 2018

@author: farehan yahya
"""

#%% On importe les modules NumPy et Scipy
import numpy as np
from scipy import linalg as LA

#Nous pouvons introduire une matrice comme un array
A = np.array([[1,2], [3,4]])
print(A)
print(A.shape) ##nous donne la taille de la matrice
print("")

#%%
#On introduit un vector et une nouvelle matrice,
B =  np.array([[5,6], [7,8]])
print(B)
print("")
b = np.array([[1],[2]])
print(b)
print(b.shape)
print("")
#on peut definir des produits matricielles de la mani`ere
#suivante
AB = np.dot(A,B)
print(AB)
Ab = np.dot(A,b)
print(Ab)

#%% matrix
#Une approche alternative en NumPy, plus adaptée à la notation
#mathématique, utilise la notion de matrix
A_M = np.matrix([[1,2], [3,4]])
print(A_M)
print(A_M.shape)
print("")
B_M =  np.matrix([[5,6], [7,8]])
print(B_M)
print("")
b_M = np.matrix([[1],[2]])
print(b_M.shape)
print(b_M)
print("")
#on peut definir des produits matricielles de la mani`ere
#suivante
AB_M = A_M*B_M
print(AB_M)
Ab_M = A_M*b_M
print(Ab_M)


#%% calcul d'inverses
AI = LA.inv(A)
AI_M = LA.inv(A_M)
print(AI)
print(AI_M)
#%% calcul de transposées
AT = np.transpose(A)
AT_M = np.transpose(A_M)
print("")
print(AT)
print(AT_M)




#%%Exercice 1 : 
#
#i) Calculer les déterminants, transposé, trace de la matrice A
#
import numpy as np
from scipy import linalg as LA

A = np.matrix([[1,2],[2,1]])
detA = np.linalg.det(A)
print(detA)
transA = A.transpose()
print(transA)
trA = np.trace(A)
print(trA)

#ii) Définir une matrice A (inversible) et un vecteur b et résoudre Ax=b.
#Vérifier le calcul en calculant A^{-1}b
b=np.array([[1],[5]])
eq = np.linalg.solve(A,b)
print(eq)
print((LA.inv(A)).dot(b))


#%% Exercice 2 : Visualisation d'une matrice
# Construire une fonction " visualisation_matrice(A) " en utilisant comme
# point de départ les lignes suivantes

from matplotlib import pyplot as plt
plt.close()
plt.imshow(A, interpolation='nearest', cmap=plt.cm.gray_r)
plt.colorbar()
plt.show()

def visualisation_matrice(A):
 plt.imshow(A, interpolation='nearest', cmap=plt.cm.ocean)
 plt.colorbar()
 plt.show()

#%% Appliquer cette fonction à la matrice aléatoire de taille ndim x ndim
ndim = 25
A= np.random.rand(ndim,ndim)
visualisation_matrice(A)

#%%Calcul de valeurs propres et vecteurs propres avec python
A = np.array([[1, 2], [3, 4]])
valeurs_propres, vecteurs_propres = LA.eig(A)

print(valeurs_propres)
print(vecteurs_propres)
  
#%% Exercice 3 : Construire une fonction python qui détermine si une matrice est diagonalisable

def test_diagonalisable(A, epsilon):
    VP , AP = LA.eig(A)
    if(abs(LA.det(AP))<epsilon) :
        return 0,LA.det(AP)
    else:
        return 1,LA.det(AP)
print(test_diagonalisable(A,1e-5))
#%% Exercice 4 : Application a la matrice
## Construction de la matrice
# ⎛ 0 1 0   ... 0 ⎞
# ⎜ 0 0 1   ... 0 ⎟
# ⎜ 0 0 0 1 ... 0 ⎟
# ⎜      ...      ⎟
# ⎜ 0 0 0 0 ... 1 ⎟
# ⎝ 0 0 0 0 ... 0 ⎠
#Montrer, de manière mathématique et en utilisant la fonction "test_diagonalisable",
#que cette matrice n'est pas diagonalisable.
# Définition de la matrice
n = 5
B = np.zeros((n,n))
print(B)
for i in range(n-1):
    B[i,i+1] = 1
print(B)
# Test de diagonalisabilité
test, det_AP = test_diagonalisable(B, 1e-5)
print(test_diagonalisable(B, 1e-5))
if test == 1:
    print("La matrice est diagonalisable avec déterminant AP =", det_AP)
else:
    print("La matrice n'est pas diagonalisable avec déterminant AP =", det_AP)
#%% Exercice 5 : Matrices du Laplacien 1-D: répondre aux questions dans l'exercice 5.
import numpy as np
from scipy import linalg as LA
# i) Définir une fonction en python qui construit la matrice ∆_n.

def delta_n(n):
    delta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                delta[i,j]=2
            elif abs(i - j) == 1:
                delta[i,j] = -1
            else:
                delta[i, j] = 0
    return delta
def laplacian_matrix(n):
    """
    Construire une matrice du Laplacien 1-D de taille n x n.
    """
    diagonale = np.ones(n-1)*(-2)
    sub_diagonale = np.ones(n-2)
    D = np.diag(diagonale) + np.diag(sub_diagonale, -1) + np.diag(sub_diagonale, 1)
    return D
# ii) Pour une matrice ∆_n, avec un choix de $n$ arbitraire,
def est_diagonalisable(A):
    """
    Vérifier si une matrice A est diagonalisable en vérifiant si toutes ses valeurs propres sont distinctes.
    """
    eigenvaleurs = LA.eigvals(A)
    if len(set(eigenvaleurs)) == len(eigenvaleurs):
        return True
    else:
        return False

n = 10
laplacian = laplacian_matrix(n)
print(est_diagonalisable(laplacian))
#      déterminer avec la fonction construite en python si elle est diagonalisable.
#iii) Afficher la structure qualitative de la matrice  $\Delta_n$ avec la fonction
import matplotlib.pyplot as plt

def visualisation_matrix(A):
    """
    Afficher la structure qualitative de la matrice A.
    """
    plt.imshow(A, cmap='gray', interpolation='nearest')
    plt.title('Matrice')
    plt.show()

n = 10
laplacian = laplacian_matrix(n)
visualisation_matrix(laplacian)
#     visualisation_matrice(A).
#iv) Calculer et afficher les valeurs propres de ∆_n.
n = 10
laplacian = laplacian_matrix(n)
eigenvalues = np.linalg.eigvals(laplacian)
print(eigenvalues)
#v) Montrer graphiquement a distribution de valeurs propres.

n = 10
laplacian = laplacian_matrix(n)
eigenvalues = np.linalg.eigvals(laplacian)
plt.hist(eigenvalues, bins=10)
plt.title('Distribution de valeurs propres')
plt.show()
#vi) Explorer l'effet sur la distribution de valeurs propre de l'addition
n = 10
laplacian = laplacian_matrix(n)
epsilon = 0.1
B = np.random.rand(n,n)*epsilon
laplacian_eps = laplacian + len(B)
eigenvalues_eps = np.linalg.eigvals(laplacian_eps)

plt.hist(eigenvalues, bins=10, alpha=0.5, label='Laplacian')
plt.hist(eigenvalues_eps, bins=10, alpha=0.5, label='Laplacian_eps')
plt.legend()
plt.title('Distribution de valeurs propres')
plt.show()
#      des matrices epsilon B, avec epsilon petit (EXERCICE D'EXPLORATION !)




#%% Exercice 6 : Méthode la puissance.
# 1. Construire une fonction qui rends la valeur propre avec plus grande valeur absolue
import numpy as np

def methode_puissance(A, epsilon=1e-5, max_iterations=1000):
    n = A.shape[0]
    x = np.ones(n)
    x /= LA.norm(x, np.inf)
    lambd = 0
    for i in range(max_iterations):
        x_new = np.dot(A, x)
        lambd_new = np.dot(x, x_new)
        if np.abs(lambd_new - lambd) < epsilon:
            break
        lambd = lambd_new
        x = x_new / LA.norm(x_new, np.inf)
    return lambd
#2. Comparer le résultat avec celui obtenu avec la fonction eig() pour une matrice aléatoire
#de taille 25 x 25
A = np.random.rand(25,25)

# calculate the eigenvalues of A using eig()

eigenvaleurs, _ = LA.eig(A)

# calculate the dominant eigenvalue of A using methode_puissance()

dominant_eigenvaleur = methode_puissance(A)

# print the results

print("Eigenvalues of A: ", eigenvaleurs)
print("Dominant eigenvalue of A: ", dominant_eigenvaleur)


#%% Exercice 7 : Decomposition QR
# Déterminer avec python la décomposition QR de la matrice ∆_7 .
#Montrer (numériquement que les vecteurs colonne de la matrice Q sont ortho-
#gonaux. Raisonner si les vecteurs lignes doivent être orthogonaux.
import numpy as np

delta_7 = np.array([[1, 2, 3],
                    [0, 1, 4],
                    [5, 6, 0]])

Q, R = np.linalg.qr(delta_7)

print("Matrice Q : ")
print(Q)

print("\nMatrice R : ")
print(R)

print("Produits scalaires des colonnes de Q : ")
print(Q[:,0] @ Q[:,1])
print(Q[:,0] @ Q[:,2])
print(Q[:,1] @ Q[:,2])
#%% Exercice 8 : Méthode QR pour le calcul de valeurs propres
# Répondre aux points dans l'exercice 8. 
import numpy as np

def diagonale_superieure_max(A):
    n = A.shape[0]
    max_val = 0
    for i in range(n):
        for j in range(i+1, n):
            max_val = max(max_val, abs(A[i,j]))
    return max_val


def QR(A, tol=1e-12, maxiter=1000):
    n = A.shape[0]
    Q = np.eye(n)
    for i in range(maxiter):
        Q, R = np.linalg.qr(A.dot(Q))
        A = R.dot(Q)
        if np.abs(A - np.diag(np.diag(A))).max() < tol:
            break
    return np.diag(A)

A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

# Calcul des valeurs propres avec la méthode QR
vals_QR = QR(A)

# Calcul des valeurs propres avec la fonction eig de NumPy
vals_eig, _ = LA.eig(A)

# Affichage des résultats
print("Méthode QR : ", vals_QR)
print("Fonction eig : ", vals_eig)
# Pour exporter la figure en pdf, utiliser
plt.close()
plt.imshow(A, interpolation='nearest', cmap=plt.cm.gray_r)
plt.colorbar()
plt.savefig("matrix_QR.pdf")







 


