# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:16:14 2023

@author: Farehan Yahya
"""
# TD2: Calcul approché d'intégrales

# Le calcul d'intégrales nécessite en général de trouver des primitives, ce qui est parfois très difficile.
# Dans ce TD, on verra comment calculer des intégrales de manière approchée, sans supposer être capable de calculer la moindre primitive



#%% Méthode des rectangles:

# La façon la plus simple d'approcher une intégrale est de faire une somme de Riemann:


#             b             b-a   n-1
# On approche ∫f(x) dx par  ───    ∑ f(xₖ) où xₖ=a+(b-a)k/n.
#             a              n    k=0

# On appelle ce procédé la "méthode des rectangles", car elle revient à approcher, sur chaque intervalle [xₖ,xₖ₊₁[, la fonction f par f(xₖ) et donc l'intégrale 

# xₖ₊₁
#  ∫  f(x) dx par le rectangle de hauteur f(xₖ).
#  xₖ
#




def integrale_rectangles(f,a,b,n):
    "integrale_rectangles(f,a,b,n) calcule l'intégrale de f sur [a,b], par la méthode des rectangle, en divisant l'intervale en n sous-intervalles"
    R=0
    for k in range(n):
        R=R+f(a+(b-a)*k/n)*(b-a)/n
    return R


""" Questions:
* Taper integrale_rectangles? dans la console pour obtenir de l'aide sur la fonction integrale_rectangles
                                                       1                   1
* Utiliser cet algorithme pour avoir une estimation de ∫sin(x)/x dx, et de ∫√(1-x²) dx.
                                                      -1                   0

 Remarque: pour accéder aux fonctions sin et sqrt (et au nombre π dont vous aurez besoin dans la suite), il vous faudra tout d'abord avoir exécuté  from math import sin, sqrt, pi  
                             1                             
* Dans le cas particulier de ∫√(1-x²) dx quelle est la valeur exacte de l'intégrale?
                             0 
"""


#%% Calcul de l'erreur


# Pour l'intégrale
# 1
# ∫√(1-x²) dx,
# 0
#dont on connait la valeur exacte, on peut calculer l'erreur (c'est à dire la valeur absolue de la différence entre la valeure exacte de l'integrale et la valeur approchée obtenue par la méthode des rectangles), afin de la représenter graphiquement.
#
#


""" Définir une fonction qui calcule l'erreur commise quand on estime cette intégrale par la méthode des rectangles pour une valeur arbitraire de n :"""

def erreur_rectangles(n):
    ...........


""" À vous de compléter les ..... avec les instructions pertinentes. """



#%% Représentation graphique de l'erreur
# ci dessous, on calcule l'erreur pour n∈{10,20,30,...,350}, puis on les représente sur une échelle logarithmique (à la fois en abscisses et en ordonnées)

valeurs_de_n=range(10,351,10)
from matplotlib import pyplot
pyplot.close()
pyplot.plot(valeurs_de_n,[erreur_rectangles(n) for n in valeurs_de_n],"g.")
pyplot.loglog()
pyplot.show()

""" Remarquer que l'erreur est proche de K×nᵅ, pour une certaine constante K et une certaine puissance α.
  Estimer la valeur de K et de α."""



#%% Méthode des trapèzes:

# Une méthode un peu plus élaborée consiste à approcher par des trapèzes:


#             b             b-a   n-1  f(xₖ)+f(xₖ₊₁)
# on approche ∫f(x) dx par  ───    ∑  ────────────────  où xₖ=a+(b-a)k/n
#             a              n    k=0       2


"""
Questions:
1. Faire un dessin et constater que cette méthode revient à approcher, sur chaque intervalle [xₖ,xₖ₊₁[, la fonction f par un segment de droite reliant les points de coordonnées respectives (xₖ,f(xₖ)) et (xₖ₊₁,f(xₖ₊₁))
2. Définir une fonction integrale_trapezes, analogue à integrale_rectangles mais qui approche l'intégrale par la méthode des trapèzes
                    1
3. Pour l'intégrale ∫√(1-x²) dx,
                    0
représenter graphiquement, en fonction de n, l'erreur de calcul de l'intégrale par la méthode des trapèzes et par celle des rectangle. Laquelle est plus précise? Déterminer des constantes K et α telles que pour la méthode des trapèzes, l'erreur soit proche de K×nᵅ"""

def integrale_trapezes(f,a,b,n):
    ...........




#%% Méthode de Simpson
# Désormais, sur chaque intervalle [xₖ,xₖ₊₁[, on approche la fonction f par un arc de parabole passant par les points de coordonnées (xₖ,f(xₖ)), ((xₖ+xₖ₊₁)/2,f((xₖ+xₖ₊₁)/2)) et (xₖ₊₁,f(xₖ₊₁))

""" Montrer qu'il existe une unique polynôme Pₖ de degré (inférieur ou égal à) 2 qui satisfait
 ⎧  Pₖ(xₖ)=f(xₖ)
 ⎨  Pₖ((xₖ+xₖ₊₁)/2)=f((xₖ+xₖ₊₁)/2)
 ⎩  Pₖ(xₖ₊₁)=f(xₖ₊₁)
 
 Calculer (en fonction de f(xₖ), de f((xₖ+xₖ₊₁)/2) et de f(xₖ₊₁)), l'intégrale de Pₖ sur [xₖ,xₖ₊₁[
                                                                 n-1   xₖ₊₁
 Définir en conséquence la fonction integrale_simpson qui calcule ∑     ∫P(x) dx
                                                                 k=0    xₖ
"""


def integrale_simpson(f,a,b,n):
    .....................






#%% Représentation graphique
"""
 Comparer graphiquement la vitesse de convergence de la méthode de Simpson avec celle des deux méthodes précédentes.
 
 Estimer des constantes K et α telles que pour la méthode de Simpson, l'erreur soit proche de K×nᵅ
 
"""




#%% Représentation graphique pour sin x/x
"""
Comment pourrait-on procéder pour estimer l'erreur dans le calcul de 
 1
 ∫sin(x)/x dx   ?
-1

Représenter graphiquement l'erreur de la méthode des trapèzes et de celle de Simpson. Estimer dans chacun de ces deux cas des constantes K et α telles que l'erreur soit proche de K×nᵅ.

Qu'est-ce qui peut expliquer que la vitesse de convergence soit si différente de celle obtenue pour ∫√(1-x²) dx ?

"""

