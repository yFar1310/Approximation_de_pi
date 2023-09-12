#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:00:43 2023

@author: yf885449
"""

"Indiquez ici vos noms et prÃ©noms afin de rendre ce document complÃ©tÃ©:"" FAREHAN Yahya  et ABOUNAIM YASSINE "


#%%Prise en main
#Ce fichier est divisÃ© en blocs de code apelÃ©s "cellules", et dÃ©limitÃ©s par les lignes qui commencent par #%%
#Les touches Ctrl+EntrÃ©e permettent d'executer toute une cellule Ã  la fois.
#Par exemple constatez ce qu'il se passe lorsque vous executez cette cellule, et la suivante.
#Comparez Ã  ce qu'il se passe lorsque vous entrez directement les commandes dans la console


x=1+1
print(x**2)
"Bonjour"
#%% cette ligne est le dÃ©but de la cellule suivante
1+1
print(3**2)
print("Bonjour")


# Remarque sur l'affichage dans la console:
#Dans pyzo, vous constatez que quand on execute une cellulle, le code est executÃ© mais aucun rÃ©sultat n'est affichÃ©, sauf si des fonctions comme "print" sont utilisÃ©es.
#De plus, toutes les lignes qui commencent par "#" sont considÃ©rÃ©es comme des commentaires, et ignorÃ©es quand le code est executÃ©.


#%% Type de donnÃ©es
print("Bonjour "+3*"!")
print(2+3*9)
#%% listes
[17]+[2]
#%% nombres Ã  virgule de type float
a=2+10**-5
print("a est de type ",type(a)," et vaut ",a)
#%%erreur de calcul avec le nombres dÃ©cimaux
print(2+10**-20-2)
#%% utilisation du module decimal
from decimal import Decimal
print(Decimal(2)+Decimal(10)**-20)
print(Decimal(2)+Decimal(10)**-20-2)
#%%configuration du module decimal
from decimal import getcontext, setcontext, Context
setcontext(Context(prec=120))
#%% Dichotomie: code fourni avec l'Ã©noncÃ©

def f(x):
    return x**2-2
x=[1.0];y=[2.0]
gn=[Decimal(1.0),Decimal(2.0)]
for n in range(0,10):
    if f((x[n]+y[n])/2)>0:
        x=x+[x[n]]
        y=y+[(x[n]+y[n])/2]
    else:
        x=x+[(x[n]+y[n])/2]
        y=y+[y[n]]
    gn=gn+[abs((x[n]**2-2)/2*x[n])]
print("x[10]=",x[10])
print("y[10]=",y[10])
print(gn)

#%% ReprÃ©sentation graphique

from matplotlib import pyplot

pyplot.plot(range(11),x,"g+")
pyplot.plot(range(11),y,"r.")
pyplot.show()

#%% Exercice: Calculez les 10 premiÃ¨res dÃ©cimales de racine de deux 
# une fois trouvÃ© un moyen de calculer ces 10 premiÃ¨res dÃ©cimales, Ã©crivez ici le code correspondant
from decimal import Decimal
def g(x):
    return x**2-2
x=[1.0];y=[2.0]

for n in range(0,10):
    if g((x[n]+y[n])/2)>0:
        x=x+[Decimal(x[n])]
        y=y+[(Decimal(x[n])+Decimal(y[n]))/2]
    else:
        x=x+[(Decimal(x[n])+Decimal(y[n]))/2]
        y=y+[Decimal(y[n])]
print("x[10]=",x[10])
print("y[10]=",y[10])

#%% Exercice: Calculer les 100 premiÃ¨res dÃ©cimales de racine de deux
from decimal import Decimal
def h(x):
    return x**2-2
x=[1.0];y=[2.0]
for n in range(0,100):
    if g((x[n]+y[n])/2)>0:
        x=x+[Decimal(x[n])]
        y=y+[(Decimal(x[n])+Decimal(y[n]))/2]
    else:
        x=x+[(Decimal(x[n])+Decimal(y[n]))/2]
        y=y+[Decimal(y[n])]
print("x[10]=",x[100])
print("y[10]=",y[100])


#%% Exercice: Calculez les dix premiÃ¨res dÃ©cimales de la racine poisitive du polynÃ´me X**4+3*X**3+12*X-2018
from decimal import Decimal
def e(x):
    return x**4+3*x**3+12*x-2018
x=[0.0];y=[7.0]
for n in range(0,10):
    if e((x[n]+y[n])/2)>0:
        x=x+[Decimal(x[n])]
        y=y+[(Decimal(x[n])+Decimal(y[n]))/2]
    else:
        x=x+[(Decimal(x[n])+Decimal(y[n]))/2]
        y=y+[Decimal(y[n])]
print("x[10]=",x[10])
print("y[10]=",y[10])

#%%
x,y=1,2
while str(x)[:12]!=str(y)[:12]:
    m=(x+y)/2
    if f(m)>0:x,y=x,m
    else: x,y=m,y
    print("[",x,",",y,"]\n")
print(str(x)[:12])
print("x=",x)

#%% MÃ©thode de la sÃ©cante : code fourni avec l'Ã©noncÃ©
def f(x):return x**2-2
x=[1.0,2.0]
fx=[f(1.0),f(2.0)]
for n in range (1,5):
    x=x+[x[n]-(x[n]-x[n-1])/(fx[n]-fx[n-1])*fx[n]]
    fx=fx+[f(x[n+1])]
print(x)

#%% Exercice: calcul des 15 premiÃ¨res valeurs
from decimal import Decimal ,getcontext, setcontext, Context
setcontext(Context(prec=300))
def f(x):return x**2-2
x=[Decimal(1.0),Decimal(2.0)]
fx=[Decimal(f(1.0)),Decimal(f(2.0))]
gn2=[Decimal(1.0),Decimal(2.0)]
for n in range (1,15):
    x=x+[x[n]-(x[n]-x[n-1])/(fx[n]-fx[n-1])*fx[n]]
    fx=fx+[f(x[n+1])]
    gn2=gn2+[abs((x[n]**2-2)/2*x[n])]
print(gn2)

#%% Exercice: reprÃ©sentation graphique de la marge d'erreur
from matplotlib import pyplot
pyplot.plot(range(len(gn)),gn,"g+")
pyplot.show()



#%% Vitesse de convergence
from matplotlib import pyplot
pyplot.plot(range(len(gn)),gn,"g+")
pyplot.plot(range(len(gn2)),gn2,"r.")
pyplot.show()




#%% MÃ©thode de Newton
from decimal import Decimal,getcontext, setcontext, Context
setcontext(Context(prec=300))
def g(x):return (x**2-2)/x*2
x=[Decimal(2.0),Decimal(4.0)]
gx=[g(2.0),g(4.0)]
gn3=[Decimal(1.0),Decimal(2.0)]
for n in range (1,10):
    x=x+[x[n]-g(x[n])]
    gx=gx+[g(x[n+1])]
    gn3=gn3+[abs((x[n]**2-2)/2*x[n])]
print(gn3)



#%% ReprÃ©sentation graphique de la marge d'erreur
from matplotlib import pyplot
pyplot.plot(range(len(gn)),gn,"g+")
pyplot.plot(range(len(gn2)),gn2,"r.")
pyplot.plot(range(len(gn3)),gn3,"bx")
pyplot.show()
