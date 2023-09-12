# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:54:51 2023

@author: Farehan Yahya
"""

#%% Une figure simple. Version 1
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-np.pi,np.pi,101)
y=np.sin(x)+np.sin(3*x)/3.0

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple plot')

plt.show()
plt.savefig("simple_plot.pdf")

#%% Une figure simple. Version 2
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Simple plot")


plt.show()
fig.savefig("simple_plot.pdf")

#%% Figure multiple
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi,np.pi,101)  
f = np.ones_like(x)
f[x<0] = -1
y1=(4/np.pi)*(np.sin(x)+np.sin(3*x)/3.0)
y2=y1+(4/np.pi)*(np.sin(5*x)/5.0+np.sin(7*x)/7.0)
y3=y2+(4/np.pi)*(np.sin(9*x)/9.0+np.sin(11*x)/11.0)

fig = plt.figure()
ax= fig.add_subplot(111)  #111 est equivalent a 1,1,1 

ax.plot(x,f,'b-',lw=3,label='f(x)')
ax.plot(x,y1,'c--',lw=2,label='two terms')
ax.plot(x,y2,'r-.',lw=2,label='four terms')
ax.plot(x,y3,'b:',lw=2,label='six terms')
ax.legend(loc='best')
ax.set_xlabel('x',style='italic')
ax.set_ylabel('partial sums',style='italic')
fig.suptitle('Partial sums for Fourier series of f(x)', size=16,weight='bold')

fig.show()
fig.savefig("not_so_simple_plot.pdf")


#%% Repondre a l'exercice 1.
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -np.pi/2 * (x<0) + np.pi/2 * (x>=0)

def fn(x, n):
    s = 0
    for k in range(n+1):
        s += (-1)**k / (2*k+1) * np.sin((2*k+1)*x)
    return 2/np.pi * s

x = np.linspace(-np.pi, np.pi, 501)
n_list = [1, 3, 5, 10, 50, 100]

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, f(x), 'k-', lw=2, label=r'$f(x)$')
for n in n_list:
    ax.plot(x, fn(x, n), label=fr'$f_{n}(x)$, n={n}')
ax.legend(loc='best', fontsize=12)
ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_title(r'Convergence de $f_n(x)$ à $f(x)$', fontsize=16, fontweight='bold')
plt.show()


#%% EDO simple: Repondre a l'exercice 2.

# Paramètres
a = 1.0  # Paramètre 'a'
y0 = 1.0  # Condition initiale

# Solution analytique
def analytic_solution(t, a, y0):
    return y0 * np.exp(-a * t)

# Calcul de la solution analytique pour les temps t_arr
y_analytic = analytic_solution(t_arr, a, y0)

# Visualisation des résultats
plt.plot(t_arr, y, label='Numérique')
plt.plot(t_arr, y_analytic, '--', label='Analytique')
plt.xlabel('Temps (t)')
plt.ylabel('y(t)')
plt.title("Résolution numérique et analytique de l'équation différentielle ẏ + ay = 0")
plt.legend()
plt.show()


#%% Resolution num´erique d'une EDO simple 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs(Y,t,a):
    y = Y
    return -a*y

t_arr=np.linspace(0,10,101)
y_init=[1]
a=1.0
y_arr=odeint(rhs, y_init, t_arr, args=(a,))
y=y_arr[:,0]


#%% Repondre a l'exercice 3.
fig=plt.figure()
ax1=fig.add_subplot(211)
ax1.plot(t_arr,y,'b-',label='sol.numerique')
ax1.set_ylabel('y')
ax1.set_xlabel('x')

ycheck=np.exp(-a*t_arr)
ax1.plot(t-arr,ycheck,'r:',lable='sol. analytique')
ax1.legend(loc='best')

ax2=fig.add_subplot(212)
ax2.plot(t_arr,y-ycheck,lable='numerique - analytique')
ax2.set_ylabel('y-ycheck')
ax2.set_xlabel('t')
ax2.legend(loc='best')

fig.show()



#%% Resolution numerique d'un systeme d'EDOs simple
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs(u,t,a,b):
    x,y = u
    return -a*y, -b*x

t_arr=np.linspace(0,10.,101)
u_init=[1,0]
a, b = 1., -1.
u_arr=odeint(rhs, u_init, t_arr, args=(a,b,))
x, y = u_arr[:,0], u_arr[:,1]

fig=plt.figure()
ax1=fig.add_subplot(121)
ax1.plot(t_arr, x, t_arr, y)
ax1.set_xlabel('t')
ax1.set_ylabel('x et y')
ax2=fig.add_subplot(122)
ax2.plot(x,y)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.tight_layout()
#fig.subplots_adjust(top=0.90)
plt.savefig("system_EDO.pdf")


#%% EDO de deuxi`eme ordre : Repondre a l'exercice 4.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs(u,x):
    y,z=u
    return  z , 3*z-2*y+2*np.cos(x)

t_arr=np.linspace(0, 2,201)
u_init=[1,0]

u_arr=odeint(rhs,u_init,t_arr)
y,z = u_arr[:,0],u_arr[:,1]

fig=plt.figure()
ax1=fig.add_subplot(121)
ax1.plot(t_arr, z, t_arr, y)
ax1.set_xlabel('t')
ax1.set_ylabel('z et y')


x =np.linspace(0,2,201)

x1=(-1/5)*(np.exp(2*x))+np.exp(x)-(3/5)*np.sin(x)+(1/5)*np.cos(x)

ax2=fig.add_subplot(122)
ax2.plot(x,y-x1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.tight_layout()
plt.show()







#%% Oscillateur harmonique : Repondre a l'exercice 5.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs_harmonic(Y, t, omega):
    y, ydot = Y
    return ydot, -omega**2 * y

def rhs_pendulum(Y, t, omega):
    theta, thetadot = Y
    return thetadot, -omega**2 * np.sin(theta)

t_arr = np.linspace(0, 2 * np.pi, 101)
y_init = [1, 0]
omega = 2.0

# Résolution de l'oscillateur harmonique (équation 7)
y_harmonic = odeint(rhs_harmonic, y_init, t_arr, args=(omega,))

# Résolution du pendule simple (équation 8)
y_pendulum = odeint(rhs_pendulum, y_init, t_arr, args=(omega,))

# Comparaison des solutions
plt.plot(t_arr, y_harmonic[:, 0], label="Oscillateur harmonique")
plt.plot(t_arr, y_pendulum[:, 0], label="Pendule simple")
plt.xlabel('Temps (t)')
plt.ylabel('y(t) / θ(t)')
plt.title("Comparaison des solutions pour l'oscillateur harmonique et le pendule simple")
plt.legend()
plt.show()

plt.figure()
plt.plot(y_harmonic[:, 0], y_harmonic[:, 1], label="Oscillateur harmonique")
plt.xlabel('y')
plt.ylabel('ẏ')
plt.title("Diagramme de phase de l'oscillateur harmonique")
plt.legend()
plt.show()

plt.figure()
plt.plot(y_pendulum[:, 0], y_pendulum[:, 1], label="Pendule simple")
plt.xlabel('θ')
plt.ylabel('θ̇')
plt.title("Diagramme de phase du pendule simple")
plt.legend()
plt.show()

#%% Implementer dans un fichier different ”trajectoires.py” le
# code dans la section 3.1.1

#%% Resolution of harmonic oscillator 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs(Y,t, omega):
    y,ydot =Y
    return ydot, -omega**2*y
    
t_arr=np.linspace(0,2*np.pi,101)
y_init=[1,0]
omega=2.0

fig=plt.figure()
y,ydot=np.mgrid[-3:3:21j, -6:6:21j]
u,v=rhs(np.array([y,ydot]),0.0,omega)
mag=np.hypot(u,v)
mag[mag==0]=1.0
plt.quiver(y,ydot,u/mag,v/mag,color='red')

#Permet dessiner un nombre arbitraire de trajectoires
print('\n\n\nUtilise la souris pour choisir le point de depart')
print('Timout, apres 30 seconds')
print ('\n\n')
choice=[(0,0)]
while len(choice)>0:
    y01=np.array([choice[0][0],choice[0][1]])
    y=odeint(rhs,y01,t_arr,args=(omega,))
    plt.plot(y[:,0],y[:,1],lw=2)
    choice=plt.ginput()
print('Le temps est passe')

#%% Oscillateur de Van dr Pol : Repondre a l'exercice 6.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs(y, t, mu):
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

def jac(y, t, mu):
    return [[0, 1], [-2 * mu * y[0] * y[1] - 1, mu * (1 - y[0]**2)]]

mu = 1.0
t_final = 15.0 if mu < 10 else 4.0 * mu
n_points = 1001 if mu < 10 else 1001 * mu
t = np.linspace(0, t_final, n_points)
y0 = np.array([2.0, 0.0])

y, info = odeint(rhs, y0, t, args=(mu,), Dfun=jac, full_output=True)

print("mu = %g, le nombre d'appels au Jacobien est %d" % (mu, info['nje'][-1]))

plt.plot(y[:, 0], y[:, 1])
plt.xlabel('y')
plt.ylabel('ẏ')
plt.title("Oscillateur de Vander Pol")
plt.savefig("Van_der_Pol.pdf")
plt.show()

    


#%% Van der Pol
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def rhs(y,t,mu):
    return [y[1], mu*(1-y[0]**2)*y[1] - y[0]]

def jac(y,t,mu): # remplir ici avec la fonction construite dans l’exercice 6

 mu=1.0
 t_final = 15.0 if mu<10 else 4.0*mu
 n_points = 1001 if mu<10 else 1001*mu
 t=np.linspace(0, t_final,n_points)
 y0=np.array([2.0,0.0])
 y,info=odeint(rhs,y0,t,args=(mu,),Dfun=jac,full_output=True)

 print (" mu = %g, le nombre d’appels au Jacobien est %d"

plt.plot(y[:,0],y[:,1])
plt.savefig("Van_der_Pol.pdf")

#%% Equations de Kepler (Cartesiennes) : Repondre a l'exercice 7.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres de la simulation
N = 100  # Nombre de points à générer
t = np.linspace(0, 10, N)  # Intervalle de temps
m1 = 1  # Masse de la première particule
m2 = 2  # Masse de la seconde particule

# Fonction pour calculer les vecteurs position et vitesse
def kepler_equations(t, m1, m2):
    x = np.cos(t)
    y = np.sin(t)
    z = t / 10
    x_dot = -np.sin(t)
    y_dot = np.cos(t)
    z_dot = 1 / 10
    r = np.array([x, y, z]).T
    r_dot = np.array([x_dot, y_dot, z_dot]).T
    r_norm = np.linalg.norm(r, axis=1)
    r_norm = r_norm[:, np.newaxis]
    acceleration = -m2 * r / r_norm**3
    return r, r_dot, acceleration

# Calcul des vecteurs position et vitesse
r, r_dot, acceleration = kepler_equations(t, m1, m2)

# Tracé des trajectoires
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title("Trajectoires Kepleriennes")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.plot(r[:, 0], r[:, 1], r[:, 2])
plt.show()


#%% Equations de Lorentz : Repondre a l'exercice 8.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres de la simulation
t = np.linspace(0, 50, 5000)  # Intervalle de temps
σ = 10
β = 8/3
ρ = 28

# Fonction pour décrire les équations de Lorentz
def lorentz_equations(t, y, σ, ρ, β):
    x, y, z = y
    x_dot = σ * (y - x)
    y_dot = ρ * x - y - x * z
    z_dot = x * y - β * z
    return [x_dot, y_dot, z_dot]

# Résolution numérique des équations de Lorentz
sol = solve_ivp(lambda t, y: lorentz_equations(t, y, σ, ρ, β), [t[0], t[-1]], [1, 1, 1])

# Tracé des trajectoires
plt.figure()
plt.title("Trajectoires dans l'attracteur de Lorentz")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(sol.y[0], sol.y[1])
plt.show()
