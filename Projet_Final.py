# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:23:58 2023

@author: Farehan Yahya
"""
#%%Méthode de Monte Carlo
import math
import random
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt

#Cette méthode consiste à générer des points aléatoires dans un carré inscrit dans un cercle
# et à compter le nombre de points qui se trouvent dans le cercle.
#On peut alors estimer la valeur de π comme étant le rapport entre
# le nombre de points dans le cercle et le nombre total de points.

def approximate_pi(num_samples, num_decimal_places):
    getcontext().prec = num_decimal_places + 1  # Add 1 to get correct number of decimal places
    num_inside = 0
    for i in range(num_samples):
        x = Decimal(random.uniform(0, 1))
        y = Decimal(random.uniform(0, 1))
        if x**2 + y**2 <= 1:
            num_inside += 1
    pi_approx = Decimal(4 * num_inside) / Decimal(num_samples)
    return pi_approx

# Example usage

print(approximate_pi(10**6, 10))  # Approximates pi to 100 decimal places using 1 million samples
#%%
def approximate_pi_monte_carlo(num_iterations):
    num_points_inside_circle = 0
    pi_values = []
    x_inside, y_inside, x_outside, y_outside = [], [], [], []

    for i in range(num_iterations):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if math.sqrt(x**2 + y**2) <= 1:
            num_points_inside_circle += 1
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

        pi_approximation = 4 * num_points_inside_circle / num_iterations
        pi_values.append(pi_approximation)

    return pi_approximation, x_inside, y_inside, x_outside, y_outside

num_iterations = 10000
pi, x_inside, y_inside, x_outside, y_outside = approximate_pi_monte_carlo(num_iterations)

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.scatter(x_inside, y_inside, s=1, color='blue')
ax.scatter(x_outside, y_outside, s=1, color='red')
ax.set_title(f'Approximation de Pi en utilisant la méthode de Monte Carlo: {pi:.5f}')
plt.show()

#%% Erreur en utilisant la méthode de Monte Carlo

def approximate_pi_monte_carlo(num_points):
    num_points_inside_circle = 0
    for i in range(num_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            num_points_inside_circle += 1
    pi_approximation = Decimal(4) * Decimal(num_points_inside_circle) / Decimal(num_points)
    return pi_approximation

# Reference value of pi
pi_reference = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')

# Generate list of number of points to use in Monte Carlo approximation
num_points_list = [10**i for i in range(1, 8)]

# Calculate pi approximations for each number of points and store errors
pi_approximations = []
errors = []
for num_points in num_points_list:
    pi_approximation = approximate_pi_monte_carlo(num_points)
    error = abs(pi_approximation - pi_reference)
    pi_approximations.append(pi_approximation)
    errors.append(error)

# Plotting
plt.loglog(num_points_list, errors, '-o')
plt.xlabel('Nombre de points générés')
plt.ylabel('Erreur')
plt.title('Erreur d\'approximation de Pi avec la méthode de Monte Carlo')
plt.show()


#%%Méthode BBP (Bailey-Browein-Plouffe) 

#Cette méthode utilise une formule de la fonction
#zêta de Riemann similaire à celle de Plouffe,
# mais qui converge plus rapidement.
def approximate_pi(num_decimal_places):
    getcontext().prec = num_decimal_places + 1  # Add 1 to get correct number of decimal places
    pi = Decimal(0)
    pi_values = []
    for k in range(num_decimal_places):
        term = (Decimal(4)/(8*k+1) - Decimal(2)/(8*k+4) - Decimal(1)/(8*k+5) - Decimal(1)/(8*k+6)) * Decimal(16)**(-k)
        pi += term
        pi_values.append(pi)
    return pi_values

# Example usage
num_decimal_places_list = [i for i in range(1, 11)]
pi_approximations = []
for num_decimal_places in num_decimal_places_list:
    pi_approximation = approximate_pi(num_decimal_places)
    pi_approximations.append(pi_approximation)

# Plotting
for i in range(len(pi_approximations)):
    plt.plot(pi_approximations[i], label=f"{num_decimal_places_list[i]} décimales")

plt.axhline(y=3.141592653589793, color='r', linestyle='--', label='Valeur de référence')
plt.xlabel('Iterations')
plt.ylabel('Valeur de Pi')
plt.title('Convergence de l\'approximation de Pi avec la méthode BBP')
plt.legend()
plt.show()

#%% Méthode Gausse Legandre

def approximate_pi(num_decimal_places):
    getcontext().prec = num_decimal_places + 1  # Add 1 to get correct number of decimal places
    a, b, t, p = Decimal(1), Decimal(1) / Decimal(2).sqrt(), Decimal(1) / Decimal(4), Decimal(1)
    for i in range(num_decimal_places // 4):
        a_next = (a + b) / Decimal(2)
        b_next = (a * b).sqrt()
        t_next = t - p * (a - a_next)**Decimal(2)
        p_next = Decimal(2) * p
        a, b, t, p = a_next, b_next, t_next, p_next
    pi = (a + b)**Decimal(2) / (Decimal(4) * t)
    return pi

# Example usage
print(approximate_pi(100))  # Approximates pi to 100 decimal places using the Borwein quartic algorithm

#%% affichage de la convergence (Méthode Gauss Legendre)

def approximate_pi_gauss_legendre(num_iterations):
    getcontext().prec = num_iterations * 2  # Set precision to twice the number of iterations
    a_old = Decimal(1)
    b_old = Decimal(1) / Decimal(Decimal(2).sqrt())
    t_old = Decimal(1) / Decimal(4)
    p_old = Decimal(1)
    for i in range(num_iterations):
        a_new = (a_old + b_old) / Decimal(2)
        b_new = (a_old * b_old).sqrt()
        t_new = t_old - p_old * (a_old - a_new)**2
        p_new = Decimal(2) * p_old
        a_old = a_new
        b_old = b_new
        t_old = t_new
        p_old = p_new
    pi_approximation = (a_new + b_new)**2 / (Decimal(4) * t_new)
    return pi_approximation

# Example usage
num_iterations_list = [i for i in range(1, 11)]
pi_approximations = []
for num_iterations in num_iterations_list:
    pi_approximation = approximate_pi_gauss_legendre(num_iterations)
    pi_approximations.append(pi_approximation)

# Plotting
plt.plot(num_iterations_list, pi_approximations)
plt.axhline(y=3.141592653589793, color='r', linestyle='--', label='Valeur de référence')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Valeur de Pi')
plt.title('Convergence de l\'approximation de Pi avec la méthode Gauss-Legendre')
plt.legend()
plt.show()

#%% Méthode Nilakantha

def approximate_pi(num_terms):
    getcontext().prec = num_terms + 2  # Add 2 to get correct number of decimal places
    pi = Decimal(3)
    sign = 1
    for k in range(2, num_terms * 2 + 2, 2):
        term = Decimal(4) / (k * (k + 1) * (k + 2))
        pi += sign * term
        sign *= -1
    return pi

# Example usage
print(approximate_pi(100))  # Approximates pi to 100 decimal places using the Nilakantha series

#%% convergenceen utilisant la méthode Nilakantha

def approximate_pi_nilakantha(num_iterations):
    getcontext().prec = num_iterations * 2 + 1  # Set precision to twice the number of iterations plus 1
    pi = Decimal(3)
    sign = Decimal(1)
    for i in range(2, num_iterations * 2 + 1, 2):
        term = Decimal(4) / (i * (i + 1) * (i + 2)) * sign
        pi += term
        sign *= Decimal(-1)
    return pi

# Example usage
num_iterations_list = [i for i in range(1, 11)]
pi_approximations = []
for num_iterations in num_iterations_list:
    pi_approximation = approximate_pi_nilakantha(num_iterations)
    pi_approximations.append(pi_approximation)

# Plotting
plt.plot(num_iterations_list, pi_approximations)
plt.axhline(y=3.141592653589793, color='r', linestyle='--', label='Valeur de référence')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Valeur de Pi')
plt.title('Convergence de l\'approximation de Pi avec la méthode de Nilakantha')
plt.legend()
plt.show()

#%% The Borwein Quartic 

def approximate_pi(num_decimal_places):
    getcontext().prec = num_decimal_places + 1  # Add 1 to get correct number of decimal places
    a, b, t, p = 1, Decimal(1) / Decimal(2).sqrt(), Decimal(1) / 4, 1
    for i in range(num_decimal_places // 4):
        a_next = (a + b) / 2
        b_next = (a * b).sqrt()
        t_next = t - p * (a - a_next)**2
        p_next = 2 * p
        a, b, t, p = a_next, b_next, t_next, p_next
    pi = (a + b)**2 / (4 * t)
    return pi

# Example usage
print(approximate_pi(100))  # Approximates pi to 100 decimal places using the Borwein quartic algorithm

#%% Convergence en utilisant la méthode Borwein_quatic

def approximate_pi_borwein_quartic(num_iterations):
    getcontext().prec = num_iterations * 14 + 1  # Set precision to 14 times the number of iterations plus 1
    a = Decimal(1)
    b = Decimal(1) / Decimal(2).sqrt()
    t = Decimal(1) / Decimal(4)
    p = Decimal(1)
    for i in range(num_iterations):
        a_new = (a + b) / Decimal(2)
        b = (a * b).sqrt()
        t_new = t - p * (a - a_new)**2
        p = Decimal(2) * p
        a = a_new
        t = t_new
    pi_approximation = (a + b)**2 / (Decimal(4) * t)
    return pi_approximation

# Example usage
num_iterations_list = [i for i in range(1, 11)]
pi_approximations = []
for num_iterations in num_iterations_list:
    pi_approximation = approximate_pi_borwein_quartic(num_iterations)
    pi_approximations.append(pi_approximation)

# Plotting
plt.plot(num_iterations_list, pi_approximations)
plt.axhline(y=3.141592653589793, color='r', linestyle='--', label='Valeur de référence')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Valeur de Pi')
plt.title('Convergence de l\'approximation de Pi avec la méthode de Borwein Quartic')
plt.legend()
plt.show()
#%% Méthode de Plouffe

#Cette méthode utilise une formule de la fonction zêta de Riemann
# pour calculer les décimales de π.
# Cette formule permet de calculer la n-ième décimale de π
# sans avoir besoin de calculer les n-1 premières décimales.


def plouffe_pi(n):
    pi = 0
    for k in range(n):
        pi += 1 / 16**k * (4 / (8*k+1) -2 / (8*k+4) -1 / (8*k+5) -1 / (8*k+6))
    return pi

print(plouffe_pi(1232))

#%% Méthode d'Archimède 
from math import sin , tan , radians
def archimede(p):
    a, b = 0, 1 # valeurs arbitraires
    n = 6
    while (b-a) > 10**(-p):
        a = n * sin( radians(180/n) )
        b = n * tan( radians(180/n) )
        n = n + 1
    return a,b
print (archimede(10))

#La fonction archimede admet un entier pour argument: c’est la précision que l’on souhaite. Dans notre exemple, on veut un encadrement d’amplitude maximale 10**(−10)
#donc on appelle archimede(10). Le retour est le suivant:
    
#%% Archimède géométriquement
from math import pi, cos, sqrt,sin
import matplotlib.pyplot as plt


def polygone_regulier(r, n):
    """Retourne les coordonnées des points d'un polygone régulier inscrit dans un cercle de rayon r et de n côtés."""
    angle = 2 * pi / n
    x = [r * cos(angle * i) for i in range(n)]
    y = [r * sin(angle * i) for i in range(n)]
    return x, y


def polygone_circonscrit(r, n):
    """Retourne les coordonnées des points d'un polygone régulier circonscrit autour d'un cercle de rayon r et de n côtés."""
    angle = 2 * pi / n
    x = [r / cos(angle / 2) * cos(angle * i + angle / 2) for i in range(n)]
    y = [r / cos(angle / 2) * sin(angle * i + angle / 2) for i in range(n)]
    return x, y


def afficher_polygone_regulier(r, n):
    """Affiche le polygone régulier inscrit dans le cercle de rayon r ainsi que le cercle lui-même."""
    fig, ax = plt.subplots()
    x, y = polygone_regulier(r, n)
    ax.plot(x, y, '-o')
    cercle = plt.Circle((0, 0), r, color='r', fill=False)
    ax.add_artist(cercle)
    plt.axis('equal')
    plt.show()


def archimede(n, r=1):
    """Calcule une approximation de pi en utilisant la méthode d'Archimède pour n itérations."""
    pi_approx = 0
    for i in range(n):
        x, y = polygone_regulier(r, 2**i)
        x_c, y_c = polygone_circonscrit(r, 2**i)
        perimetre_polygone_regulier = sum([sqrt((x[j]-x[(j+1)%len(x)])**2 + (y[j]-y[(j+1)%len(y)])**2) for j in range(len(x))])
        perimetre_polygone_circonscrit = sum([sqrt((x_c[j]-x_c[(j+1)%len(x_c)])**2 + (y_c[j]-y_c[(j+1)%len(y_c)])**2) for j in range(len(x_c))])
        pi_approx = 0.5 * (perimetre_polygone_regulier + perimetre_polygone_circonscrit) / r
        afficher_polygone_regulier(r, 2**i)
        print(f"Itération {i+1}: pi ≈ {(pi_approx)/2:.10f}")
        
#Enfin, pour tester la fonction archimede(), on peut l'appeler avec un nombre d'itérations et un rayon initial au choix.
#Par exemple, pour obtenir une approximation de pi avec 6 itérations et un rayon initial de 1,
#on peut appeler la fonction de la manière suivante :
archimede(10, 1)

#%% Méthode Cues
def cotes_method(n):
    pi_approx = 0
    for i in range(n):
        pi_approx += (1 / (2**i)) * math.sqrt(1 - (1 / (2**(2*i+1)))) * 2
    return pi_approx

# Exemple d'utilisation
num_iterations_list = [i for i in range(1, 11)]
pi_approximations = []
for num_iterations in num_iterations_list:
    pi_approximation = cotes_method(num_iterations)
    pi_approximations.append(pi_approximation)

# Plotting
plt.plot(num_iterations_list, pi_approximations)
plt.axhline(y=math.pi, color='r', linestyle='--', label='Valeur de référence')
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Valeur de Pi')
plt.title('Convergence de l\'approximation de Pi avec la méthode de Cotes')
plt.legend()
plt.show()

#Ce code calcule l'approximation de pi en utilisant la méthode de Cotes jusqu'à n itérations
#et affiche ensuite la convergence de l'approximation en fonction du nombre d'itérations. 
#La méthode de Cotes est une méthode d'intégration numérique qui permet d'approximer l'intégrale d'une fonction
# en utilisant des formules de quadrature. Dans le cas de la méthode de Cotes pour approximer pi
#on approxime l'intégrale de la fonction sqrt(1-x^2) sur l'intervalle [0,1] en utilisant des formules 
#de quadrature de plus en plus précises à chaque itération.

#%% visualisation 

def polygone_regulier(r, n):
    angle = 2*math.pi/n
    x = [r*math.cos(k*angle) for k in range(n)]
    y = [r*math.sin(k*angle) for k in range(n)]
    return x, y

def afficher_polygone_regulier(i):
    r = 1/i
    x, y = polygone_regulier(r, i)
    plt.plot(x, y)
    plt.axis('equal')

def approx_pi_cesaro(n):
    getcontext().prec = 30
    pi_approx = Decimal('3')
    for i in range(2, n+1):
        pi_approx = (pi_approx*(i) + 2*Decimal(math.sin(math.pi/i)))/Decimal(i)
        afficher_polygone_regulier(i)
        print(f"Itération {i}: pi ≈ {(pi_approx)/2:.10f}")
    return pi_approx

approx_pi_cesaro(200)
plt.show()
