import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from collections import deque
import time


#
# Test simu avec Metropolis (repris du notebook)
#

class Modele_Ising:
    def __init__(self, taille, energie_interaction, champ):
        self.N = taille
        self.model = np.random.choice([-1, 1], size=(self.N, self.N))
        self.J = energie_interaction
        self.h = champ
        self.m = np.sum(self.model)
        self.E = self.Energie()

    def Energie(self):

        s = self.model

        voisins = (
            np.roll(s,1,axis=0) +
            np.roll(s,-1,axis=0) +
            np.roll(s,1,axis=1) +
            np.roll(s,-1,axis=1)
        )

        E_interaction = -self.J * np.sum(s * voisins) / 2
        E_champ = -self.h * np.sum(s)

        return E_interaction + E_champ

    def plot(self):
        plt.imshow(self.model, cmap='coolwarm')
        plt.colorbar()
        plt.show()

    def magnetisation(self):
        return self.m/(self.N**2)

    def Metropolis(self, beta):
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)

        s = self.model[i][j]
        voisins = (
            self.model[(i+1)%self.N, j] +
            self.model[(i-1)%self.N, j] +
            self.model[i, (j+1)%self.N] +
            self.model[i, (j-1)%self.N]
        )

        diffE = 2 * s * (self.J * voisins + self.h)

        if diffE < 0:
            self.m -= 2*self.model[i, j]
            self.E += diffE
            self.model[i, j] *= -1
        else:
            R = np.exp(-beta * diffE)
            p = np.random.uniform(0.0, 1.0)

            if p <= R:
                self.m -= 2*self.model[i, j]
                self.E += diffE
                self.model[i, j] *= -1

    def Wolff_step(self, beta):
        p_add = 1.0 - np.exp(-2.0 * beta * self.J)
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        seed_spin = self.model[i, j]
        
        stack = [(i, j)]
        self.model[i, j] *= -1 
        self.m += 2 * self.model[i, j]
        
        cluster_size = 1 # <--- NOUVEAU COMPTEUR
        
        while stack:
            cy, cx = stack.pop()
            
            neighbors = [
                ((cy - 1) % self.N, cx),
                ((cy + 1) % self.N, cx),
                (cy, (cx - 1) % self.N),
                (cy, (cx + 1) % self.N)
            ]
            
            for ny, nx in neighbors:
                if self.model[ny, nx] == seed_spin:
                    if np.random.rand() < p_add:
                        stack.append((ny, nx))
                        self.model[ny, nx] *= -1
                        self.m += 2 * self.model[ny, nx]
                        cluster_size += 1 # <--- INCREMENTATION
                        
        return cluster_size # <--- RENVOIE LA TAILLE

def thermalisation (modele,beta):
    for i in range (10000*(modele.N**2)):
        if i % (1000*(modele.N**2)) == 0:
            print(f"Thermalisation : {10*i//(1000*(modele.N**2))}%")
        modele.Metropolis(beta)

def mesures_magnetization_MC(modele,beta,pas, nb_mesures): # Mesures de grandeurs par la méthodes de Monte-Carlo
    M = np.zeros(nb_mesures) # mesures d'aimantation
    print("Thermalisation en cours...")
    thermalisation (modele, beta)
    print("Thermalisation terminée. Début des mesures...")

    for i in range(nb_mesures):
        for _ in range(pas):
            modele.Metropolis(beta)
        if (i+1) % (nb_mesures//100) == 0:
            print(f"Mesure {(i+1)//(nb_mesures//100)}%")
        M[i]=modele.m

    #M_mesure = np.mean (M/modele.N**2)
    #Msquare  = np.mean ((M/modele.N**2)**2)
    #Mabs     = np.mean (np.abs(M)/modele.N**2)

    return M




#
# Houdayer version
#

def houdayer_move_2d_pbc(grid1, grid2):
    """
    Effectue un mouvement de Houdayer sur deux configurations d'Ising 2D
    en respectant les conditions aux limites périodiques (topologie torique).
    """
    H, W = grid1.shape
    
    # 1. Calcul du chevauchement local
    diff_mask = (grid1 != grid2)
    
    if not diff_mask.any():
        return grid1, grid2

    # 2. Choisir un site de départ au hasard parmi les différences
    diff_indices = np.argwhere(diff_mask)
    start_y, start_x = diff_indices[np.random.randint(len(diff_indices))]
    
    # 3. Construire le cluster via un parcours en largeur (BFS)
    cluster = []
    queue = [(start_y, start_x)]
    
    # Un set pour un accès en O(1) lors de la vérification des sites déjà visités
    visited = set([(start_y, start_x)])
    
    while queue:
        y, x = queue.pop(0)
        cluster.append((y, x))
        
        # Définition des 4 voisins avec conditions aux limites périodiques (modulo)
        neighbors = [
            ((y - 1) % H, x),  # Haut (revient en bas si y=0)
            ((y + 1) % H, x),  # Bas (revient en haut si y=H-1)
            (y, (x - 1) % W),  # Gauche (revient à droite si x=0)
            (y, (x + 1) % W)   # Droite (revient à gauche si x=W-1)
        ]
        
        for ny, nx in neighbors:
            # Si le voisin est aussi en désaccord et n'a pas encore été visité
            if diff_mask[ny, nx] and (ny, nx) not in visited:
                visited.add((ny, nx))
                queue.append((ny, nx))
                
    # 4. Basculement (Flip) des spins du cluster identifié
    # On convertit le cluster en tuples d'indices pour indexer facilement numpy
    cluster_y = [pos[0] for pos in cluster]
    cluster_x = [pos[1] for pos in cluster]
    
    grid1[cluster_y, cluster_x] *= -1
    grid2[cluster_y, cluster_x] *= -1

    return grid1, grid2

def houdayer_thermalisation(modele1, modele2, beta, steps):
    """Thermalisation d'une paire de modèles avec Houdayer steps."""
    for i in range (steps):
        """if i % (steps//10) == 0:
            print(f"Thermalisation : {(i)//(steps//100)}%")"""
        if i % (modele1.N**2) == 0:
            # Houdayer tous les N^2 étapes
            grid1, grid2 = houdayer_move_2d_pbc(modele1.model, modele2.model)
            modele1.model, modele2.model = grid1, grid2
        modele1.Metropolis(beta)

def houdayer_mesures_magnetization_MC(modele1,modele2,beta,pas, nb_mesures): # Mesures de grandeurs par la méthodes de Monte-Carlo
    # deux modèles en parallèle pour utiliser Houdayer

    M = np.zeros(nb_mesures) # mesures d'aimantation
    print("Thermalisation en cours...")
    houdayer_thermalisation(modele1, modele2, beta, steps=1000)
    print("Thermalisation terminée. Début des mesures...")

    for i in range(nb_mesures):
        for _ in range(pas):
            modele1.Metropolis(beta)
            modele2.Metropolis(beta)
        # Houdayer step
        grid1, grid2 = houdayer_move_2d_pbc(modele1.model, modele2.model)
        modele1.model, modele2.model = grid1, grid2        
        """if (i+1) % (nb_mesures//100) == 0:
            print(f"Mesure {(i+1)//(nb_mesures//100)}%")"""
        
        M[i]=modele1.m

    #M_mesure = np.mean (M/modele.N**2)
    #Msquare  = np.mean ((M/modele.N**2)**2)
    #Mabs     = np.mean (np.abs(M)/modele.N**2)

    return M



#
# Mesures de correlation time
#

def compute_autocorr_fft(observable_data):
    """
    Calcule la fonction d'autocorrélation normalisée rho(t) via FFT.
    """
    N = len(observable_data)
    
    # 1. Centrer la série temporelle (O_i - <O>)
    data_centered = observable_data - np.mean(observable_data)
    
    # 2. FFT avec zero-padding (taille 2N pour éviter l'aliasing circulaire)
    # On utilise rfft car nos données sont réelles (plus rapide)
    fft_result = np.fft.rfft(data_centered, n=2*N)
    
    # 3. Densité spectrale de puissance (|FFT|^2)
    power_spectrum = np.abs(fft_result)**2
    
    # 4. Transformée de Fourier inverse
    autocorr = np.fft.irfft(power_spectrum, n=2*N)[:N]
    
    # 5. Normalisation pour obtenir un estimateur non biaisé
    # On divise par (N - t) car on a de moins en moins de points pour les grands t
    autocorr /= (N - np.arange(N))
    
    # 6. Normalisation finale pour que rho(0) = 1
    rho = autocorr / autocorr[0]
    
    return rho

def compute_tau_int(rho, c=5.0):
    """
    Calcule le temps de corrélation intégré avec la méthode de Madras-Sokal.
    """
    tau_int = 0.5
    for t in range(1, len(rho)):
        tau_int += rho[t]
        
        # Critère d'arrêt : on coupe quand t >= c * tau_int
        # ou quand rho(t) devient négatif (bruit pur)
        if t >= c * tau_int or rho[t] <= 0:
            return tau_int
            
    return tau_int # Cas limite si la condition n'est jamais atteinte

def exponential_decay(t, tau):
    return np.exp(-t / tau)

def compute_tau_exp(rho, t_min, t_max):
    """
    Estime tau_exp en ajustant une exponentielle sur rho(t) entre t_min et t_max.
    """
    # Sélection de la fenêtre temporelle
    t_data = np.arange(t_min, t_max)
    rho_data = rho[t_min:t_max]
    
    # Ajustement par moindres carrés
    # p0 est la valeur de départ estimée pour tau
    popt, pcov = curve_fit(exponential_decay, t_data, rho_data, p0=[10.0])
    
    tau_exp = popt[0]
    error = np.sqrt(pcov[0][0]) # Erreur sur l'ajustement
    
    return tau_exp, error


#
# Estimer z
#

def estimate_dynamic_exponent_z(L_values, tau_values):
    """
    Estime l'exposant dynamique z par régression linéaire sur un graphe log-log,
    et affiche le tracé pour vérification visuelle.
    
    L_values: liste ou array des tailles de grille (ex: [16, 32, 64, 128])
    tau_values: liste ou array des temps de corrélation correspondants
    """
    # 1. Transformation logarithmique
    log_L = np.log(L_values)
    log_tau = np.log(tau_values)
    
    # 2. Régression linéaire : log(tau) = z * log(L) + constante
    slope, intercept, r_value, p_value, std_err = linregress(log_L, log_tau)
    
    z = slope
    z_err = std_err
    
    # 3. Tracé du graphique de vérification
    plt.figure(figsize=(8, 5))
    plt.plot(log_L, log_tau, 'o', color='royalblue', label="Mesures $\\tau_{int}$", markersize=8)
    
    # Droite d'ajustement
    fit_line = z * log_L + intercept
    plt.plot(log_L, fit_line, 'r--', label=f"Ajustement : $z = {z:.3f} \pm {z_err:.3f}$")
    
    plt.xlabel("$\ln(L)$", fontsize=12)
    plt.ylabel("$\ln(\\tau)$", fontsize=12)
    plt.title("Estimation de l'exposant dynamique $z$", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
    return z, z_err

#
# Test complet
#
"""
# Mesures de l'aimantation à partir d'une simulation Ising
beta_critique = 1.0 / 2.269185

N=64
modele1 = Modele_Ising(taille=N, energie_interaction=1.0, champ=0.0)
modele2 = Modele_Ising(taille=N, energie_interaction=1.0, champ=0.0)
magnetizations = houdayer_mesures_magnetization_MC(modele1, modele2, beta=beta_critique, pas=N**2, nb_mesures=1_000)

# 1. Calcul ultra-rapide de rho(t)
rho = compute_autocorr_fft(magnetizations)

# 2. Calcul du temps intégré
tau_int = compute_tau_int(rho, c=5.0)
print(f"Temps de corrélation intégré : {tau_int:.2f} MCS")

# Calcul de l'erreur statistique réelle sur l'aimantation
print(f"Aimantation moyenne : {np.mean(magnetizations):.5f}")
N = len(magnetizations)
variance = np.var(magnetizations)
true_error = np.sqrt((variance / N) * 2 * tau_int)
print(f"Erreur corrigée : {true_error:.5f}")

# 3. Calcul du temps exponentiel
# Il faut choisir la fenêtre visuellement en traçant log(rho), 
# ou empiriquement (par exemple entre 1*tau_int et 3*tau_int)
t_min = int(1.5 * tau_int)
t_max = int(4.0 * tau_int)
tau_exp, tau_exp_err = compute_tau_exp(rho, t_min, t_max)
print(f"Temps de corrélation exponentiel : {tau_exp:.2f} +/- {tau_exp_err:.2f} MCS")
"""
"""
#
# (Houdayer) Test pour différentes tailles de grille pour estimer z
# 

# Tailles de réseau à tester
beta_critique = 1.0 / 2.269185
tailles_L = [50, 100, 150, 200]
mesures_tau_int = []

for L in tailles_L:
    print(f"\nSimulation pour L = {L}...")
    modele1 = Modele_Ising(taille=L, energie_interaction=1.0, champ=0.0)
    modele2 = Modele_Ising(taille=L, energie_interaction=1.0, champ=0.0)
    magnetizations = houdayer_mesures_magnetization_MC(modele1, modele2, beta=beta_critique, pas=L**2, nb_mesures=10_000)
    
    # Calculer l'autocorrélation et le tau_int
    rho = compute_autocorr_fft(magnetizations)
    tau_int = compute_tau_int(rho, c=5.0)
    
    print(f"-> tau_int obtenu : {tau_int:.2f} MCS")
    mesures_tau_int.append(tau_int)


# Calcul de z et de son erreur à partir des mesures de tau_int pour différentes tailles L
z_estime, erreur_z = estimate_dynamic_exponent_z(tailles_L, mesures_tau_int)

print(f"\nRésultat final : z = {z_estime:.3f} +/- {erreur_z:.3f}")"""


#
# Test complet avec Wolff pour différentes tailles de grille pour estimer z
#

tailles_L = [50, 100, 150, 200]
mesures_tau_int = []
beta_critique = 1.0 / 2.269185

for L in tailles_L:
    print(f"\nSimulation Wolff pour L = {L}...")
    modele = Modele_Ising(taille=L, energie_interaction=1.0, champ=0.0)
    
    # 1. Thermalisation équivalente à 100 MCS (bien suffisant avec Wolff)
    for _ in range(100):
        spins_flipped = 0
        while spins_flipped < L**2:
            spins_flipped += modele.Wolff_step(beta_critique)
            
    # 2. Mesures
    nb_mesures = 10000
    magnetizations = np.zeros(nb_mesures)
    
    for i in range(nb_mesures):
        spins_flipped = 0
        # On définit 1 "pas" d'horloge = avoir retourné au moins L^2 spins
        while spins_flipped < L**2:
            spins_flipped += modele.Wolff_step(beta_critique)
            
        magnetizations[i] = abs(modele.m) 
        
    # 3. Calcul de tau_int
    rho = compute_autocorr_fft(magnetizations)
    tau_int = compute_tau_int(rho, c=5.0)
    
    print(f"-> tau_int obtenu : {tau_int:.2f} MCS")
    mesures_tau_int.append(tau_int)

# Calcul final
z_estime, erreur_z = estimate_dynamic_exponent_z(tailles_L, mesures_tau_int)
print(f"\nRésultat final : z = {z_estime:.3f} +/- {erreur_z:.3f}")