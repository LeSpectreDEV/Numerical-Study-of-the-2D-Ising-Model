import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


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


#
# Dynamic exponent
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
    plt.plot(log_L, log_tau, 'o', color='royalblue', label="Measures $\\tau_{int}$", markersize=8)
    
    # Droite d'ajustement
    fit_line = z * log_L + intercept
    plt.plot(log_L, fit_line, 'r--', label=f"Regression : $z = {z:.3f} \pm {z_err:.3f}$")
    
    plt.xlabel("$\ln(n)$", fontsize=12)
    plt.ylabel("$\ln(\\tau)$", fontsize=12)
    plt.title("Estimation of the dynamic exponent $z$ (Metropolis)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
    return z, z_err


"""
#
# Metropolis dynamic exponent
#
tailles_L = [16, 32, 48, 64]
tau = [291.46, 1248.31, 3766.30, 5876.79]
estimate_dynamic_exponent_z(tailles_L, tau)
"""


#
# Wolff décroissance de 
#

L = 128
beta_critique = 1.0 / 2.269185

print(f"\nSimulation Wolff pour L = {L}...")
modele = Modele_Ising(taille=L, energie_interaction=1.0, champ=0.0)

# 1. Thermalisation équivalente à 100 MCS (bien suffisant avec Wolff)
for _ in range(100):
    spins_flipped = 0
    while spins_flipped < L**2:
        spins_flipped += modele.Wolff_step(beta_critique)
        
# 2. Mesures
nb_mesures =  200#10000
magnetizations = np.zeros(nb_mesures)

for i in range(nb_mesures):
    spins_flipped = 0
    # On définit 1 "pas" d'horloge = avoir retourné au moins L^2 spins
    """while spins_flipped < L**2:
        spins_flipped += modele.Wolff_step(beta_critique)"""
    spins_flipped += modele.Wolff_step(beta_critique)
        
    magnetizations[i] = abs(modele.m) 
    
# 3. Calcul de tau_int
rho = compute_autocorr_fft(magnetizations)
t_values = list(range(0, len(rho)))

plt.plot(t_values, rho, 'r--')
    
plt.xlabel("Algorithm steps (t)", fontsize=12)
plt.ylabel(r"$\rho (t)$", fontsize=12)
plt.title(r"Evolution of the autocorrelation fonction $\rho$", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.7)
plt.show()
