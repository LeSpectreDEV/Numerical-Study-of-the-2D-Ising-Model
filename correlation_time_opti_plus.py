import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import time

def format_execution_time(seconds: float) -> str:
    """
    Convertit un temps d'exécution en secondes en une chaîne de caractères lisible.
    Gère les millisecondes, secondes, minutes, heures et jours.
    """
    if seconds < 0:
        return "0 ms"
        
    # Pour les temps d'exécution inférieurs à une seconde
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"

    # Calcul des différentes unités avec divmod
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    # Construction de la chaîne de résultat
    parts = []
    if days > 0:
        parts.append(f"{int(days)} d")
    if hours > 0:
        parts.append(f"{int(hours)} h")
    if minutes > 0:
        parts.append(f"{int(minutes)} m")
    
    # On ajoute toujours les secondes s'il y a un reste, ou si le temps est exactement 0
    if seconds > 0 or not parts:
        parts.append(f"{seconds:.2f} s")

    return " ".join(parts)

# Initialisation de Taichi (Vulkan est idéal pour AMD/Windows, CUDA pour NVIDIA)
ti.init(arch=ti.vulkan)

@ti.data_oriented
class Modele_Ising_GPU:
    def __init__(self, taille, J, h, nb_mesures):
        self.N = taille
        self.J = J
        self.h = h
        self.nb_mesures = nb_mesures
        
        # Champs alloués sur la carte graphique (VRAM)
        self.spins = ti.field(dtype=ti.i8, shape=(self.N, self.N))
        self.magnetizations = ti.field(dtype=ti.f32, shape=self.nb_mesures)

    @ti.kernel
    def init_grid(self):
        """Initialisation aléatoire (T = infini)"""
        for i, j in self.spins:
            if ti.random(ti.f32) < 0.5:
                self.spins[i, j] = ti.cast(1, ti.i8)
            else:
                self.spins[i, j] = ti.cast(-1, ti.i8)

    @ti.kernel
    def metropolis_phase(self, color: ti.i32, beta: ti.f32):
        """Exécute une demi-étape de Metropolis (Cases noires ou rouges)"""
        for i, j in self.spins:
            if (i + j) % 2 == color:
                s = self.spins[i, j]
                
                # Voisins avec conditions aux limites périodiques
                voisins = self.spins[(i + 1) % self.N, j] + \
                          self.spins[(i - 1 + self.N) % self.N, j] + \
                          self.spins[i, (j + 1) % self.N] + \
                          self.spins[i, (j - 1 + self.N) % self.N]
                
                diffE = 2.0 * self.J * float(s) * (float(voisins) + self.h)
                
                if diffE <= 0.0 or ti.random(ti.f32) < ti.exp(-beta * diffE):
                    self.spins[i, j] = ti.cast(-s, ti.i8)

    @ti.kernel
    def record_magnetization(self, step: ti.i32):
        """Calcule et stocke l'aimantation totale sur le GPU"""
        m = 0
        for i, j in self.spins:
            m += self.spins[i, j] # Taichi gère la réduction (somme) automatiquement
        
        self.magnetizations[step] = float(m) / float(self.N * self.N)

    def run_thermalization(self, beta, steps):
        """Effectue 'steps' MCS par paquets pour éviter le Timeout Windows."""
        for i in range(steps):
            self.metropolis_phase(0, beta)
            self.metropolis_phase(1, beta)
            
            # Tous les 10 000 pas, on force la carte graphique à respirer
            if i % 10000 == 0 and i > 0:
                ti.sync()
                pourcentage = (i / self.nb_mesures) * 100
                print(f"    Progression : {pourcentage:.1f}%", end="\r")
        ti.sync() # Synchronisation finale de sécurité

    def run_measurements(self, beta):
        """Effectue les mesures en forçant des synchronisations régulières."""
        for t in range(self.nb_mesures):
            self.metropolis_phase(0, beta)
            self.metropolis_phase(1, beta)
            self.record_magnetization(t)
            
            # On purge la file d'attente Taichi tous les 10 000 pas
            if t % 10000 == 0 and t > 0:
                ti.sync()
                pourcentage = (t / self.nb_mesures) * 100
                print(f"    Progression : {pourcentage:.1f}%", end="\r")
                
        # Rapatrie le tableau 1D de la VRAM vers la RAM (NumPy) à la toute fin
        return self.magnetizations.to_numpy()


# --- VOS FONCTIONS DE CALCUL D'AUTOCORRÉLATION ---

def compute_autocorr_fft(observable_data):
    N = len(observable_data)
    data_centered = observable_data - np.mean(observable_data)
    fft_result = np.fft.rfft(data_centered, n=2*N)
    power_spectrum = np.abs(fft_result)**2
    autocorr = np.fft.irfft(power_spectrum, n=2*N)[:N]
    autocorr /= (N - np.arange(N))
    rho = autocorr / autocorr[0]
    return rho

def compute_tau_int(rho, c=5.0):
    tau_int = 0.5
    for t in range(1, len(rho)):
        tau_int += rho[t]
        if t >= c * tau_int or rho[t] <= 0:
            return tau_int
    return tau_int


# --- SCRIPT PRINCIPAL ---

def main_asynchrone():
    beta_critique = 1.0 / 2.269185
    tailles_L = [16, 32, 48, 64, 80] 
    
    # Paramètres globaux (calibrés pour le plus grand réseau)
    nb_mesures = 4_000_000
    thermalisation_steps = 100_000
    
    print("1. Initialisation des modèles en mémoire (VRAM)...")
    modeles = []
    for L in tailles_L:
        # On crée nos 4 modèles indépendants
        m = Modele_Ising_GPU(taille=L, J=1.0, h=0.0, nb_mesures=nb_mesures)
        m.init_grid()
        modeles.append(m)
        
    print(f"2. Thermalisation simultanée ({thermalisation_steps} MCS)...")
    start_time  = time.time()
    # LE SECRET EST ICI : La boucle temporelle est à l'extérieur !
    for i in range(thermalisation_steps):
        for m in modeles:
            # Taichi empile ces instructions sans bloquer le CPU
            m.metropolis_phase(0, beta_critique)
            m.metropolis_phase(1, beta_critique)
            
        # On purge la file d'attente pour TOUS les modèles en même temps
        if i % 5000 == 0:
            ti.sync()
            if i > 0 and i % 50000 == 0:
                print(f"   Progression : {(i / nb_mesures) * 100:.1f}%", end="\r")
            
    ti.sync() # Fin de la thermalisation
    print(f"Thermalisation terminée en {format_execution_time(time.time()-start_time)}.")
    
    print(f"3. Mesures simultanées ({nb_mesures} MCS)...")
    start_time  = time.time()
    for t in range(nb_mesures):
        for m in modeles:
            m.metropolis_phase(0, beta_critique)
            m.metropolis_phase(1, beta_critique)
            m.record_magnetization(t)
            
        if t % 5000 == 0:
            ti.sync()
            if t > 0 and t % 50000 == 0:
                print(f"   Progression : {(t / nb_mesures) * 100:.1f}% (temps restant estimé : {format_execution_time(((1-(t / nb_mesures))/(t / nb_mesures))*(time.time()-start_time))})", end="\r")
                
    ti.sync()
    print(f"\n   Mesures terminées en {format_execution_time(time.time()-start_time)}.")

    print("4. Rapatriement des données et calcul d'autocorrélation...")
    mesures_tau_int = []
    
    for L, m in zip(tailles_L, modeles):
        # C'est seulement maintenant qu'on bloque le programme pour lire la VRAM
        mags = m.magnetizations.to_numpy() 
        
        rho = compute_autocorr_fft(mags)
        tau_int = compute_tau_int(rho, c=5.0)
        
        print(f"-> L={L} | tau_int : {tau_int:.2f} MCS")
        mesures_tau_int.append(tau_int)

    # --- Calcul de z (identique à avant) ---
    log_L = np.log(tailles_L)
    log_tau = np.log(mesures_tau_int)
    slope, intercept, _, _, std_err = linregress(log_L, log_tau)
    
    print(f"\nRésultat final : exposant dynamique z = {slope:.3f} +/- {std_err:.3f}")
    
    # Tracé
    plt.figure(figsize=(8, 5))
    plt.plot(log_L, log_tau, 'o', color='royalblue', label="Mesures $\\tau_{int}$", markersize=8)
    plt.plot(log_L, slope * log_L + intercept, 'r--', label=f"Ajustement : $z = {slope:.3f} \pm {std_err:.3f}$")
    plt.xlabel("$\ln(L)$", fontsize=12)
    plt.ylabel("$\ln(\\tau)$", fontsize=12)
    plt.title("Estimation de l'exposant dynamique $z$ (Metropolis GPU)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main_asynchrone()