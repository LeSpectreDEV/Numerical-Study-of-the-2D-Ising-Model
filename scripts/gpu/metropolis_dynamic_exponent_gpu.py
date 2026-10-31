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


#
# Ajout pour les incertitudes
#

def generate_block_bootstrap_sample(data, block_size):
    """
    Génère un nouvel échantillon de même taille via Moving Block Bootstrap.
    """
    N = len(data)
    n_blocks = N // block_size
    
    # Création du tableau pour le nouvel échantillon
    boot_sample = np.zeros(n_blocks * block_size)
    
    for i in range(n_blocks):
        # Tirage d'un point de départ aléatoire pour le bloc
        start_idx = np.random.randint(0, N - block_size + 1)
        # Copie du bloc dans le nouvel échantillon
        boot_sample[i*block_size : (i+1)*block_size] = data[start_idx : start_idx + block_size]
        
    return boot_sample

def estimate_z_bootstrap_end_to_end(dict_magnetizations, tailles_L, n_boot=200, block_size=500):
    """
    Estime l'exposant z et son incertitude par bootstrapping global,
    et affiche le nuage de points de tous les tirages.
    """
    z_bootstrap_values = np.zeros(n_boot)
    intercept_bootstrap_values = np.zeros(n_boot) # Pour tracer la droite moyenne
    
    # Dictionnaire pour stocker TOUS les tau de chaque itération
    all_tau_boot = {L: [] for L in tailles_L}
    
    log_L = np.log(tailles_L)
    
    print(f"Lancement du Bootstrap ({n_boot} itérations)...")
    
    for b in range(n_boot):
        tau_int_boot = []
        
        # 1. Pour chaque taille L, on rééchantillonne et on calcule tau
        for L in tailles_L:
            data = dict_magnetizations[L]
            boot_data = generate_block_bootstrap_sample(data, block_size)
            
            rho = compute_autocorr_fft(boot_data)
            tau = compute_tau_int(rho, c=5.0)
            
            tau_int_boot.append(tau)
            all_tau_boot[L].append(tau) # Sauvegarde du point pour le graphique
            
        # 2. Régression linéaire sur cet univers virtuel
        log_tau = np.log(tau_int_boot)
        slope, intercept, _, _, _ = linregress(log_L, log_tau)
        
        # 3. Stockage de la pente et de l'ordonnée à l'origine
        z_bootstrap_values[b] = slope
        intercept_bootstrap_values[b] = intercept
        
        if (b + 1) % 50 == 0:
            print(f"Progression : {b + 1} / {n_boot}")
            
    # Calcul des statistiques finales
    z_mean = np.mean(z_bootstrap_values)
    z_std = np.std(z_bootstrap_values, ddof=1) # écart-type corrigé de Bessel
    intercept_mean = np.mean(intercept_bootstrap_values)
    
    # =================================================================
    # GRAPHIQUE 1 : Le nuage de points Bootstrap et la régression
    # =================================================================
    plt.figure(figsize=(9, 6))
    
    # Tracer le "nuage" de points avec une forte transparence (alpha)
    for L in tailles_L:
        x_vals = np.full(n_boot, np.log(L))
        y_vals = np.log(all_tau_boot[L])
        # Étiquette unique pour la légende
        label = "Bootstrap samples" if L == tailles_L[0] else ""
        plt.scatter(x_vals, y_vals, color='gray', alpha=0.1, s=20, label=label)
        
    # Tracer les points moyens
    mean_log_tau_per_L = [np.mean(np.log(all_tau_boot[L])) for L in tailles_L]
    plt.plot(log_L, mean_log_tau_per_L, 'o', color='royalblue', markersize=8, label="Average $\\ln(\\tau)$")
    
    # Tracer la droite de régression moyenne
    fit_line = z_mean * log_L + intercept_mean
    plt.plot(log_L, fit_line, 'r--', linewidth=2, label=f"Fit: $z = {z_mean:.3f} \pm {z_std:.3f}$")
    
    plt.xlabel("$\ln(L)$", fontsize=12)
    plt.ylabel("$\ln(\\tau)$", fontsize=12)
    #plt.title("Nuage de dispersion Bootstrap des temps de corrélation", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    # =================================================================
    # GRAPHIQUE 2 : Histogramme de la distribution de z
    # =================================================================
    plt.figure(figsize=(8, 5))
    plt.hist(z_bootstrap_values, bins=20, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.axvline(z_mean, color='red', linestyle='dashed', linewidth=2, label=f"Moyenne: {z_mean:.3f}")
    plt.axvline(z_mean - z_std, color='black', linestyle='dotted', linewidth=2, label=f"$\pm 1 \sigma$ ({z_std:.3f})")
    plt.axvline(z_mean + z_std, color='black', linestyle='dotted', linewidth=2)
    
    plt.xlabel("Exposant dynamique estimé $z^*$", fontsize=12)
    plt.ylabel(f"Fréquence (sur {n_boot} tirages)", fontsize=12)
    plt.title("Distribution Bootstrap de l'exposant dynamique $z$", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
    return z_mean, z_std




# --- SCRIPT PRINCIPAL ---

def main_asynchrone():
    beta_critique = 1.0 / 2.269185
    tailles_L = [32, 48, 64, 80] 
    B = 2000 # nombre de batchs bootstrap
    l = 40_000 # longueur des blocs bootstrap
    
    # Paramètres globaux (calibrés pour le plus grand réseau)
    nb_mesures = 2_000_000
    thermalisation_steps = 150_000
    
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
    donnees_brutes = {} # <--- 1. CRÉATION DU DICTIONNAIRE
    
    for L, m in zip(tailles_L, modeles):
        # C'est seulement maintenant qu'on bloque le programme pour lire la VRAM
        # Utilisation de la valeur absolue de l'aimantation (fortement recommandé à Tc)
        mags = np.abs(m.magnetizations.to_numpy()) 
        
        # <--- 2. STOCKAGE DES DONNÉES POUR LE BOOTSTRAP
        donnees_brutes[L] = mags 
        
        rho = compute_autocorr_fft(mags)
        tau_int = compute_tau_int(rho, c=5.0)
        
        print(f"-> L={L} | tau_int : {tau_int:.2f} MCS")
        mesures_tau_int.append(tau_int)

    z_final, erreur_z_final = estimate_z_bootstrap_end_to_end(
        dict_magnetizations=donnees_brutes, 
        tailles_L=tailles_L, 
        n_boot=B, 
        block_size=l
    )

    print(f"\n==========================================")
    print(f"Résultat final (Bootstrap) : z = {z_final:.3f} ± {erreur_z_final:.3f}")
    print(f"==========================================")

if __name__ == "__main__":
    main_asynchrone()