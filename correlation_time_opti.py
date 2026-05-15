import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

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
                self.spins[i, j] = 1
            else:
                self.spins[i, j] = -1

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
                    self.spins[i, j] = -s

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

def main():
    beta_critique = 1.0 / 2.269185
    tailles_L = [32, 48, 64]
    mesures_tau_int = []

    for L in tailles_L:
        print(f"\n--- Simulation Metropolis GPU pour L = {L} ---")
        
        """# On ajuste dynamiquement le nombre de mesures pour avoir de bonnes statistiques
        nb_mesures = 5000 * L  
        thermalisation_steps = 200 * L """

        # nombre fixe très important de mesures
        nb_mesures = 2_000_000
        thermalisation_steps = 100_000
        
        modele_gpu = Modele_Ising_GPU(taille=L, J=1.0, h=0.0, nb_mesures=nb_mesures)
        modele_gpu.init_grid()
        
        print(f"Thermalisation ({thermalisation_steps} MCS)...")
        modele_gpu.run_thermalization(beta=beta_critique, steps=thermalisation_steps)
        
        print(f"Mesures ({nb_mesures} MCS)...")
        mags = modele_gpu.run_measurements(beta=beta_critique)
        
        # Calcul de tau_int
        rho = compute_autocorr_fft(mags)
        tau_int = compute_tau_int(rho, c=5.0)

        if L == 64:
            # permet de voir si on a fait assez de mesures
            plt.plot(rho[:20000])
            plt.axhline(0, color='red')
            plt.title("Déclin de rho(t) pour L=64")
            plt.show()
        
        print(f"-> tau_int obtenu : {tau_int:.2f} MCS")
        mesures_tau_int.append(tau_int)

    # 3. Estimation de z
    log_L = np.log(tailles_L)
    log_tau = np.log(mesures_tau_int)
    slope, intercept, r_value, p_value, std_err = linregress(log_L, log_tau)
    
    print(f"\nRésultat final : exposant dynamique z = {slope:.3f} +/- {std_err:.3f} (Théorie Metropolis ≈ 2.16)")

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
    main()