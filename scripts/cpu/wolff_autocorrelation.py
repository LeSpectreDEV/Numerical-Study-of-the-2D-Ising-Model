"""
Autocorrelation function of |m| for the Wolff algorithm (CPU).

Records a time series of |m| at T = Tc for an L x L lattice (one time unit =
one Wolff cluster flip), then plots:
  - the normalised autocorrelation function rho(t)            (report, Figure 5)
  - ln(rho(t)) and a linear fit giving a rough tau_exp        (report, Figure 6)

Report: Section 4.2.1.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


class Modele_Ising:
    def __init__(self, taille, energie_interaction, champ):
        self.N = taille
        self.model = np.random.choice([-1, 1], size=(self.N, self.N))
        self.J = energie_interaction
        self.h = champ
        self.m = np.sum(self.model)

    def Wolff_step(self, beta):
        p_add = 1.0 - np.exp(-2.0 * beta * self.J)
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        seed_spin = self.model[i, j]

        stack = [(i, j)]
        self.model[i, j] *= -1
        self.m += 2 * self.model[i, j]

        cluster_size = 1

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
                        cluster_size += 1

        return cluster_size  # taille de l'amas retourné


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


def main(L=128, nb_mesures=10_000, t_affichage=50, t_fit=30):
    beta_critique = 1.0 / 2.269185

    print(f"Simulation Wolff pour L = {L}...")
    modele = Modele_Ising(taille=L, energie_interaction=1.0, champ=0.0)

    # 1. Thermalisation équivalente à 100 MCS (bien suffisant avec Wolff)
    for _ in range(100):
        spins_flipped = 0
        while spins_flipped < L**2:
            spins_flipped += modele.Wolff_step(beta_critique)

    # 2. Mesures : ici, une unité de temps = un pas de Wolff (un amas retourné)
    magnetizations = np.zeros(nb_mesures)
    for i in range(nb_mesures):
        modele.Wolff_step(beta_critique)
        magnetizations[i] = abs(modele.m)

    # 3. Fonction d'autocorrélation (Figure 5)
    rho = compute_autocorr_fft(magnetizations)
    t_values = np.arange(t_affichage + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(t_values, rho[:t_affichage + 1], 'r.-')
    plt.xlabel("Algorithm steps (t)", fontsize=12)
    plt.ylabel(r"Autocorrelation function $\rho(t)$", fontsize=12)
    plt.grid(True, alpha=0.7)
    plt.show()

    # 4. Estimation grossière de tau_exp : régression de ln(rho) sur t (Figure 6)
    t_fit_values = np.arange(t_fit + 1)
    rho_fit = rho[:t_fit + 1]
    positif = rho_fit > 0  # ln(rho) n'est défini que pour rho > 0
    slope, intercept, _, _, _ = linregress(t_fit_values[positif], np.log(rho_fit[positif]))
    tau_exp = -1.0 / slope
    print(f"tau_exp ≈ {tau_exp:.2f} pas de Wolff")

    plt.figure(figsize=(7, 4))
    plt.plot(t_fit_values[positif], np.log(rho_fit[positif]), 'o', color='royalblue', label=r"Measures $\ln(\rho(t))$")
    plt.plot(t_fit_values, slope * t_fit_values + intercept, 'r--', label=rf"Linear fit : $\tau_{{exp}} \approx {tau_exp:.2f}$")
    plt.xlabel("Algorithm steps (t)", fontsize=12)
    plt.ylabel(r"$\ln(\rho(t))$", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
