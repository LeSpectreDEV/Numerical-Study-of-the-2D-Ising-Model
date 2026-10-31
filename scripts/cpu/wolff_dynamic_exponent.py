"""
Dynamic exponent z of the Wolff algorithm (CPU).

At T = Tc and for several lattice sizes L, a time series of |m| is recorded
with the Wolff algorithm (one time unit = 1 MCS, i.e. at least L^2 flipped
spins). The integrated autocorrelation time tau_int is computed for each L and
z is obtained from tau ~ L^z, with uncertainties from a block bootstrap.

Report: Section 4.2.2 (Figure 8).

The file also contains an exploratory implementation of the Niedermayer
algorithm (a generalisation of Wolff's algorithm), which is not discussed in
the report.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


#
# Modèle d'Ising (repris du notebook) avec les pas de Metropolis, Wolff et Niedermayer
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

        return cluster_size  # taille de l'amas (pour l'horloge en MCS)

    def Niedermayer_step(self, beta, E0):
        """
        Effectue une étape de l'algorithme de Niedermayer.
        Retourne la taille de l'amas exploré (pour l'horloge MCS).
        """
        # 1. Probabilité d'ajout et facteur d'acceptation
        p_add = 1.0 - np.exp(-beta * E0)
        delta_E_factor = beta * (E0 - 2.0 * self.J)

        # Tirage de la graine
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        seed_spin = self.model[i, j]

        # 2. Initialisation des structures de données
        stack = [(i, j)]
        cluster = [(i, j)]

        # Matrice booléenne pour un accès O(1) ultra-rapide
        in_cluster = np.zeros((self.N, self.N), dtype=bool)
        in_cluster[i, j] = True

        # 3. Construction de l'amas
        while stack:
            cy, cx = stack.pop()

            neighbors = [
                ((cy - 1) % self.N, cx),
                ((cy + 1) % self.N, cx),
                (cy, (cx - 1) % self.N),
                (cy, (cx + 1) % self.N)
            ]

            for ny, nx in neighbors:
                if not in_cluster[ny, nx]:
                    if self.model[ny, nx] == seed_spin:
                        if np.random.rand() < p_add:
                            in_cluster[ny, nx] = True
                            stack.append((ny, nx))
                            cluster.append((ny, nx))

        # 4. Évaluation de la frontière (Boundary)
        Np = 0 # Liens parallèles sur le bord
        Na = 0 # Liens anti-parallèles sur le bord

        # On parcourt uniquement les spins de l'amas pour vérifier leurs voisins
        for cy, cx in cluster:
            neighbors = [
                ((cy - 1) % self.N, cx),
                ((cy + 1) % self.N, cx),
                (cy, (cx - 1) % self.N),
                (cy, (cx + 1) % self.N)
            ]
            for ny, nx in neighbors:
                # Si le voisin n'est pas dans l'amas, c'est un lien de frontière
                if not in_cluster[ny, nx]:
                    if self.model[ny, nx] == seed_spin:
                        Np += 1
                    else:
                        Na += 1

        # 5. Bilan détaillé et basculement
        # Calcul de la probabilité d'acceptation A
        power = delta_E_factor * (Np - Na)

        # Prévention d'overflow si la puissance est positive et très grande
        if power >= 0:
            A = 1.0
        else:
            A = np.exp(power)

        # Si le mouvement est accepté, on bascule tout
        if np.random.rand() < A:
            # Récupération rapide des indices x et y
            ys = [pos[0] for pos in cluster]
            xs = [pos[1] for pos in cluster]

            self.model[ys, xs] *= -1
            self.m += 2 * seed_spin * len(cluster) * (-1) # Mise à jour de l'aimantation

        # On retourne la taille de l'amas (qu'il soit accepté ou rejeté !)
        # Cela garantit que 1 MCS correspond bien à un effort de calcul de L^2 spins visités
        return len(cluster)


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
    plt.plot(log_L, log_tau, 'o', color='royalblue', label=r"Measures $\tau_{int}$", markersize=8)

    # Droite d'ajustement
    fit_line = z * log_L + intercept
    plt.plot(log_L, fit_line, 'r--', label=rf"Fitting : $z = {z:.3f} \pm {z_err:.3f}$")

    plt.xlabel(r"$\ln(L)$", fontsize=12)
    plt.ylabel(r"$\ln(\tau)$", fontsize=12)
    #plt.title("Estimation de l'exposant dynamique $z$", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    return z, z_err


#
# Incertitudes par block bootstrap
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
    plt.plot(log_L, mean_log_tau_per_L, 'o', color='royalblue', markersize=8, label=r"Average $\ln(\tau)$")

    # Tracer la droite de régression moyenne
    fit_line = z_mean * log_L + intercept_mean
    plt.plot(log_L, fit_line, 'r--', linewidth=2, label=rf"Fit: $z = {z_mean:.3f} \pm {z_std:.3f}$")

    plt.xlabel(r"$\ln(L)$", fontsize=12)
    plt.ylabel(r"$\ln(\tau)$", fontsize=12)
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
    plt.axvline(z_mean - z_std, color='black', linestyle='dotted', linewidth=2, label=rf"$\pm 1 \sigma$ ({z_std:.3f})")
    plt.axvline(z_mean + z_std, color='black', linestyle='dotted', linewidth=2)

    plt.xlabel("Exposant dynamique estimé $z^*$", fontsize=12)
    plt.ylabel(f"Fréquence (sur {n_boot} tirages)", fontsize=12)
    plt.title("Distribution Bootstrap de l'exposant dynamique $z$", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    return z_mean, z_std


#
# Expériences
#

def experience_wolff_bootstrap(tailles_L=(50, 100, 150, 200), nb_mesures=15000, B=3000, l=200):
    """Estimation de z pour l'algorithme de Wolff, incertitudes par block bootstrap (Figure 8)."""
    tailles_L = list(tailles_L)
    beta_critique = 1.0 / 2.269185

    # === Phase de collecte des données ===
    # Dictionnaire pour stocker les séries temporelles brutes
    donnees_brutes = {}

    for L in tailles_L:
        print(f"\nSimulation Wolff pour L = {L}...")
        modele = Modele_Ising(taille=L, energie_interaction=1.0, champ=0.0)

        # 1. Thermalisation (100 MCS)
        for _ in range(100):
            spins_flipped = 0
            while spins_flipped < L**2:
                spins_flipped += modele.Wolff_step(beta_critique)

        # 2. Collecte des mesures (1 mesure par MCS)
        magnetizations = np.zeros(nb_mesures)
        for i in range(nb_mesures):
            spins_flipped = 0
            while spins_flipped < L**2:
                spins_flipped += modele.Wolff_step(beta_critique)
            magnetizations[i] = abs(modele.m)

        # Stockage des données brutes
        donnees_brutes[L] = magnetizations

    # === Phase d'analyse Bootstrap ===
    z_final, erreur_z_final = estimate_z_bootstrap_end_to_end(
        dict_magnetizations=donnees_brutes,
        tailles_L=tailles_L,
        n_boot=B,
        block_size=l
    )

    print("\n==========================================")
    print(f"Résultat final (Bootstrap) : z = {z_final:.3f} ± {erreur_z_final:.3f}")
    print("==========================================")


def experience_niedermayer(valeurs_E0=(-1/2, -1/4, 0, 1/4), tailles_L=(50, 100, 150, 200), nb_mesures=1000):
    """(Exploratoire, non présenté dans le rapport) Estimation de z pour l'algorithme de Niedermayer."""
    tailles_L = list(tailles_L)
    beta_critique = 1.0 / 2.269185
    mesures_tau_int = [[0 for _ in range(len(tailles_L))] for _ in range(len(valeurs_E0))]

    for k, E0 in enumerate(valeurs_E0):
        for j, L in enumerate(tailles_L):
            print(f"\nSimulation Niedermayer pour L = {L} et E0 = {E0}...")
            modele = Modele_Ising(taille=L, energie_interaction=1.0, champ=0.0)

            # 1. Thermalisation équivalente à 100 MCS
            for _ in range(100):
                spins_flipped = 0
                while spins_flipped < L**2:
                    spins_flipped += modele.Niedermayer_step(beta_critique, E0)

            # 2. Mesures
            magnetizations = np.zeros(nb_mesures)

            for i in range(nb_mesures):
                spins_flipped = 0
                # On définit 1 "pas" d'horloge = avoir visité au moins L^2 spins
                while spins_flipped < L**2:
                    spins_flipped += modele.Niedermayer_step(beta_critique, E0)

                magnetizations[i] = abs(modele.m)

            # 3. Calcul de tau_int
            rho = compute_autocorr_fft(magnetizations)
            tau_int = compute_tau_int(rho, c=5.0)

            print(f"-> tau_int obtenu : {tau_int:.2f} MCS")
            mesures_tau_int[k][j] = tau_int

    # Calcul final
    for i, E0 in enumerate(valeurs_E0):
        z_estime, erreur_z = estimate_dynamic_exponent_z(tailles_L, mesures_tau_int[i])
        print(f"\nRésultat final (pour E0 = {E0}) : z = {z_estime:.3f} +/- {erreur_z:.3f}")


if __name__ == "__main__":
    experience_wolff_bootstrap()
    # experience_niedermayer()
