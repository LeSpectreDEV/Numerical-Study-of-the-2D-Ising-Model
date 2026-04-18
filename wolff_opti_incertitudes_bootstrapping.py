import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# --- Mesures de temps d'exécution ---
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


def run_fss_gpu_bootstrapping():
    # --- Paramètres d'exécution ---
    L_list = np.array([128, 256, 512, 1024]) # Tailles à explorer
    TOTAL_SIMS_TARGET = 2000  # Nombre de mesures (simulations indépendantes)
    NUM_BATCHS_BOOT = 1000    # Nombre de tirages pour le bootstrapping
    
    VRAM_BUDGET_GB = 4.0      # Limite VRAM
    MAX_SPINS_PER_BATCH = 1_000_000_000 # Limite TDR Windows
    
    # --- Constantes physiques ---
    J_int = 1
    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))
    beta_c = 1.0 / Tc_exact
    p_add_c = 1.0 - np.exp(-2.0 * J_int * beta_c)

    # Dictionnaires pour stocker les mesures brutes de chaque taille L
    raw_M_density = {}
    raw_E_tot = {}

    print("=== DÉBUT DES SIMULATIONS SUR GPU ===")
    
    for L in L_list:
        print(f"\n-> Grille {L}x{L}")

        start_time = time.time() # mesure du temps d'exécution
        
        # 1. Calcul du découpage (Batching)
        bytes_per_spin = 9
        bytes_per_sim = bytes_per_spin * (L**2)
        max_sims_vram = int((VRAM_BUDGET_GB * 1024**3) / bytes_per_sim)
        max_sims_compute = int(MAX_SPINS_PER_BATCH / (L**2))
        max_sims_autorisees = max(1, min(max_sims_vram, max_sims_compute))
        
        batch_size = min(TOTAL_SIMS_TARGET, max_sims_autorisees)
        num_batches = int(np.ceil(TOTAL_SIMS_TARGET / batch_size))
        actual_total = batch_size * num_batches
        
        print(f"Mémoire/sim : {bytes_per_sim / 1024**2:.2f} Mo")
        print(f"Batching : {num_batches} itérations de {batch_size} simulations (Total : {actual_total})")

        # 2. Initialisation de Taichi
        ti.init(arch=ti.vulkan)
        
        spins = ti.field(dtype=ti.i8, shape=(batch_size, L, L))
        queue_x = ti.field(dtype=ti.i16, shape=(batch_size, L * L))
        queue_y = ti.field(dtype=ti.i16, shape=(batch_size, L * L))
        visited = ti.field(dtype=ti.i32, shape=(batch_size, L, L))
        visit_tag = ti.field(dtype=ti.i32, shape=batch_size)
        
        # NOUVEAU : Allocation pour l'énergie
        M_out = ti.field(dtype=ti.i32, shape=batch_size)
        E_out = ti.field(dtype=ti.i32, shape=batch_size)

        @ti.kernel
        def init_simulations():
            for sim, i, j in ti.ndrange(batch_size, L, L):
                spins[sim, i, j] = ti.cast(1, ti.i8) # T=0 pour thermalisation rapide
                visited[sim, i, j] = 0
            for sim in range(batch_size):
                visit_tag[sim] = 0

        @ti.kernel
        def wolff_multi_step(num_steps: ti.i32):
            for sim in range(batch_size):
                for _ in range(num_steps):
                    tag = visit_tag[sim] + 1
                    visit_tag[sim] = tag
                    
                    seed_i = ti.cast(ti.random(ti.f32) * L, ti.i32) % L
                    seed_j = ti.cast(ti.random(ti.f32) * L, ti.i32) % L
                    
                    target_spin = spins[sim, seed_i, seed_j]
                    
                    queue_x[sim, 0] = ti.cast(seed_i, ti.i16)
                    queue_y[sim, 0] = ti.cast(seed_j, ti.i16)
                    visited[sim, seed_i, seed_j] = tag
                    spins[sim, seed_i, seed_j] = ti.cast(-target_spin, ti.i8)
                    
                    head = 0
                    tail = 1
                    
                    while head < tail:
                        cx = queue_x[sim, head]
                        cy = queue_y[sim, head]
                        head += 1
                        
                        dx = [0, 0, 1, -1]
                        dy = [1, -1, 0, 0]
                        
                        for d in ti.static(range(4)):
                            nx = (cx + dx[d] + L) % L
                            ny = (cy + dy[d] + L) % L
                            
                            if visited[sim, nx, ny] != tag:
                                if spins[sim, nx, ny] == target_spin:
                                    if ti.random(ti.f32) < p_add_c:
                                        visited[sim, nx, ny] = tag
                                        spins[sim, nx, ny] = ti.cast(-target_spin, ti.i8)
                                        queue_x[sim, tail] = ti.cast(nx, ti.i16)
                                        queue_y[sim, tail] = ti.cast(ny, ti.i16)
                                        tail += 1

        @ti.kernel
        def compute_observables():
            """Calcule l'aimantation ET l'énergie pour chaque grille."""
            for sim in range(batch_size):
                m = 0
                e = 0
                for i, j in ti.ndrange(L, L):
                    s = spins[sim, i, j]
                    m += s
                    
                    # Pour éviter de compter les liaisons en double, 
                    # on ne regarde que les voisins de Droite et d'En-Bas.
                    s_right = spins[sim, (i + 1) % L, j]
                    s_down  = spins[sim, i, (j + 1) % L]
                    
                    e += -J_int * s * (s_right + s_down)
                    
                M_out[sim] = m
                E_out[sim] = e

        # 3. Boucle d'exécution des batchs
        batch_M_densities = []
        batch_E_totals = []
        
        nb_etapes_totales = 100 #int(10 * L)
        etapes_par_bloc = 1 # Respiration GPU
        nb_blocs = max(1, nb_etapes_totales // etapes_par_bloc)
        
        print(f"Nombre de pas de l'algorithme : {nb_etapes_totales}.")

        for b in range(num_batches):
            print(f"  Batch {b+1}/{num_batches}...", end="\r")
            init_simulations()
            
            for _ in range(nb_blocs):
                wolff_multi_step(etapes_par_bloc)
                ti.sync()
            
            compute_observables()
            
            # Récupération de l'aimantation (densité) et de l'énergie (totale)
            M_tot_array = M_out.to_numpy()
            E_tot_array = E_out.to_numpy()
            
            batch_M_densities.extend(M_tot_array / (L**2))
            batch_E_totals.extend(E_tot_array)
            
        print(f"  {num_batches} batch(s) terminés !                     ")
        print(f"Temps d'exécution : {format_execution_time(time.time()-start_time)}.")
        
        # Stockage de l'ensemble (on coupe à TOTAL_SIMS_TARGET exact)
        raw_M_density[L] = np.array(batch_M_densities[:TOTAL_SIMS_TARGET])
        raw_E_tot[L] = np.array(batch_E_totals[:TOTAL_SIMS_TARGET])
        
        ti.reset() # Libération de la VRAM


    print("\n=== CALCUL DES INCERTITUDES PAR BOOTSTRAPPING (CPU) ===")

    # Filtrage des effets de bords (On garde L >= 16)
    masque = L_list >= 16
    L_fit = L_list[masque]
    
    # --- A. Mesures sur l'échantillon complet ---
    Mabs_mes = np.zeros(len(L_fit))
    Chi_mes = np.zeros(len(L_fit))
    C_mes = np.zeros(len(L_fit))
    
    for idx, L in enumerate(L_fit):
        M = raw_M_density[L]
        E = raw_E_tot[L]
        
        Mabs_mes[idx] = np.mean(np.abs(M))
        Chi_mes[idx] = beta_c * (L**2) * (np.mean(M**2) - np.mean(np.abs(M))**2)
        C_mes[idx] = (beta_c**2 / L**2) * (np.mean(E**2) - np.mean(E)**2)
        
    beta_sur_nu_mes = -np.polyfit(np.log(L_fit), np.log(Mabs_mes), 1)[0]
    gamma_sur_nu_mes = np.polyfit(np.log(L_fit), np.log(Chi_mes), 1)[0]

    # --- B. Bootstrapping ---
    beta_sur_nu_boot = np.zeros(NUM_BATCHS_BOOT)
    gamma_sur_nu_boot = np.zeros(NUM_BATCHS_BOOT)
    
    # Matrice de tirages aléatoires (Indices)
    bootstraping_indices = np.random.choice(TOTAL_SIMS_TARGET, size=(NUM_BATCHS_BOOT, TOTAL_SIMS_TARGET), replace=True)

    for b in tqdm(range(NUM_BATCHS_BOOT), desc="Génération des batchs FSS"):
        Mabs_tc = np.zeros(len(L_fit))
        chi_tc = np.zeros(len(L_fit))
        
        for idx, L in enumerate(L_fit):
            M_batch = raw_M_density[L][bootstraping_indices[b]]
            
            Mabs_tc[idx] = np.mean(np.abs(M_batch))
            chi_tc[idx] = beta_c * (L**2) * (np.mean(M_batch**2) - np.mean(np.abs(M_batch))**2)
            
        beta_sur_nu_boot[b] = -np.polyfit(np.log(L_fit), np.log(Mabs_tc), 1)[0]
        gamma_sur_nu_boot[b] = np.polyfit(np.log(L_fit), np.log(chi_tc), 1)[0]

    sigma_beta = np.std(beta_sur_nu_boot, ddof=1)
    sigma_gamma = np.std(gamma_sur_nu_boot, ddof=1)

    print("\n--- RÉSULTATS FINAUX ---")
    print(f"β/ν = {beta_sur_nu_mes:.5f} ± {sigma_beta:.5f}  (Théorie : 0.125)")
    print(f"γ/ν = {gamma_sur_nu_mes:.5f} ± {sigma_gamma:.5f}  (Théorie : 1.750)")

    """# --- Affichage Graphique ---
    plt.figure(figsize=(9, 6))
    plt.loglog(L_fit, Mabs_mes, 'o-', label=f"Aimantation ($\\beta/\\nu \\approx$ {beta_sur_nu_mes:.3f})")
    plt.loglog(L_fit, Chi_mes, 's-', label=f"Susceptibilité ($\\gamma/\\nu \\approx$ {gamma_sur_nu_mes:.3f})")
    plt.xlabel("Taille du réseau $L$")
    plt.ylabel("Observables")
    plt.title("Finite Size Scaling à $T_c$ avec erreurs Bootstrap")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()"""
    # --- Affichage Graphique avec Droites de Régression ---
    plt.figure(figsize=(10, 7))

    # 1. Cas de l'Aimantation (Exposant beta/nu)
    plt.loglog(L_fit, Mabs_mes, 'ob', label="Aimantation (mesures)")
    # Calcul de la droite de régression
    coeffs_m = np.polyfit(np.log(L_fit), np.log(Mabs_mes), 1)
    fit_m = np.exp(np.polyval(coeffs_m, np.log(L_fit)))
    plt.loglog(L_fit, fit_m, '--b', alpha=0.8, label=f"Fit M (pente={-coeffs_m[0]:.3f})")

    # 2. Cas de la Susceptibilité (Exposant gamma/nu)
    plt.loglog(L_fit, Chi_mes, 'sr', label="Susceptibilité (mesures)")
    # Calcul de la droite de régression
    coeffs_chi = np.polyfit(np.log(L_fit), np.log(Chi_mes), 1)
    fit_chi = np.exp(np.polyval(coeffs_chi, np.log(L_fit)))
    plt.loglog(L_fit, fit_chi, '--r', alpha=0.8, label=f"Fit $\chi$ (pente={coeffs_chi[0]:.3f})")

    # Cosmétique du graphique
    plt.xlabel("Taille du réseau $L$ (Log)", fontsize=12)
    plt.ylabel("Observables (Log)", fontsize=12)
    plt.title("Analyse de Finite Size Scaling : Droites de Régression", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    total_start_time = time.time() # mesure du temps d'exécution
    run_fss_gpu_bootstrapping()
    print(f"\n Temps d'exécution global : {format_execution_time(time.time()-total_start_time)}.")
