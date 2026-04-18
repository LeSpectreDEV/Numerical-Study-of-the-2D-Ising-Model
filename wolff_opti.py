import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Initialisation pour AMD sous Windows
ti.init(arch=ti.vulkan)

# --- Paramètres Globaux ---
MAX_L = 128          # Taille maximale du réseau autorisée en VRAM
NUM_SIMS = 2000      # Nombre de simulations PARALLÈLES (remplace nb_mesures du CPU)
J = 1.0
Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))
beta_c = 1.0 / Tc_exact
p_add_c = 1.0 - np.exp(-2.0 * J * beta_c)

# --- Allocation Mémoire GPU (VRAM) ---
# On alloue pour la taille maximale. NUM_SIMS grilles indépendantes.
spins = ti.field(dtype=ti.i8, shape=(NUM_SIMS, MAX_L, MAX_L))

# Files d'attente (queues) pour le parcours en largeur (BFS) de Wolff
queue_x = ti.field(dtype=ti.i16, shape=(NUM_SIMS, MAX_L * MAX_L))
queue_y = ti.field(dtype=ti.i16, shape=(NUM_SIMS, MAX_L * MAX_L))

# Optimisation critique : visit_tag évite de remettre à zéro le tableau 'visited' à chaque pas (O(L^2))
visited = ti.field(dtype=ti.i32, shape=(NUM_SIMS, MAX_L, MAX_L))
visit_tag = ti.field(dtype=ti.i32, shape=NUM_SIMS)

# Sorties pour les mesures
M_out = ti.field(dtype=ti.i32, shape=NUM_SIMS)


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


"""
# Intialisation aléatoire
@ti.kernel
def init_simulations(current_L: ti.i32):
    '''Initialise aléatoirement toutes les grilles et remet les tags à zéro.'''
    for sim, i, j in ti.ndrange(NUM_SIMS, current_L, current_L):
        if ti.random(ti.f32) < 0.5:
            spins[sim, i, j] = ti.cast(1, ti.i8)
        else:
            spins[sim, i, j] = ti.cast(-1, ti.i8)
        visited[sim, i, j] = 0
        
    for sim in range(NUM_SIMS):
        visit_tag[sim] = 0"""

# Initialisation à T = 0
@ti.kernel
def init_simulations(current_L: ti.i32):
    """Initialisation à T=0 (état ordonné) pour accélérer la thermalisation Wolff."""
    for sim, i, j in ti.ndrange(NUM_SIMS, current_L, current_L):
        spins[sim, i, j] = ti.cast(1, ti.i8) # Tous les spins alignés
        visited[sim, i, j] = 0
        
    for sim in range(NUM_SIMS):
        visit_tag[sim] = 0

@ti.kernel
def wolff_multi_step(num_steps: ti.i32, p_add: ti.f32, current_L: ti.i32):
    """Exécute des pas de Wolff sur les 2000 univers en parallèle massif."""
    for sim in range(NUM_SIMS): # <- Taichi parallélise cette boucle sur les cœurs GPU !
        for _ in range(num_steps):
            # Incrémente le tag unique pour cette itération
            tag = visit_tag[sim] + 1
            visit_tag[sim] = tag
            
            # 1. Graine aléatoire
            seed_i = ti.cast(ti.random(ti.f32) * current_L, ti.i32) % current_L
            seed_j = ti.cast(ti.random(ti.f32) * current_L, ti.i32) % current_L
            
            target_spin = spins[sim, seed_i, seed_j]
            
            # 2. Initialisation de la file et du premier spin
            queue_x[sim, 0] = ti.cast(seed_i, ti.i16)
            queue_y[sim, 0] = ti.cast(seed_j, ti.i16)
            visited[sim, seed_i, seed_j] = tag
            spins[sim, seed_i, seed_j] = -target_spin
            
            head = 0
            tail = 1
            
            # 3. Parcours BFS pour construire le cluster
            while head < tail:
                cx = queue_x[sim, head]
                cy = queue_y[sim, head]
                head += 1
                
                # Directions pour les 4 voisins
                dx = [0, 0, 1, -1]
                dy = [1, -1, 0, 0]
                
                for d in ti.static(range(4)):
                    nx = (cx + dx[d] + current_L) % current_L
                    ny = (cy + dy[d] + current_L) % current_L
                    
                    # Condition : non visité ET même signe
                    if visited[sim, nx, ny] != tag:
                        if spins[sim, nx, ny] == target_spin:
                            if ti.random(ti.f32) < p_add:
                                visited[sim, nx, ny] = tag       # Marque comme visité
                                spins[sim, nx, ny] = -target_spin # Retourne le spin          
                                queue_x[sim, tail] = ti.cast(nx, ti.i16) # Ajoute à la file
                                queue_y[sim, tail] = ti.cast(ny, ti.i16)
                                tail += 1

@ti.kernel
def compute_magnetization(current_L: ti.i32):
    """Calcule l'aimantation totale de chaque simulation."""
    for sim in range(NUM_SIMS):
        m = 0
        for i, j in ti.ndrange(current_L, current_L):
            m += spins[sim, i, j]
        M_out[sim] = m

# version initiale
"""
def run_fss_gpu():
    L_list = np.array([128, 256, 512, 1024]) # Tailles à explorer
    
    M_tc_list = []
    Chi_tc_list = []
    
    print(f"Lancement du Finite Size Scaling sur GPU (Wolff)")
    print(f"Simulations par taille L : {NUM_SIMS} en parallèle. \n")
    
    for L in L_list:
        print(f"-> Traitement de la grille {L}x{L}...")
        
        # 1. Initialisation VRAM
        init_simulations(L)
        
        # 2. Thermalisation par petits blocs pour éviter le Timeout Windows (TDR)
        nb_etapes_totales = int(10 * L) # nombre d'étapes de Wolff proportionnel à la taille du réseau
        etapes_par_bloc = max(1, nb_etapes_totales // 20)  # Le GPU fait etapes_par_bloc pas de Wolff, puis rend la main

        for _ in range(nb_etapes_totales // etapes_par_bloc):
            wolff_multi_step(etapes_par_bloc, p_add_c, L)
            ti.sync()  # Force la synchronisation avec l'ordinateur pour "valider" la respiration
        
        # 3. Mesures (on utilise les NUM_SIMS grilles indépendantes comme nos échantillons statistiques)
        compute_magnetization(L)
        
        # Rapatriement des aimantations vers le CPU (NumPy)
        # Chaque élément est l'aimantation totale d'une simulation
        M_tot_array = M_out.to_numpy() 
        
        # Densité d'aimantation absolue par spin : |m|
        m_abs_array = np.abs(M_tot_array) / (L**2)
        
        # Moyennes statistiques sur l'ensemble
        M_abs_mean = np.mean(m_abs_array)
        M_sq_mean = np.mean(m_abs_array**2)
        
        # Susceptibilité selon votre notebook
        Chi = beta_c * (L**2) * (M_sq_mean - M_abs_mean**2)
        
        M_tc_list.append(M_abs_mean)
        Chi_tc_list.append(Chi)

    # --- Extraction des exposants (Même logique que votre notebook) ---
    logL = np.log(L_list)
    
    beta_sur_nu = -np.polyfit(logL, np.log(M_tc_list), 1)[0]
    gamma_sur_nu = np.polyfit(logL, np.log(Chi_tc_list), 1)[0]

    print("\n --- RÉSULTATS ---")
    print(f"β/ν (Théorique ≈ 0.125) : Obtenu = {beta_sur_nu:.4f}")
    print(f"γ/ν (Théorique ≈ 1.750) : Obtenu = {gamma_sur_nu:.4f}")

    # --- Graphique ---
    plt.figure(figsize=(8, 5))
    plt.loglog(L_list, M_tc_list, 'o-', label=f"Magnétisation (Pente = {-beta_sur_nu:.2f})")
    plt.loglog(L_list, Chi_tc_list, 's-', label=f"Susceptibilité (Pente = {gamma_sur_nu:.2f})")
    plt.xlabel("L (Taille du réseau)")
    plt.ylabel("Observable")
    plt.title("Finite Size Scaling à $T_c$ - GPU Wolff")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Enregistrement du fichier plutôt que plt.show() pour l'environnement GPU
    plt.savefig("fss_wolff_gpu.png")
    print("Graphique sauvegardé sous 'fss_wolff_gpu.png'")

if __name__ == "__main__":
    run_fss_gpu()
    
"""

# version avec gestion améliorée de la VRAM

def run_fss_gpu_batched():
    # Paramètres d'exécution
    L_list = np.array([16, 32, 64, 128, 256, 512])#, 1024])
    TOTAL_SIMS_TARGET = 2000
    VRAM_BUDGET_GB = 4.0 # Limite stricte à 3 Go de VRAM
    
    # Paramètres physiques
    J = 1.0
    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))
    beta_c = 1.0 / Tc_exact
    p_add_c = 1.0 - np.exp(-2.0 * J * beta_c)

    M_tc_list = []
    Chi_tc_list = []

    for L in L_list:
        print(f"\n=== Traitement de la grille {L}x{L} ===")
        
        start_time = time.time() # mesure du temps d'exécution

        # 1. Calcul de l'encombrement VRAM et paramétrage des batchs
        bytes_per_spin = 1 + 2 + 2 + 4 # i8 + i16 + i16 + i32 = 9 octets
        bytes_per_sim = bytes_per_spin * (L**2)
        
        # Limite de mémoire (Ce que vous aviez déjà)
        max_sims_vram = int((VRAM_BUDGET_GB * 1024**3) / bytes_per_sim)
        
        # NOUVEAU : Limite de temps de calcul (TDR Windows)
        # On interdit de calculer plus de 15 millions de spins en même temps.
        # Vous pouvez monter/descendre cette valeur si ça plante encore ou si c'est trop lent.
        MAX_SPINS_PER_BATCH = 2_000_000_000 
        max_sims_compute = int(MAX_SPINS_PER_BATCH / (L**2))
        
        # On prend la limite la plus stricte (toujours au moins 1)
        max_sims_autorisees = max(1, min(max_sims_vram, max_sims_compute))
        
        batch_size = min(TOTAL_SIMS_TARGET, max_sims_autorisees)
        num_batches = int(np.ceil(TOTAL_SIMS_TARGET / batch_size))
        actual_total = batch_size * num_batches
        
        print(f"Mémoire/sim : {bytes_per_sim / 1024**2:.2f} Mo")
        print(f"Batching    : {num_batches} itérations de {batch_size} simulations (Total ciblé : {actual_total}).")

        # 2. Initialisation Taichi propre pour CE sous-réseau
        ti.init(arch=ti.vulkan)
        
        # Allocation des champs pour la taille du batch (et non plus la taille totale !)
        spins = ti.field(dtype=ti.i8, shape=(batch_size, L, L))
        queue_x = ti.field(dtype=ti.i16, shape=(batch_size, L * L))
        queue_y = ti.field(dtype=ti.i16, shape=(batch_size, L * L))
        visited = ti.field(dtype=ti.i32, shape=(batch_size, L, L))
        visit_tag = ti.field(dtype=ti.i32, shape=batch_size)
        M_out = ti.field(dtype=ti.i32, shape=batch_size)

        # 3. Définition des noyaux (Ils sont définis ICI pour "voir" les allocations ci-dessus)
        @ti.kernel
        def init_simulations():
            for sim, i, j in ti.ndrange(batch_size, L, L):
                spins[sim, i, j] = ti.cast(1, ti.i8) # Initialisation à froid (T=0)
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
        def compute_magnetization():
            for sim in range(batch_size):
                m = 0
                for i, j in ti.ndrange(L, L):
                    m += spins[sim, i, j]
                M_out[sim] = m

        # 4. Exécution séquentielle des batchs
        all_m_abs = []
        
        # Adaptation de la thermalisation à la taille du réseau
        nb_etapes_totales = 100 #int(10 * L)
        etapes_par_bloc = max(1, nb_etapes_totales // 500)

        print(f"Nombre de pas de l'algorithme : {nb_etapes_totales}.")
        
        for b in range(num_batches):
            print(f"  -> Batch {b+1}/{num_batches} en cours...", end="\r")
            init_simulations()
            
            for _ in range(nb_etapes_totales // etapes_par_bloc):
                wolff_multi_step(etapes_par_bloc)
                ti.sync()
            
            compute_magnetization()
            
            # On rapatrie les aimantations de ce batch vers le CPU
            batch_M_tot = M_out.to_numpy()
            batch_m_abs = np.abs(batch_M_tot) / (L**2)
            all_m_abs.extend(batch_m_abs) # On allonge la liste de nos échantillons
            
        print(f"  -> {num_batches} batch(s) terminés !                     ")
        print(f"Temps d'exécution : {format_execution_time(time.time()-start_time)}.")
        
        # 5. Agréger et calculer les moyennes sur l'ensemble des données
        all_m_abs_arr = np.array(all_m_abs)
        M_abs_mean = np.mean(all_m_abs_arr)
        M_sq_mean = np.mean(all_m_abs_arr**2)
        Chi = beta_c * (L**2) * (M_sq_mean - M_abs_mean**2)
        
        M_tc_list.append(M_abs_mean)
        Chi_tc_list.append(Chi)
        
        # 6. NETTOYAGE VRAM CRUCIAL
        # On détruit l'environnement Taichi actuel pour libérer les 3 Go alloués
        ti.reset()

    # --- Extraction des exposants (Même code que précédemment) ---
    L_array = np.array(L_list)
    M_array = np.array(M_tc_list)
    Chi_array = np.array(Chi_tc_list)

    # Filtrage asymptotique (on garde L >= 16)
    masque = L_array >= 16
    L_fit = L_array[masque]
    M_fit = M_array[masque]
    Chi_fit = Chi_array[masque]

    beta_sur_nu = -np.polyfit(np.log(L_fit), np.log(M_fit), 1)[0]
    gamma_sur_nu = np.polyfit(np.log(L_fit), np.log(Chi_fit), 1)[0]

    print("\n--- RÉSULTATS FINAUX ---")
    print(f"β/ν obtenu = {beta_sur_nu:.4f} (Théorie : 0.125)")
    print(f"γ/ν obtenu = {gamma_sur_nu:.4f} (Théorie : 1.750)")

    # --- Graphique ---
    plt.figure(figsize=(8, 5))
    plt.loglog(L_list, M_tc_list, 'o-', label=f"Magnétisation (Pente = {-beta_sur_nu:.4f})")
    plt.loglog(L_list, Chi_tc_list, 's-', label=f"Susceptibilité (Pente = {gamma_sur_nu:.4f})")
    plt.xlabel("L (Taille du réseau)")
    plt.ylabel("Observable")
    plt.title("Finite Size Scaling à $T_c$ - GPU Wolff")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Enregistrement du fichier plutôt que plt.show() pour l'environnement GPU
    plt.savefig("fss_wolff_gpu.png")
    print("Graphique sauvegardé sous 'fss_wolff_gpu.png'")

if __name__ == "__main__":
    total_start_time = time.time() # mesure du temps d'exécution
    run_fss_gpu_batched()
    print(f"\n Temps d'exécution global : {format_execution_time(time.time()-total_start_time)}.")
