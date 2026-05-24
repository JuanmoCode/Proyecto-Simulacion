import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import time


# ==============================================================================
# CONFIGURACIÓN Y PARÁMETROS
# ==============================================================================

Nx = 100  # por ahora para probar 
Ny = 10

N_total = Nx * Ny  
h=1 # Tamaño del paso (ya se discretizo así que no se usa explícitamente en las fórmulas)
V0 = 1.0 # Velocidad de entrada  (cual era?????)
Re = 0.1

tol = 1e-6 # Tolerancia para convergencia
max_iter = 50 #| Número máximo de iteraciones

# damping opcional
damping = 0.5

# ==============================================================================
# OBSTÁCULOS
# ==============================================================================

beam1_x_start = 10
beam1_x_end   = 22
beam1_y_start = 5
beam1_y_end   = 7

beam2_x_start = 40
beam2_x_end   = 55
beam2_y_start = 1
beam2_y_end   = 5

print("="*60)
print("NAVIER STOKES 2D - NEWTON RAPHSON")
print("="*60)

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

# Dado un índice (i,j) y el tipo de variable ('vx' o 'vy'), devuelve el índice correspondiente en el vector x_vec
def get_idx(i, j, variable):
    idx = i * Ny + j
    if variable == 'vx':
        return idx
    else:
        return idx + N_total
    
# Dado el vector x_vec, lo convierte a las matrices vx y vy
def vector_to_grids(x_vec):
    vx = x_vec[:N_total].reshape((Nx, Ny))
    vy = x_vec[N_total:].reshape((Nx, Ny))
    return vx, vy

# Dado las matrices vx y vy, las convierte a un vector x_vec
def grids_to_vector(vx, vy):
    return np.concatenate([vx.flatten(), vy.flatten()])

# Con un índice (i,j), devuelve True si es un nodo dentro de un obstáculo, False en caso contrario
def is_obstacle(i, j):
    beam1 = (
        beam1_x_start <= i <= beam1_x_end and
        beam1_y_start <= j <= beam1_y_end
    )
    beam2 = (
        beam2_x_start <= i <= beam2_x_end and
        beam2_y_start <= j <= beam2_y_end
    )
    return beam1 or beam2


# ==============================================================================
# FUNCIÓN F
# ==============================================================================

#calcula el vector F dado el vector x_vec (que contiene vx y vy)
# el vector es x_vec = [vx_00, vx_01, ..., vx_Nx-1,Ny-1, vy_00, vy_01, ..., vy_Nx-1,Ny-1]
def calculate_F(x_vec):

    vx, vy = vector_to_grids(x_vec)

    F = np.zeros_like(x_vec)

    F_vx = F[:N_total].reshape((Nx, Ny))
    F_vy = F[N_total:].reshape((Nx, Ny))

    # --------------------------------------------------------------------------
    # NODOS INTERNOS
    # --------------------------------------------------------------------------

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            if is_obstacle(i, j):
                continue

            # DIFUSIÓN

            diff_x = (1/Re) * (
                vx[i+1,j] +
                vx[i-1,j] +
                vx[i,j+1] +
                vx[i,j-1] -
                4*vx[i,j]
            ) / 16.0

            diff_y = (1/Re) * (
                vy[i+1,j] +
                vy[i-1,j] +
                vy[i,j+1] +
                vy[i,j-1] -
                4*vy[i,j]
            ) / 16.0

            # CONVECCIÓN

            conv_x = (
                vx[i,j] * (vx[i+1,j] - vx[i-1,j]) / 8.0
                +
                vy[i,j] * (vx[i,j+1] - vx[i,j-1]) / 8.0
            )

            conv_y = (
                vx[i,j] * (vy[i+1,j] - vy[i-1,j]) / 8.0
                +
                vy[i,j] * (vy[i,j+1] - vy[i,j-1]) / 8.0
            )

            F_vx[i,j] = diff_x - conv_x
            F_vy[i,j] = diff_y - conv_y

    # --------------------------------------------------------------------------
    # ENTRADA
    # --------------------------------------------------------------------------

    for j in range(1, Ny-1):
        if is_obstacle(0, j):
            continue

        F_vx[0, j] = vx[0, j] - V0
        F_vy[0, j] = vy[0, j]

    # --------------------------------------------------------------------------
    # PAREDES
    # --------------------------------------------------------------------------

    for i in range(1, Nx-1):
        # inferior
        if not is_obstacle(i, 0):

            F_vx[i,0] = vx[i,0]
            F_vy[i,0] = vy[i,0]

        # superior
        if not is_obstacle(i, Ny-1):

            F_vx[i,Ny-1] = vx[i,Ny-1]
            F_vy[i,Ny-1] = vy[i,Ny-1]

    # --------------------------------------------------------------------------
    # OBSTÁCULOS
    # --------------------------------------------------------------------------

    for i in range(Nx):
        for j in range(Ny):

            if is_obstacle(i, j):

                F_vx[i,j] = vx[i,j]
                F_vy[i,j] = vy[i,j]

    # --------------------------------------------------------------------------
    # SALIDA
    # --------------------------------------------------------------------------

    for j in range(1, Ny-1):

        F_vx[Nx-1, j] = vx[Nx-1, j] - vx[Nx-2, j]
        F_vy[Nx-1, j] = vy[Nx-1, j] - vy[Nx-2, j]

    return F


# ==============================================================================
# JACOBIANO
# ==============================================================================
# Calcula la matriz Jacobiana J dado el vector x_vec (que contiene vx y vy)
# el Jacobiano es: J[i,j] = dF_i/dx_j
# no se calcula como malla 2d fisicamente, sino que se calcula directamente en formato vectorizado para luego convertirlo a sparse
#Jmn =∂xm/∂Fn     donde:  m = número de ecuación,  n = número de variable
def calculate_J(x_vec):
    vx, vy = vector_to_grids(x_vec)

    size = len(x_vec)

    J = lil_matrix((size, size)) # solo almacenamos los elementos no nulos para eficiencia.

    # --------------------------------------------------------------------------
    # NODOS INTERNOS
    # --------------------------------------------------------------------------

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            if is_obstacle(i, j):
                continue

            idx_vx = get_idx(i, j, 'vx')
            idx_vy = get_idx(i, j, 'vy')

            # ==============================================================
            # ECUACIÓN VX
            # ==============================================================

            J[idx_vx, idx_vx] = (
                -(4/Re)/16.0
                -
                (vx[i+1,j] - vx[i-1,j]) / 8.0
            )

            J[idx_vx, get_idx(i+1,j,'vx')] = (
                (1/Re)/16.0
                -
                vx[i,j]/8.0
            )

            J[idx_vx, get_idx(i-1,j,'vx')] = (
                (1/Re)/16.0
                +
                vx[i,j]/8.0
            )

            J[idx_vx, get_idx(i,j+1,'vx')] = (
                (1/Re)/16.0
                -
                vy[i,j]/8.0
            )

            J[idx_vx, get_idx(i,j-1,'vx')] = (
                (1/Re)/16.0
                +
                vy[i,j]/8.0
            )

            J[idx_vx, idx_vy] = (
                -(vx[i,j+1] - vx[i,j-1]) / 8.0
            )

            # ==============================================================
            # ECUACIÓN VY
            # ==============================================================

            J[idx_vy, idx_vy] = (
                -(4/Re)/16.0
                -
                (vy[i,j+1] - vy[i,j-1]) / 8.0
            )

            J[idx_vy, get_idx(i+1,j,'vy')] = (
                (1/Re)/16.0
                -
                vx[i,j]/8.0
            )

            J[idx_vy, get_idx(i-1,j,'vy')] = (
                (1/Re)/16.0
                +
                vx[i,j]/8.0
            )

            J[idx_vy, get_idx(i,j+1,'vy')] = (
                (1/Re)/16.0
                -
                vy[i,j]/8.0
            )

            J[idx_vy, get_idx(i,j-1,'vy')] = (
                (1/Re)/16.0
                +
                vy[i,j]/8.0
            )

            J[idx_vy, idx_vx] = (
                -(vy[i+1,j] - vy[i-1,j]) / 8.0
            )

    # --------------------------------------------------------------------------
    # ENTRADA
    # --------------------------------------------------------------------------

    for j in range(1, Ny-1):

        id_vx = get_idx(0, j, 'vx')
        id_vy = get_idx(0, j, 'vy')

        J[id_vx, :] = 0
        J[id_vx, id_vx] = 1.0

        J[id_vy, :] = 0
        J[id_vy, id_vy] = 1.0

    # --------------------------------------------------------------------------
    # PAREDES
    # --------------------------------------------------------------------------

    for i in range(Nx):

        # inferior

        id_vx = get_idx(i, 0, 'vx')
        id_vy = get_idx(i, 0, 'vy')

        J[id_vx, :] = 0
        J[id_vx, id_vx] = 1.0

        J[id_vy, :] = 0
        J[id_vy, id_vy] = 1.0

        # superior

        id_vx = get_idx(i, Ny-1, 'vx')
        id_vy = get_idx(i, Ny-1, 'vy')

        J[id_vx, :] = 0
        J[id_vx, id_vx] = 1.0

        J[id_vy, :] = 0
        J[id_vy, id_vy] = 1.0

    # --------------------------------------------------------------------------
    # OBSTÁCULOS
    # --------------------------------------------------------------------------

    for i in range(Nx):
        for j in range(Ny):

            if is_obstacle(i, j):

                id_vx = get_idx(i, j, 'vx')
                id_vy = get_idx(i, j, 'vy')

                J[id_vx, :] = 0
                J[id_vx, id_vx] = 1.0

                J[id_vy, :] = 0
                J[id_vy, id_vy] = 1.0

    # --------------------------------------------------------------------------
    # SALIDA
    # --------------------------------------------------------------------------

    for j in range(1, Ny-1):

        id_vx = get_idx(Nx-1, j, 'vx')
        id_vx_prev = get_idx(Nx-2, j, 'vx')

        J[id_vx, :] = 0
        J[id_vx, id_vx] = 1.0
        J[id_vx, id_vx_prev] = -1.0

        id_vy = get_idx(Nx-1, j, 'vy')
        id_vy_prev = get_idx(Nx-2, j, 'vy')

        J[id_vy, :] = 0
        J[id_vy, id_vy] = 1.0
        J[id_vy, id_vy_prev] = -1.0

    return J.tocsr() # Convertimos a formato CSR para eficiencia en el solver


# ==============================================================================
# INICIALIZACIÓN
# ==============================================================================

vx_init = np.zeros((Nx, Ny))
vy_init = np.zeros((Nx, Ny))

vx_init[0, 1:Ny-1] = V0

# obstáculos

for i in range(Nx):
    for j in range(Ny):

        if is_obstacle(i, j):

            vx_init[i,j] = 0.0
            vy_init[i,j] = 0.0

x_vec = grids_to_vector(vx_init, vy_init)

# ==============================================================================
# NEWTON
# ==============================================================================

print("\nIniciando Newton-Raphson...\n")

error_history = []

start_time = time.time()

for k in range(max_iter):

    # ----------------------------------------------------------------------
    # F
    # ----------------------------------------------------------------------

    F = calculate_F(x_vec)

    print("NaN en F:", np.any(np.isnan(F)))
    print("Inf en F:", np.any(np.isinf(F)))

    error = np.linalg.norm(F, np.inf) # norma infinito del vector F para medir el error

    error_history.append(error)

    print(f"Iteración {k+1:02d} | Error = {error:.8e}")

    if error < tol:

        print("\nCONVERGIÓ\n")
        break

    # ----------------------------------------------------------------------
    # J
    # ----------------------------------------------------------------------

    J = calculate_J(x_vec)

    print("Filas cero:", np.sum(J.getnnz(axis=1) == 0))

    # ----------------------------------------------------------------------
    # RESOLVER SISTEMA LINEAL J * delta = -F
    # ----------------------------------------------------------------------
    try:

        delta = spsolve(J, -F)

        print("NaN en delta:", np.any(np.isnan(delta))) #Nan es un valor indefinido, resultado de operaciones como 0/0 o inf - inf
        print("Inf en delta:", np.any(np.isinf(delta))) #Inf es un valor que representa una cantidad infinita, resultado de operaciones como 1/0 o overflow

        if np.any(np.isnan(delta)):

            print("\nJacobiano singular\n")
            break

    except Exception as e:

        print("\nError resolviendo sistema:\n")
        print(e)
        break

    # ----------------------------------------------------------------------
    # ACTUALIZAR SOLUCIÓN
    # ----------------------------------------------------------------------

    x_vec = x_vec + damping * delta

    vx, vy = vector_to_grids(x_vec)

    # entrada, volvemos a imponer la condición de velocidad de entrada para evitar que se corrompa por el método numérico

    vx[0, 1:Ny-1] = V0
    vy[0, 1:Ny-1] = 0.0

    # paredes, volvemos a imponer la condición de no deslizamiento para evitar que se corrompa por el método numérico

    vx[:,0] = 0.0
    vy[:,0] = 0.0

    vx[:,Ny-1] = 0.0
    vy[:,Ny-1] = 0.0

    # obstáculos, volvemos a imponer la condición de no deslizamiento para evitar que se corrompa por el método numérico

    for i in range(Nx):
        for j in range(Ny):

            if is_obstacle(i, j):

                vx[i,j] = 0.0
                vy[i,j] = 0.0

    # salida, volvemos a imponer la condición de velocidad de salida 

    vx[Nx-1, 1:Ny-1] = vx[Nx-2, 1:Ny-1]
    vy[Nx-1, 1:Ny-1] = vy[Nx-2, 1:Ny-1]

    x_vec = grids_to_vector(vx, vy)

# ==============================================================================
# FINAL
# ==============================================================================

elapsed = time.time() - start_time

vx_final, vy_final = vector_to_grids(x_vec)
print("\n============== RESULTADOS FINALES ==============\n")
print("vector vx final:")
print(vx_final)
print("\n ==============================\n")
print("vector vy final:")
print(vy_final)
print(f"\nTiempo total = {elapsed:.2f} s")

# Verificar convergencia final
if error_history[-1] > tol:
    print("\nNO convergió completamente")
# 
# ==============================================================================
# PRUEBAS FINALES
# ==============================================================================

print("\n============== TESTS ==============\n")

print("NaN en vx:", np.any(np.isnan(vx_final)))
print("NaN en vy:", np.any(np.isnan(vy_final)))

print("Max vx:", np.max(vx_final))
print("Min vx:", np.min(vx_final))

print("Max vy:", np.max(vy_final))
print("Min vy:", np.min(vy_final))

print("Norma final F:", np.linalg.norm(calculate_F(x_vec), np.inf))

""" 
# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================

velocity = np.sqrt(vx_final**2 + vy_final**2)

plt.figure(figsize=(12,5))

# magnitud

plt.subplot(1,2,1)

plt.imshow(
    velocity.T,
    origin='lower',
    cmap='viridis',
    extent=[0, Nx, 0, Ny],
    aspect='auto'
)

plt.colorbar(label='|v|')

plt.title("Magnitud velocidad")

# obstáculos

plt.axvspan(beam1_x_start, beam1_x_end,
            ymin=beam1_y_start/Ny,
            ymax=beam1_y_end/Ny,
            color='red',
            alpha=0.4)

plt.axvspan(beam2_x_start, beam2_x_end,
            ymin=beam2_y_start/Ny,
            ymax=beam2_y_end/Ny,
            color='red',
            alpha=0.4)

# quiver

plt.subplot(1,2,2)

skip = 3

X, Y = np.meshgrid(
    np.arange(0, Nx, skip),
    np.arange(0, Ny, skip)
)

U = vx_final[::skip, ::skip].T
V = vy_final[::skip, ::skip].T

plt.quiver(X, Y, U, V)

plt.title("Campo velocidades")

plt.tight_layout()
plt.show()


"""
"""

# ==============================================================================
# 5. VISUALIZACIÓN
# ==============================================================================
velocity_mag = np.sqrt(vx_final**2 + vy_final**2)

plt.figure(figsize=(12, 5))

# Mapa de calor
plt.subplot(1, 2, 1)
plt.imshow(velocity_mag.T, origin='lower', cmap='viridis', extent=[0, Nx, 0, Ny])
plt.colorbar(label='|v| (m/s)')
plt.title('Magnitud de Velocidad')
plt.xlabel('x')
plt.ylabel('y')

# Vectores (submuestreados)
skip = 4
X, Y = np.meshgrid(np.arange(0, Nx, skip), np.arange(0, Ny, skip))
U = vx_final[::skip, ::skip].T
V = vy_final[::skip, ::skip].T
plt.quiver(X, Y, U, V, color='white', scale=50)

# Dibujar Beam
rect = plt.Rectangle((beam_x_start, beam_y_start), 
                     beam_x_end-beam_x_start, 
                     beam_y_end-beam_y_start, 
                     color='gray', alpha=0.5)
plt.gca().add_patch(rect)

# Gráfico de convergencia
plt.subplot(1, 2, 2)
# Re-calculamos residuos para graficar histórico si fuera necesario, 
# pero aquí mostramos el estado final.
plt.imshow(vx_final.T, origin='lower', cmap='coolwarm')
plt.colorbar(label='Vx')
plt.title('Campo Vx Final')

plt.tight_layout()
plt.show()

"""