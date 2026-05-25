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
V0 = 1 # Velocidad de entrada  (cual era?????)
Re = 0.1 # Número de Reynolds, controla la relación entre convección y difusión. 
#         Valores bajos = flujo laminar, valores altos = flujo turbulento.

tol = 1e-6 # Tolerancia para convergencia
max_iter = 50 #| Número máximo de iteraciones

# damping 
damping = 1.0

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



