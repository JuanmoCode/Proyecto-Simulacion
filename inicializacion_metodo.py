from algo_metodo_newton import *

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
# PRUEBAS CONFIGURABLES
# ==============================================================================

# Reynolds a probar
reynolds_tests = [0.01, 0.1, 1.0, 10.0]

# Guardar resultados
results = []

# ==============================================================================
# LOOP DE PRUEBAS
# ==============================================================================

for Re in reynolds_tests:

    print("\n" + "="*70)
    print(f"PRUEBA CON REYNOLDS = {Re}")
    print("="*70)

    # reiniciar solución inicial para cada prueba

    vx_init = np.zeros((Nx, Ny))
    vy_init = np.zeros((Nx, Ny))

    vx_init[0, 1:Ny-1] = V0

    for i in range(Nx):
        for j in range(Ny):

            if is_obstacle(i, j):

                vx_init[i,j] = 0.0
                vy_init[i,j] = 0.0

    x_vec = grids_to_vector(vx_init, vy_init)

    # historial de error para esta simulación

    error_history = []

    # medir tiempo

    start_time = time.time()

    converged = False

    # ==========================================================================
    # NEWTON
    # ==========================================================================

    for k in range(max_iter):

        # ----------------------------------------------------------------------
        # F
        # ----------------------------------------------------------------------

        F = calculate_F(x_vec)
        print("NaN en F:", np.any(np.isnan(F)))
        print("Inf en F:", np.any(np.isinf(F)))

        
        error = np.linalg.norm(F, np.inf)

        error_history.append(error)

        print(f"Iteración {k+1:02d} | Error = {error:.8e}")

        # prueba de convergencia

        if error < tol:

            print("\nCONVERGIÓ\n")

            converged = True

            break

        # ----------------------------------------------------------------------
        # J
        # ----------------------------------------------------------------------

        J = calculate_J(x_vec)

        print("Filas cero:", np.sum(J.getnnz(axis=1) == 0))

        # ----------------------------------------------------------------------
        # CONDICIÓN DEL JACOBIANO
        # ----------------------------------------------------------------------

        try:

            cond_J = np.linalg.cond(J.toarray())
            

            print(f"Condición Jacobiano: {cond_J:.4e}")

        except:

            cond_J = np.inf

            print("No se pudo calcular condición del Jacobiano")

        # ----------------------------------------------------------------------
        # RESOLVER SISTEMA
        # ----------------------------------------------------------------------

        try:

            delta = spsolve(J, -F)

            print("NaN en delta:", np.any(np.isnan(delta)))
            print("Inf en delta:", np.any(np.isinf(delta)))

            if np.any(np.isnan(delta)):

                print("\nJacobiano singular\n")

                break

        except Exception as e:

            print("\nError resolviendo sistema:\n")
            print(e)

            break

        # ----------------------------------------------------------------------
        # NORMA DE DELTA
        # ----------------------------------------------------------------------

        delta_norm = np.linalg.norm(delta, np.inf)

        print(f"Norma delta = {delta_norm:.8e}")

        # ----------------------------------------------------------------------
        # ACTUALIZAR SOLUCIÓN
        # ----------------------------------------------------------------------

        x_vec = x_vec + delta

        vx, vy = vector_to_grids(x_vec)

        # entrada

        vx[0, 1:Ny-1] = V0
        vy[0, 1:Ny-1] = 0.0

        # paredes

        vx[:,0] = 0.0
        vy[:,0] = 0.0

        vx[:,Ny-1] = 0.0
        vy[:,Ny-1] = 0.0

        # obstáculos

        for i in range(Nx):
            for j in range(Ny):

                if is_obstacle(i, j):

                    vx[i,j] = 0.0
                    vy[i,j] = 0.0

        # salida

        vx[Nx-1, 1:Ny-1] = vx[Nx-2, 1:Ny-1]
        vy[Nx-1, 1:Ny-1] = vy[Nx-2, 1:Ny-1]

        x_vec = grids_to_vector(vx, vy)

    # ==========================================================================
    # FINAL DE ESTA PRUEBA
    # ==========================================================================

    elapsed = time.time() - start_time

    vx_final, vy_final = vector_to_grids(x_vec)

    final_residual = np.linalg.norm(calculate_F(x_vec), np.inf)

    print(f"\nTiempo total = {elapsed:.2f} s")

    if not converged:

        print("\nNO convergió completamente")

    # ==========================================================================
    # TESTS FINALES
    # ==========================================================================

    print("\n============== TESTS ==============\n")

    # historial de error

    for i in range(len(error_history)):

        print(f"Iteración {i+1:02d} | Error = {error_history[i]:.8e}")

    # NaN

    print("\n----- VALIDACIÓN NaN/Inf vectores finales -----\n")

    print("NaN en vx:", np.any(np.isnan(vx_final)))
    print("NaN en vy:", np.any(np.isnan(vy_final)))

    print("Inf en vx:", np.any(np.isinf(vx_final)))
    print("Inf en vy:", np.any(np.isinf(vy_final)))

    # rangos físicos

    print("\n----- RANGOS DE VELOCIDAD -----\n")

    print("Max vx:", np.max(vx_final))
    print("Min vx:", np.min(vx_final))

    print("Max vy:", np.max(vy_final))
    print("Min vy:", np.min(vy_final))

    # residual final

    print("\n----- RESIDUAL FINAL -----\n")

    print("Norma final F:", final_residual)

    # guardar resultados

    results.append({
        "Re": Re,
        "converged": converged,
        "iterations": len(error_history),
        "final_error": final_residual,
        "time": elapsed,
        "max_vx": np.max(vx_final),
        "max_vy": np.max(vy_final),
        "final_vx": vx_final,
        "final_vy": vy_final
    })
    print(f"FIN PRUEBA CON REYNOLDS = {Re}")
    print("\n\n\n\n" + "="*70)
    

# ==============================================================================
# RESUMEN GLOBAL
# ==============================================================================

print("\n" + "="*70)
print("RESUMEN GLOBAL")
print("="*70)

for r in results:

    print(f"""
Reynolds:        {r['Re']}
Convergió:       {r['converged']}
Iteraciones:     {r['iterations']}
Error final:     {r['final_error']:.8e}
Tiempo:          {r['time']:.4f} s
Max vx:          {r['max_vx']:.4f}
Max vy:          {r['max_vy']:.4f}
""")


"""
# ==============================================================================
# GRÁFICA DE CONVERGENCIA FINAL
# ==============================================================================

plt.figure(figsize=(8,5))

plt.plot(error_history, marker='o')

plt.yscale('log')

plt.xlabel("Iteración")
plt.ylabel("Norma infinito del residual")
plt.title(f"Convergencia Newton-Raphson (Re = {Re})")

plt.grid(True)

plt.show()
"""


# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================

# magnitud de la velocidad

velocity = np.sqrt(vx_final**2 + vy_final**2)

plt.figure(figsize=(12,5))

# ==============================================================================
# MAPA DE CALOR
# ==============================================================================

plt.subplot(1,2,1)

plt.imshow(
    velocity.T,
    origin='lower',
    cmap='viridis',
    extent=[0, Nx, 0, Ny],
    aspect='auto'
)

plt.colorbar(label='|v|')

plt.title(f"Magnitud de Velocidad (Re = {Re})")

plt.xlabel("x")
plt.ylabel("y")

# ------------------------------------------------------------------------------
# Obstáculo 1
# ------------------------------------------------------------------------------

plt.axvspan(
    beam1_x_start,
    beam1_x_end,
    ymin=beam1_y_start/Ny,
    ymax=beam1_y_end/Ny,
    color='red',
    alpha=0.4
)

# ------------------------------------------------------------------------------
# Obstáculo 2
# ------------------------------------------------------------------------------

plt.axvspan(
    beam2_x_start,
    beam2_x_end,
    ymin=beam2_y_start/Ny,
    ymax=beam2_y_end/Ny,
    color='red',
    alpha=0.4
)

# ==============================================================================
# CAMPO VECTORIAL
# ==============================================================================

plt.subplot(1,2,2)

skip = 3

X, Y = np.meshgrid(
    np.arange(0, Nx, skip),
    np.arange(0, Ny, skip)
)

U = vx_final[::skip, ::skip].T
V = vy_final[::skip, ::skip].T

plt.quiver(X, Y, U, V)

plt.title(f"Campo de Velocidades (Re = {Re})")

plt.xlabel("x")
plt.ylabel("y")

# ==============================================================================
# DIBUJAR OBSTÁCULOS
# ==============================================================================

plt.axvspan(
    beam1_x_start,
    beam1_x_end,
    ymin=beam1_y_start/Ny,
    ymax=beam1_y_end/Ny,
    color='red',
    alpha=0.4
)

plt.axvspan(
    beam2_x_start,
    beam2_x_end,
    ymin=beam2_y_start/Ny,
    ymax=beam2_y_end/Ny,
    color='red',
    alpha=0.4
)

# ==============================================================================
# AJUSTAR Y MOSTRAR
# ==============================================================================

plt.tight_layout()

plt.show()
















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