import matplotlib.pyplot as plt
import numpy as np

def cargar_resultados(archivo):
    episodios = []
    recompensas = []
    with open(archivo, 'r') as f:
        for linea in f:
            if ',' in linea:
                ep, rw = linea.strip().split(',')
                episodios.append(int(ep))
                recompensas.append(float(rw))
    return np.array(episodios), np.array(recompensas)

# Cambia estos nombres de archivo según el gráfico que quieras hacer
archivo_q = 'qE2.txt'
archivo_s = 'sE2.txt'

episodios_q, recompensas_q = cargar_resultados(archivo_q)
episodios_s, recompensas_s = cargar_resultados(archivo_s)

# Suavizado opcional para ver mejor la tendencia
def suavizar(y, ventana=20):
    return np.convolve(y, np.ones(ventana)/ventana, mode='valid')

plt.figure(figsize=(10,6))
plt.plot(episodios_q[:len(suavizar(recompensas_q))], suavizar(recompensas_q), label='Q-Learning', color='blue')
plt.plot(episodios_s[:len(suavizar(recompensas_s))], suavizar(recompensas_s), label='SARSA', color='green')
plt.xlabel('Episodio')
plt.ylabel('Recompensa acumulada')
plt.title('Curva de aprendizaje - Cliff walking (Estocástico)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()