#%%
import numpy as np
import os
import matplotlib.pyplot as plt
#%% Cargar el archivo, omitiendo las líneas de comentarios (que comienzan con '#')
# 1: ciclo frio 002 
data_1 = np.genfromtxt(os.path.join('resultados_2025','135kHz_100dA_100Mss_NEdd002_espectro.txt'), delimiter='\t', skip_header=10)
frecuencia_1 = data_1[:, 0]  # Primera columna: Frecuencia_Hz
amplitud_1 = data_1[:, 1]    # Segunda columna: Amplitud
fase_1 = data_1[:, 2]        # Tercera columna: Fase

# 2: ciclo hot 094
data_2 = np.genfromtxt(os.path.join('resultados_2025','135kHz_100dA_100Mss_NEdd094_espectro.txt'), delimiter='\t', skip_header=10)
frecuencia_2 = data_2[:, 0]  # Primera columna: Frecuencia_Hz
amplitud_2 = data_2[:, 1]    # Segunda columna: Amplitud
fase_2 = data_2[:, 2]        # Tercera columna: Fase

# Imprimir los arrays para verificar
print("Frecuencia (Hz):", frecuencia_2)
print("Amplitud:", amplitud_2)
print("Fase:", fase_2)
#%%
# Crear el gráfico del espectro

n = len(frecuencia_1)  # Número de frecuencias
indices = np.arange(n)  # Índices para las barras

fig,((ax0,ax1))=plt.subplots(ncols=1,nrows=2,figsize=(9,5),constrained_layout=True,sharex=True)

ancho_sep = 0.1
f0 = frecuencia_1[0]/1000  

ax0.stem(indices - ancho_sep/2, amplitud_1, label='NEdd002',basefmt=' ' )
ax0.stem(indices + ancho_sep/2, amplitud_2, label='NEdd094',linefmt='C1-', markerfmt='D', basefmt=' ' )
ax0.text(1/2,1/2,f'$f_0 =$ {f0:.0f} kHz\n$H_0 = 38.4$ kA/m',transform=ax0.transAxes,fontsize=14,ha='center',bbox=dict(alpha=0.8))
ax0.legend()
# ax0.set_xlabel('Frecuencia')
ax0.set_ylabel('Amplitud')
ax0.set_ylim(0,3.5)

xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
ax0.set_xticks(indices)
ax0.set_xticklabels(xticks_labels)
ax0.grid()
ax0.set_xlim(-0.4,4.5)

ax1.stem(indices - ancho_sep/2, fase_1, label='NEdd002',basefmt=' ' )
ax1.stem(indices + ancho_sep/2, fase_2, label='NEdd094',linefmt='C1-', markerfmt='D', basefmt=' ' )
#ax1.text(3/4,1/2,f'{f0:.0f} kHz',transform=ax1.transAxes,fontsize=14,bbox=dict(alpha=0.8))
ax1.legend()
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('Amplitud')
# ax1.set_ylim(0,3.5)

xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
ax1.set_xticks(indices)
ax1.set_xticklabels(xticks_labels)
ax1.grid()
ax1.legend()
ax1.set_xlabel('Frecuencia')
ax1.set_ylabel('Fase')
ax1.set_xlim(-0.4,4.5)
ax1.set_ylim(0,15)


plt.savefig('espectro_paper.png',dpi=400)

# %%
fig,ax0=plt.subplots(constrained_layout=True)
ancho_barra = 0.2
f0 = frecuencia_1[0]/1000  

ax0.stem(indices - ancho_barra/2, amplitud_1, label='NEdd002',basefmt=' ' )
ax0.stem(indices + ancho_barra/2, amplitud_2, label='NEdd094',linefmt='C1-', markerfmt='D', basefmt=' ' )
ax0.text(3/4,1/2,f'{f0:.0f} kHz',transform=ax0.transAxes,fontsize=14,bbox=dict(alpha=0.8))
ax0.legend()
ax0.set_xlabel('Frecuencia (Hz)')
ax0.set_ylabel('Amplitud')
ax0.set_ylim(0,3.5)

xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
ax0.set_xticks(indices)
ax0.set_xticklabels(xticks_labels)
ax0.grid()
ax0.set_xlim(-0.4,5.5)


#%% Fase

fig,ax1=plt.subplots(constrained_layout=True)
ancho_barra = 0.2
f0 = frecuencia_1[0]/1000  

ax1.stem(indices - ancho_barra/2, fase_1, label='NEdd002',basefmt=' ' )
ax1.stem(indices + ancho_barra/2, fase_2, label='NEdd094',linefmt='C1-', markerfmt='D', basefmt=' ' )
#ax1.text(3/4,1/2,f'{f0:.0f} kHz',transform=ax1.transAxes,fontsize=14,bbox=dict(alpha=0.8))
ax1.legend()
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('Amplitud')
# ax1.set_ylim(0,3.5)

xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
ax1.set_xticks(indices)
ax1.set_xticklabels(xticks_labels)
ax1.grid()
ax1.legend()
ax1.set_xlabel('Frecuencia')
ax1.set_ylabel('Fase')
ax1.set_xlim(-0.4,5.5)



# %%
