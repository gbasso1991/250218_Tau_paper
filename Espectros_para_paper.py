#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
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
#%% amplitud y fase


fig,((ax0,ax1))=plt.subplots(ncols=1,nrows=2,figsize=(7,4),constrained_layout=True,sharex=True)

n = len(frecuencia_1)  # Número de frecuencias
indices = np.arange(n)  # Índices para las barras
ancho_sep = 0.1
f0 = frecuencia_1[0]/1000

ax0.stem(indices - ancho_sep/2, amplitud_1/max(amplitud_1), label='-20ºC',basefmt=' ' )
ax0.stem(indices + ancho_sep/2, amplitud_2/max(amplitud_2), label='20ºC',linefmt='C3-', markerfmt='D', basefmt=' ' )
#ax0.text(1/2,1/2,f'$f_0 =$ {f0:.0f} kHz\n$H_0 = 38.4$ kA/m',transform=ax0.transAxes,fontsize=14,ha='center',bbox=dict(alpha=0.8))
ax0.legend(ncol=2)
# ax0.set_xlabel('Frecuencia')
ax0.set_ylabel('Normalized Amplitude')
ax0.set_ylim(0,1.1)

xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
ax0.set_xticks(indices)
ax0.set_xticklabels(xticks_labels)
ax0.grid()
ax0.set_xlim(-0.4,4.5)

ax1.stem(indices - ancho_sep/2, fase_1, label='-20ºC',basefmt=' ' )
ax1.stem(indices + ancho_sep/2, fase_2, label='20 ºC',linefmt='C3-', markerfmt='D', basefmt=' ' )
#ax1.text(3/4,1/2,f'{f0:.0f} kHz',transform=ax1.transAxes,fontsize=14,bbox=dict(alpha=0.8))
ax1.legend(ncol=2)
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('Amplitude')
# ax1.set_ylim(0,3.5)

xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
ax1.set_xticks(indices)
ax1.set_xticklabels(xticks_labels)
ax1.grid()
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Phase (rad)')
ax1.set_xlim(-0.4,4.5)
ax1.set_ylim(0,15)
#plt.savefig('espectro_paper.png',facecolor='w',dpi=400)
#%% Ciclos
_,_, _,H_1,M_1,meta_1=lector_ciclos(os.path.join('resultados_2025','135kHz_100dA_100Mss_NEdd002_ciclo_H_M.txt'))

_,_, _,H_2,M_2,meta_2=lector_ciclos(os.path.join('resultados_2025','135kHz_100dA_100Mss_NEdd094_ciclo_H_M.txt'))
H_1/=1e3
H_2/=1e3

#%%
fig,ax = plt.subplots(figsize=(7,6),constrained_layout=True)
ax.plot(H_1,M_1,'C0',lw=1.5,label='-20 ºC')
ax.plot(H_2,M_2,'C3',lw=1.5,label='20 ºC')

ax.legend(ncol=2)
ax.set_xlabel('H (kA/m)')
ax.set_ylabel('M (A/m)')
ax.grid()
#plt.savefig('ciclos_paper.png',facecolor='w',dpi=400)

# %% Amplitud
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
fig = plt.figure(figsize=(12,7),constrained_layout=True)

# Usar gridspec para definir la disposición
gs = fig.add_gridspec(4,3, width_ratios=[1.1,1,1], height_ratios=[1, 1, 1,1])

ax0 = fig.add_subplot(gs[:2, 0])
ax1 = fig.add_subplot(gs[0, 1:3], sharex=ax2)
ax2 = fig.add_subplot(gs[1, 1:3])  # Segunda fila, segunda 

ax0.plot(H_1,M_1,'C0',lw=1.5,label='-20 ºC')
ax0.plot(H_2,M_2,'C3',lw=1.5,label='20 ºC')

ax0.legend(ncol=2)
ax0.set_xlabel('H (kA/m)')
ax0.set_ylabel('M (A/m)')
ax0.grid()

n = len(frecuencia_1)  # Número de frecuencias
indices = np.arange(n)  # Índices para las barras
ancho_sep = 0.1
f0 = frecuencia_1[0]/1000

ax1.stem(indices - ancho_sep/2, amplitud_1/max(amplitud_1), label='-20ºC',basefmt=' ' )
ax1.stem(indices + ancho_sep/2, amplitud_2/max(amplitud_2), label='20ºC',linefmt='C3-', markerfmt='D', basefmt=' ' )
#ax1.text(1/2,1/2,f'$f_0 =$ {f0:.0f} kHz\n$H_0 = 38.4$ kA/m',transform=ax1.transAxes,fontsize=14,ha='center',bbox=dict(alpha=0.8))
ax1.legend(ncol=2)
# ax1.set_xlabel('Frecuencia')
ax1.set_ylabel('Normalized Amplitude')
ax1.set_ylim(0,1.1)

# xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
# ax1.set_xticks(indices)
# ax1.set_xticklabels(xticks_labels)
ax1.grid()
ax1.set_xlim(-0.4,4.5)

ax2.stem(indices - ancho_sep/2, fase_1, label='-20ºC',basefmt=' ' )
ax2.stem(indices + ancho_sep/2, fase_2, label='20 ºC',linefmt='C3-', markerfmt='D', basefmt=' ' )
#ax2.text(3/4,1/2,f'{f0:.0f} kHz',transform=ax2.transAxes,fontsize=14,bbox=dict(alpha=0.8))
ax2.legend(ncol=2)
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Amplitude')
# ax2.set_ylim(0,3.5)

xticks_labels = [f'${2*i+1}\\cdot f_0$' if (2*i+1) != 1 else '$f_0$' for i in range(len(frecuencia_1))]
ax2.set_xticks(indices)
ax2.set_xticklabels(xticks_labels)
ax2.grid()
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Phase (rad)')
ax2.set_xlim(-0.4,4.5)
ax2.set_ylim(0,15)
#lt.savefig('figura_paper.png',facecolor='w',dpi=500)
# Mostrar la figura
plt.show()
#%% 

import numpy as np
import pandas as pd

def leer_archivo_histeresis(nombre_archivo):
    """
    Lee un archivo CSV con datos de histéresis y devuelve arrays NumPy con los datos.
    
    Parámetros:
    nombre_archivo (str): Ruta del archivo CSV a leer
    
    Retorna:
    tuple: (tiempo, campo, magnetizacion_seno, magnetizacion_ecdif)
           Todos como arrays de NumPy
    """
    try:
        # Leer el archivo CSV usando pandas
        df = pd.read_csv(nombre_archivo)
        
        # Extraer cada columna y convertir a arrays NumPy
        tiempo = df['tiempo_s'].to_numpy()
        campo_fit_Apm = df['campo_fit_Apm'].to_numpy()
        magnetizacion_fit_Apm_seno = df['magnetizacion_fit_Apm_seno'].to_numpy()
        magnetizacion_fit_Apm_ecdif = df['magnetizacion_fit_Apm_ecdif'].to_numpy()
        
        return tiempo, campo_fit_Apm, magnetizacion_fit_Apm_seno, magnetizacion_fit_Apm_ecdif
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {nombre_archivo}")
        return None, None, None, None
    except Exception as e:
        print(f"Error al leer el archivo: {str(e)}")
        return None, None, None, None

# Ejemplo de uso:
tiempo, campo_fit, magnetizacion_fit_Apm_seno, magnetizacion_fit_Apm_ecdif = leer_archivo_histeresis('135kHz_100dA_100Mss_NEdd094_ciclo_H_M_ajustes.csv')

t,_,_,H_kAm,M_Am,metadata = lector_ciclos( '135kHz_100dA_100Mss_NEdd094_ciclo_H_M.txt')
#%%

fig,(ax1,ax2)= plt.subplots(ncols=2,figsize=(13,13/3),gridspec_kw={'width_ratios': [6,4]},constrained_layout=True)
ax1.plot(t*1e6, H_kAm/1000,'o-',label='Field')
ax1.plot(tiempo*1e6, campo_fit/1000,label='Field fit')
ax1.plot(t*1e6,M_Am/10,'o-',label='Magnetization x100')
ax1.plot(tiempo*1e6,magnetizacion_fit_Apm_ecdif/10,label='Magnetization fit x100')

ax2.plot(H_kAm/1000,M_Am/1000,'go-',label='Data')
ax2.plot(campo_fit/1000,magnetizacion_fit_Apm_ecdif/1000,'r-',label='Fit')

ax1.set_ylabel('Field (kA/m) & Magnetization x100 (kA/m)')
ax1.set_xlabel('Time ($\mu$s)')
ax2.legend()
ax1.legend(title='$R^2$: Field 1.00000, Magnetization 0.98625',loc='lower center')
for a in [ax1,ax2]:
    a.grid()
    
ax2.set_ylabel('Magnetization (kA/m)')
ax2.set_xlabel('Field (kA/m)')
plt.savefig('data_vs_fit_t_H.png',dpi=400,facecolor='w')
