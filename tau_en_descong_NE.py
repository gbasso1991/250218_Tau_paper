#%%
"""
Tau vs T para distintas concentraciones
Giuliano Basso
"""
import os
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chardet
import re
from scipy.optimize import curve_fit
from glob import glob
#%% Funcion de lectura
def lector_resultados(path):
    '''return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N'''
    
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(6):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=14,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.to_datetime(data['Time_m'][:],dayfirst=True)
    # delta_t = np.array([dt.total_seconds() for dt in (time-time[0])])
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
     
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    
    
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N


#%% Diluida al 25 %
f_idc=['135kHz','100dA']
indentif_dd = '_NEdd_'

files_NE_dd=[f for f in os.listdir(os.path.join(os.getcwd(),'resultados_2025','tablas')) if indentif_dd in f]
    
files_NE_dd.sort()
paths_NE_dd = [os.path.join(os.getcwd(),'resultados_2025','tablas',f) for f in files_NE_dd]

# Obtener la lista de archivos en el directorio seleccionado
for i, e in enumerate(files_NE_dd):
    print(i, e.split('/')[-1])

H_max_NE_dd = []
M_max_NE_dd = []
M_max_NE_dd_err = []
taus_NE_dd = []
SARs_NE_dd = []
Hcs_NE_dd = []
Mrs_NE_dd = []
xi_M_0_s_NE_dd = []
Concentracion_NE_dd = [] 
Temps_NE_dd = []
times_NE_dd=[]

# Iterar sobre cada archivo seleccionado
for file in paths_NE_dd:
    # Leer los resultados del archivo actual
    meta, _, time, Temp, Mr, Hc, Hmax, M_max, xi_M_0, _, _, _, SAR, tau, _ = lector_resultados(file)
    # Append en lista de resultados
    H_max_NE_dd.append(Hmax.mean() / 1e3)
    M_max_NE_dd.append(M_max.mean())
    M_max_NE_dd_err.append(M_max.std())
    taus_NE_dd.append(tau)  
    SARs_NE_dd.append(SAR)
    Hcs_NE_dd.append(Hc)
    Mrs_NE_dd.append(Mr)
    xi_M_0_s_NE_dd.append(xi_M_0)
    Concentracion_NE_dd.append(meta['Concentracion g/m^3'] / 1e6) # g/cm³ 
    Temps_NE_dd.append(Temp)
    times_NE_dd.append(time)
tau_mean_NE_dd = [t.mean() for t in taus_NE_dd]
tau_err_NE_dd = [t.std() for t in taus_NE_dd]
SAR_mean_NE_dd = [s.mean() for s in SARs_NE_dd]
SAR_err_NE_dd = [s.std() for s in SARs_NE_dd]
Hc_mean_NE_dd = [s.mean() for s in Hcs_NE_dd]
Hc_err_NE_dd = [s.std() for s in Hcs_NE_dd]
Mr_mean_NE_dd = [s.mean() for s in Mrs_NE_dd]
Mr_err_NE_dd = [s.std() for s in Mrs_NE_dd]
xi_M_0_mean_NE_dd = [s.mean() for s in xi_M_0_s_NE_dd]
xi_M_0_err_NE_dd = [s.std() for s in xi_M_0_s_NE_dd]

# Normalizo por magnetizacioin maxima
Mr_mean_norm_NE_dd = np.array(Mr_mean_NE_dd)/np.array(M_max_NE_dd) 
Mr_err_norm_NE_dd = np.array(Mr_err_NE_dd)/np.array(M_max_NE_dd)

# Normalizo por concentracion y paso Magnetizacion a emu/g (10³ A/m == 1 emu/cm³ )
Concentracion_NE_dd = np.array(Concentracion_NE_dd).mean()
M_max_NE_dd = np.array(M_max_NE_dd)*(1e-3/Concentracion_NE_dd)
M_max_NE_dd_err = np.array(M_max_NE_dd_err)*(1e-3/np.array(Concentracion_NE_dd).mean())
xi_M_0_mean_NE_dd = np.array(xi_M_0_mean_NE_dd)/Concentracion_NE_dd
xi_M_0_err_NE_dd = np.array(xi_M_0_err_NE_dd)/Concentracion_NE_dd
# TEMPLOGS y gradiente 
#hay fechas repetidas, me arruinan el calculo de derivada 

indices_duplicados_1 = times_NE_dd[0][times_NE_dd[0].duplicated()].index
indices_duplicados_2 = times_NE_dd[1][times_NE_dd[1].duplicated()].index
indices_duplicados_3 = times_NE_dd[2][times_NE_dd[2].duplicated()].index

#elimino e indexo
Temps_NE_dd_1 = np.delete(Temps_NE_dd[0], indices_duplicados_1)
Temps_NE_dd_3 = np.delete(Temps_NE_dd[2], indices_duplicados_3)
Temps_NE_dd_2 = np.delete(Temps_NE_dd[1], indices_duplicados_2)

times_NE_dd_1 = times_NE_dd[0].drop_duplicates()
times_NE_dd_2 = times_NE_dd[1].drop_duplicates()
times_NE_dd_3 = times_NE_dd[2].drop_duplicates()

dt1=(times_NE_dd_1-times_NE_dd_1[0]).dt.total_seconds().to_numpy()
dt2=(times_NE_dd_2-times_NE_dd_2[0]).dt.total_seconds().to_numpy()
dt3=(times_NE_dd_3-times_NE_dd_3[0]).dt.total_seconds().to_numpy()

indices_20_1 = np.nonzero(Temps_NE_dd_1 >= 10.0)[0][0]
indices_20_2 = np.nonzero(Temps_NE_dd_2 >= 10.0)[0][0]
indices_20_3 = np.nonzero(Temps_NE_dd_3 >= 10.0)[0][0]
tiempos_sync_1 = dt1-dt1[indices_20_1]
tiempos_sync_2 = dt2-dt2[indices_20_2]
tiempos_sync_3 = dt3- dt3[indices_20_3]

dT1 = np.gradient(Temps_NE_dd_1,dt1)
dT2 = np.gradient(Temps_NE_dd_2,dt2)
dT3 = np.gradient(Temps_NE_dd_3,dt3)
indx_max_1 = np.nonzero(dT1==max(dT1)) 
indx_max_2 = np.nonzero(dT2==max(dT2))
indx_max_3 = np.nonzero(dT3==max(dT3))

#ploteo Tau y SAR
fig,(ax,ax1,ax2) = plt.subplots(3, 1, figsize=(7, 9), constrained_layout=True,sharex=False)
ax.plot(Temps_NE_dd[0],taus_NE_dd[0],'o-',label='1')
ax.plot(Temps_NE_dd[1],taus_NE_dd[1],'o-',label='2')
ax.plot(Temps_NE_dd[2],taus_NE_dd[2],'o-',label='3')
ax.axvspan(-10,0,facecolor='blue', alpha=0.3)
ax.grid()
ax.legend()
ax.set_ylabel('$\\tau$ (s)')

ax1.plot(Temps_NE_dd[0],SARs_NE_dd[0],'o-',label='1')
ax1.plot(Temps_NE_dd[1],SARs_NE_dd[1],'o-',label='2')
ax1.plot(Temps_NE_dd[2],SARs_NE_dd[2],'o-',label='3')
ax1.axvspan(-10,0,facecolor='blue', alpha=0.3)
ax1.grid()
ax1.legend()
ax1.set_xlabel('T (ºC)')
ax1.set_ylabel('SAR (W/g)')

# 5 Templog 
ax2.plot(tiempos_sync_1,Temps_NE_dd_1,'.-')
ax2.plot(tiempos_sync_2,Temps_NE_dd_2,'.-')
ax2.plot(tiempos_sync_3,Temps_NE_dd_3,'.-')

ax2.scatter(tiempos_sync_1[indx_max_1],Temps_NE_dd_1[indx_max_1],marker='D',zorder=2,color='blue',label=f'dT/dt = {max(dT1):.2f} ºC/s')
ax2.scatter(tiempos_sync_2[indx_max_2],Temps_NE_dd_2[indx_max_2],marker='D',zorder=2,color='orange',label=f'dT/dt = {max(dT2):.2f} ºC/s')
ax2.scatter(tiempos_sync_3[indx_max_3],Temps_NE_dd_3[indx_max_3],marker='D',zorder=2,color='green',label=f'dT/dt = {max(dT3):.2f} ºC/s')

ax2.legend(loc='upper left')
ax2.axhspan(-10,0,color='blue',alpha=0.3)
ax2.grid()
ax2.set_xlim(min((min(tiempos_sync_1),min(tiempos_sync_2),min(tiempos_sync_3))))
ax2.set_xlabel('t (s)')
ax2.set_ylabel('T (ºC)')

axin = ax2.inset_axes([0.5, 0.18, 0.48, 0.5])
axin.plot(tiempos_sync_1,dT1,'.-',label='1')
axin.plot(tiempos_sync_2,dT2,'.-',label='2')
axin.plot(tiempos_sync_3,dT3,'.-',label='3')
axin.set_xlabel('t (s)')
axin.set_ylabel('dT/dt (ºC/s)')
axin.legend(ncol=3)
axin.grid()
plt.suptitle(f'{f_idc[0]} - {f_idc[1]} - g/L')
#plt.savefig('tau_SAR_NE_dd.png', dpi=300, facecolor='w')

#%% COMPARATIVA TAU y SAR 
### TAU
fig, axs = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
# Gráfico 1

axs[0].plot(Temps_NE_dd[0], taus_NE_dd[0],'v-',label=f'{Concentracion_NE_dd*1e3} g/L')
axs[0].plot(Temps_NE_dd[1], taus_NE_dd[1],'v-',label=f'{Concentracion_NE_dd*1e3} g/L')
axs[0].plot(Temps_NE_dd[2], taus_NE_dd[2],'v-',label=f'{Concentracion_NE_dd*1e3} g/L')

axs[0].axvspan(-10,0,facecolor='blue', alpha=0.3)
axs[0].set_ylabel(r'$\tau$ (s)')
axs[0].set_title(r'$\tau$ (s)')
axs[0].legend(ncol=3)
axs[0].grid()

# Gráfico 2

axs[1].plot(Temps_NE_dd[0], SARs_NE_dd[0],'v-',label=f'{Concentracion_NE_dd*1e3} g/L')
axs[1].plot(Temps_NE_dd[1], SARs_NE_dd[1],'v-',label=f'{Concentracion_NE_dd*1e3} g/L')
axs[1].plot(Temps_NE_dd[2], SARs_NE_dd[2],'v-',label=f'{Concentracion_NE_dd*1e3} g/L')

axs[1].axvspan(-10,0,facecolor='blue', alpha=0.3)
axs[1].set_xlabel('T (ºC)')
axs[1].set_ylabel('SAR (W/g)')
axs[1].set_title('SAR (W/g)')
axs[1].legend(ncol=3)
axs[1].grid()

plt.suptitle('NE@citrato - 135 kHz - 38 kA/m\nComparativa por concentración', fontsize=18)
#plt.savefig('tau_SAR_NE_135_10_comparativa_por_concentracion.png', dpi=300, facecolor='w')
plt.show()

#%% Promedio los tau por temperatura

temperatura1 = Temps_NE_dd[0] 
tau1 = taus_NE_dd[0]
SAR1= SARs_NE_dd[0]
temperatura2 = Temps_NE_dd[1] 
tau2 = taus_NE_dd[1]
SAR2 = SARs_NE_dd[1]
temperatura3 = Temps_NE_dd[2] 
tau3 = taus_NE_dd[2]
SAR3 = SARs_NE_dd[2]

#recorto a -20 - 20 °C
indx_1 = np.nonzero((temperatura1>=-20)&(temperatura1<=20))
indx_2 = np.nonzero((temperatura2>=-20)&(temperatura2<=20))
indx_3 = np.nonzero((temperatura3>=-20)&(temperatura3<=20))

temperatura1=temperatura1[indx_1] 
tau1=tau1[indx_1] 
SAR1=SAR1[indx_1] 

temperatura2=temperatura2[indx_2] 
tau2=tau2[indx_2]
SAR2=SAR2[indx_2]

temperatura3=temperatura3[indx_3] 
tau3=tau3[indx_3]
SAR3=SAR3[indx_3]

# Concatenamos todas las temperaturas y taus
temperatura_total = np.concatenate((temperatura1, temperatura2, temperatura3))
tau_total = np.concatenate((tau1, tau2, tau3))
SAR_total = np.concatenate((SAR1, SAR2, SAR3))

# Definimos los intervalos de temperatura
intervalo_temperatura = 1
temperaturas_intervalo = np.arange(np.min(temperatura_total), np.max(temperatura_total) + intervalo_temperatura, intervalo_temperatura)
intervalo_SAR = 1
SAR_intervalo = np.arange(np.min(SAR_total), np.max(SAR_total) + intervalo_SAR, intervalo_SAR)

# Lista para almacenar los promedios de tau
promedios_tau = []
errores_tau =[]
promedios_SAR = []
errores_SAR =[]
# Iteramos sobre los intervalos de temperatura
for temp in temperaturas_intervalo:
    # Seleccionamos los valores de tau correspondientes al intervalo de temperatura actual
    tau_intervalo = tau_total[(temperatura_total >= temp) & (temperatura_total < temp + intervalo_temperatura)]
    SAR_intervalo = SAR_total[(temperatura_total >= temp) & (temperatura_total < temp + intervalo_temperatura)]
    
    # Calculamos el promedio y lo agregamos a la lista
    promedios_tau.append(np.mean(tau_intervalo))
    errores_tau.append(np.std(tau_intervalo))
    promedios_SAR.append(np.mean(SAR_intervalo))
    errores_SAR.append(np.std(SAR_intervalo))
# Convertimos la lista de promedios a un array de numpy
promedios_tau = np.array(promedios_tau)
err_temperatura=np.full(len(temperaturas_intervalo),intervalo_temperatura/2)
promedios_SAR = np.array(promedios_SAR)

print("Intervalo de Temperatura   |   Promedio de Tau  |   Promedio SAR   |")
print("--------------------------------------------------------------------")
for i in range(len(temperaturas_intervalo)):
    print(f"{temperaturas_intervalo[i]:.2f} - {temperaturas_intervalo[i] + intervalo_temperatura:.2f} °C   |   {promedios_tau[i]:.2e}")

#indices = [0,32,42,92] 

#%% PLOT TAU
import matplotlib as mpl
fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
ax.errorbar(x=temperaturas_intervalo,y=promedios_tau,xerr=err_temperatura,yerr=errores_tau,capsize=4,fmt='.-')
plt.grid()
plt.xlabel('Temperature (°C)')
plt.ylabel(r'$\tau$ (s)')
plt.xlim(-21,21)
# plt.title(f'135 kHz - 38 kA/m - {Concentracion_NE_dd*1e3:.2f} g/L')
#plt.savefig('tau_135_38_ddiluido.png',dpi=400)
plt.show()

fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
ax.errorbar(x=temperaturas_intervalo,y=promedios_SAR,xerr=err_temperatura,yerr=errores_SAR,capsize=4,fmt='.-')
plt.grid()
plt.xlabel('Temperature (°C)')
plt.ylabel('SAR (W/g)')
plt.xlim(-21,21)
# plt.title(f'135 kHz - 38 kA/m - {Concentracion_NE_dd*1e3:.2f} g/L')
#plt.savefig('tau_135_38_ddiluido.png',dpi=400)
plt.show()
#%% PLOT TAU COLORES
cmap = mpl.cm.get_cmap('jet')
normalized_temperaturas = (temperaturas_intervalo - temperaturas_intervalo.min()) / (temperaturas_intervalo.max() - temperaturas_intervalo.min())
colors = cmap(normalized_temperaturas)

temp_aux = np.linspace(-20,20,1000)
normalized_temperaturas_2 = (temp_aux - temp_aux.min()) / (temp_aux.max() - temp_aux.min())
colors_2 = cmap(normalized_temperaturas_2)

fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)

for i,e in enumerate(temperaturas_intervalo):
    ax.errorbar(x=e, xerr=err_temperatura[i], y=promedios_tau[i]*1e9, yerr=errores_tau[i]*1e9,capsize=4,ecolor=colors[i])

ax.scatter(temperaturas_intervalo[0], y=promedios_tau[0]*1e9,marker='o',color=colors[0],zorder=3)
ax.scatter(temperaturas_intervalo[-2], y=promedios_tau[-2]*1e9,marker='o',color=colors[-2],zorder=3)

ax.scatter(temperaturas_intervalo, y=promedios_tau*1e9,color=colors,marker='.',zorder=2)
ax.plot(temperaturas_intervalo,promedios_tau*1e9,'k',zorder=-2,alpha=0.6)
# ax.yaxis.set_label_position("right")

ax.text(0.75,0.3,'C = 7,4 g/L',bbox=dict(alpha=0.9),ha='center',va='center',transform=ax.transAxes)
plt.grid()
plt.xlabel('Temperatura (°C)')
plt.ylabel(r'$\tau$ (ns)')
plt.xlim(-21, 21)
# plt.title(f'135 kHz - 38 kA/m - {Concentracion_NE_dd*1e3:.2f} g/L')
plt.savefig('tau_135_38_7.4gL.png', dpi=400)

#%%

temperaturas_intervalo=temperaturas_intervalo[:-1]
err_temperatura=err_temperatura[:-1]
promedios_tau=promedios_tau[:-1]
errores_tau=errores_tau[:-1]
combined_array = np.vstack((temperaturas_intervalo, err_temperatura,promedios_tau,errores_tau)).T
np.savetxt('tau_vs_T_NE@citrato_7,4gL.txt', combined_array,header='| T | err T | tau | err tau |' ,fmt=['%f','%f','%e','%e'])












#%% Ploteo ciclos frios
def lector_ciclos(path):
    '''return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N'''
    
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i in range(6):
            line = f.readline().strip()
            if i == 0:
                match = re.search(r'Temperatura_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Temperatura'
                    value = float(match.group(1))
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1).replace(' ', '_')
                    value = float(match.group(2))
                    meta[key] = value
    # Leer los datos del archivo
    data = pd.read_table(path, header=6,
                         names=('Tiempo_(s)','Campo_(V.s)','Magnetizacion_(V.s)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                         usecols=(0, 1, 2, 3, 4),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    t = pd.Series(data['Tiempo_(s)']).to_numpy(dtype=float)
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)
    M_Am = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)
        
    return meta,t, H_kAm , M_Am






#%% 2025 agrego taus de pedro
data1 = np.genfromtxt('reporte_tau_240314_120348_10.txt', delimiter=',', skip_header=1)
temp_1 = data1[:-1, 0]
tau_1 = data1[:-1, 1]
tau_1_stderr = data1[:-1, 2]

data2 = np.genfromtxt('reporte_tau_240314_121301_10.txt', delimiter=',', skip_header=1)
temp_2 = data2[:-1, 0]
tau_2 = data2[:-1, 1]
tau_2_stderr = data2[:-1, 2]

data3 = np.genfromtxt('reporte_tau_240314_122338_135_10.txt', delimiter=',', skip_header=1)
temp_3 = data3[:-1, 0]
tau_3 = data3[:-1, 1]
tau_3_stderr = data3[:-1, 2]

#descarto el ultimo punto 
#%%Promedio los tau por temperatura

temperatura1 = temp_1 
tau1 = tau_1
temperatura2 = temp_2 
tau2 = tau_2
temperatura3 = temp_3 
tau3 = tau_3

#recorto a -20 - 20 °C
indx_1 = np.nonzero((temperatura1>=-20)&(temperatura1<=20))
indx_2 = np.nonzero((temperatura2>=-20)&(temperatura2<=20))
indx_3 = np.nonzero((temperatura3>=-20)&(temperatura3<=20))

temperatura1=temperatura1[indx_1] 
tau1=tau1[indx_1] 

temperatura2=temperatura2[indx_2] 
tau2=tau2[indx_2]

temperatura3=temperatura3[indx_3] 
tau3=tau3[indx_3]

# Concatenamos todas las temperaturas y taus
temperatura_total = np.concatenate((temperatura1, temperatura2, temperatura3))
tau_total = np.concatenate((tau1, tau2, tau3))

# Definimos los intervalos de temperatura
intervalo_temperatura = 1
temperaturas_intervalo_pedro = np.arange(np.min(temperatura_total), np.max(temperatura_total) + intervalo_temperatura, intervalo_temperatura)

# Lista para almacenar los promedios de tau
promedios_tau_pedro = []
errores_tau_pedro =[]
# Iteramos sobre los intervalos de temperatura
for temp in temperaturas_intervalo_pedro:
    # Seleccionamos los valores de tau correspondientes al intervalo de temperatura actual
    tau_intervalo = tau_total[(temperatura_total >= temp) & (temperatura_total < temp + intervalo_temperatura)]
    # Calculamos el promedio y lo agregamos a la lista
    promedios_tau_pedro.append(np.mean(tau_intervalo))
    errores_tau_pedro.append(np.std(tau_intervalo))
# Convertimos la lista de promedios a un array de numpy
promedios_tau_pedro = np.array(promedios_tau_pedro)
err_temperatura_pedro=np.full(len(temperaturas_intervalo_pedro),intervalo_temperatura/2)
#%%
print("Intervalo de Temperatura   |   Promedio de Tau  |")
print("-------------------------------------------------")
for i in range(len(temperaturas_intervalo_pedro)):
    print(f"{temperaturas_intervalo_pedro[i]:.2f} - {temperaturas_intervalo_pedro[i] + intervalo_temperatura:.2f} °C |   {promedios_tau_pedro[i]:.2e}")

#%%
promedios_tau=np.array(promedios_tau)*1e9
errores_tau=np.array(errores_tau)*1e9

#%%

label_1='$\\tan (\phi_1) /\omega$'
label_2='$M(t)$ fitting'
fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
ax.errorbar(x=temperaturas_intervalo,y=promedios_tau,xerr=err_temperatura,yerr=errores_tau,capsize=4,
fmt='.-',color='C1',label=label_1)

ax.errorbar(x=temperaturas_intervalo_pedro,y=promedios_tau_pedro,xerr=err_temperatura_pedro,yerr=errores_tau_pedro,
capsize=4,fmt='.-',color='C2',label=label_2)


#ax.errorbar(x=temperatura,y=tau_ed,yerr=tau_ed_stderr,capsize=4,fmt='.-')

# ax.errorbar(x=temp_1,y=tau_1,yerr=tau_1_stderr,capsize=4,fmt='.-')
# ax.errorbar(x=temp_2,y=tau_2,yerr=tau_2_stderr,capsize=4,fmt='.-')
# ax.errorbar(x=temp_3,y=tau_3,yerr=tau_3_stderr,capsize=4,fmt='.-')

plt.legend(ncol=1,fontsize=13)
plt.grid()
plt.xlabel('Temperature (°C)')
plt.ylabel(r'$\tau$ (s)')
plt.xlim(-21,21)
# plt.title(f'135 kHz - 38 kA/m - {Concentracion_NE_dd*1e3:.2f} g/L')
plt.savefig('tau_135_38_ddiluid_vs_tau_Pedro.png',facecolor='w',dpi=400)
plt.show()

# %%
