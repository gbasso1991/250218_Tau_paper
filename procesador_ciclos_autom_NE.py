#%%
'''
procesador_ciclos_OWON.py

https://youtu.be/2VBqV8RN9RA

Giuliano Andrés Basso

basado en procesador_ciclos.py, adaptado para leer los archivos obtenidos con el
osciloscopio OWON controlandolo digitalmente

IMPORTANTE: requiere 'funciones_del_procesador.py' en el mismo directorio 

0 -  Lee archivos en el directorio seleccionado
    usando identificador 'nombre'(FF, FR, cit05, etc.)
        muestra: xxxkHz_yyydA_zzzMss_nombre.txt 

    Identfica archivos con 'fondo...' en su nombre, y toma el 1ero 
    (_fondo00.txt) como fondo a restar en todas las muestras.
    
        fondo:  xxxkHz_yyydA_zzzMss_fondo*.txt 

1 - En loop:
    
    Caracteriza el campo H aplicado. 
    Corta en caso de incopatibilidad de frecuencias
    Resta fondo interpolado a las señales de muestra
    Fourier:
        Toma señal recortada a N ciclos (importante)
        Copia los vectores
        rfft sobre Resta_m_3
        Filtra a N armonicos especificados
        reconstruye e integra (int(muestra delta_t) = M )

    grafica H y M normalizados


Grafica todos los ciclos 
Grafica todos los ciclos filtrados
Exporta tablas:
    ciclo    >   | t | H | M | 
    espectro  >  |'Frecuencia_Hz'|'Amplitud'|'Fase'|
    resultados > |fname|time|Temp|Mr|Hc|Hmax|f0|mag0|phi0|dphi0|SAR|Tau|
'''
import time
import os
import fnmatch
from datetime import datetime,timedelta
from numpy.core.numeric import indices 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cmx  
import matplotlib as mpl
import pandas as pd
from uncertainties import ufloat, unumpy
import tkinter as tk
from tkinter import filedialog
import scipy as sc
from scipy.signal import find_peaks 
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.fft import fft, ifft, rfftfreq,irfft 
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from sklearn.metrics import r2_score
from pprint import pprint
from scipy.optimize import curve_fit
from funciones_procesado import medida_cruda, medida_cruda_autom,ajusta_seno, sinusoide,resta_inter,filtrando_ruido,recorte,promediado_ciclos,fourier_señales_5,lector_templog_2,lector_templog,susceptibilidad_M_0

#%% Configuracion de Script
start_time = time.time() 
todos=1
un_solo_fondo=1
resto_fondo=1
templog = 1
nombre='*NE*'
Analisis_de_Fourier = 1 # sobre las señales, imprime espectro de señal muestra
N_armonicos_impares = 10
concentracion =(29.6/4)*1e3 #[concentracion]= g/m^3 (1 g/l == 1e3 g/m^3) (Default = 10000 g/m^3)
capsula_glucosa=0   # capsula para solventes organicos
#¿Qué gráficos desea ver? (1 = sí, ~1 = no)
graficos={
        'Referencias_y_ajustes': 0,
        'Resta_m-f': 0,
        'Resta_c-f': 0, 
        'Resta_mf_y_cf':0,
        'Filtrado_calibracion': 0,
         
        'Filtrado_muestra': 0,
        'Recorte_a_periodos_enteros_c': 0,
        'Recorte_a_periodos_enteros_m': 0,
        'Campo_y_Mag_norm_c': 0,
        'Ciclos_HM_calibracion': 0,
        'Campo_y_Mag_norm_m': 0,
        'Ciclo_HM_m': 0 ,
        'Susceptibilidad_M_0':0,
        'Ciclos_HM_m_todos': 1}
mu_0 = 4*np.pi*10**-7 #[mu_0]=N/A^2
# =============================================================================
#Calibracion del campo en la bobina: cte que dimensionaliza al campo en A/m a partir de la calibracion
#realizada sobre la bobina  del RF
# Valores actualizados en mar 2023 con medidas hechas por Giuliano en 2022 y las calibraciones de sonda Hall
pendiente_HvsI = 3716.3 # 1/m
ordenada_HvsI = 1297.0 # A/m
# =============================================================================
#Calibracion de la magnetizacion: cte que dimensionaliza a M en Vs --> A/m
xi_patron_Dy2O3_v = 5.314e-3 #adimensional. Valor de VSM sobre capsula  
pendiente_patron_Dy2O3 = ufloat(2.089470553242323e-14,2.419522898110445e-15) #Vsm/A
C_Vs_to_Am_magnetizacion = xi_patron_Dy2O3_v/pendiente_patron_Dy2O3.nominal_value #A/mVs
if capsula_glucosa ==1:
    C_Vs_to_Am_magnetizacion = C_Vs_to_Am_magnetizacion*0.506
# =============================================================================
# Imprimo configuraciones
print(time.strftime('%Y %m %d', time.localtime()),'-'*50)
print('Configuracion del script:')
print(f'''
      Dir de trabajo: {os.getcwd()}
      Selecciono todos los archivos del directorio: {bool(todos)}
      Resto señal de fondo: {bool(resto_fondo)}
      Identificador de archivos de muestra': {nombre}
      Filtrado en armonicos impares: {bool(Analisis_de_Fourier)}
      Num de armonicos impares considerados: {N_armonicos_impares}
      Templog: {bool(templog)}
      Concentracion: {concentracion/1000} g/l
      Capsula glucosa: {bool(capsula_glucosa)}
      ''','\n','-'*60)

#%Defino listas para almacenar datos en cada iteracion
long_arrays=[]
Ciclos_eje_H = []
Ciclos_eje_H_sf = []
Ciclos_eje_M = []
Ciclos_eje_M_filt =[]
Ciclos_eje_M_filt_norm=[]
Frecuencia_ref_muestra_kHz = []
Frecuencia_ref_fondo_kHz = []
Frec_fund=[]
SAR = []
SAR_filt=[]

Campo_maximo = []
Mag_max=[]
Coercitividad_kAm = []
Remanencia_Am = []

Tau=[]
Frec_fem = []
Defasaje_1er_arm = []
Magnitud_1er_arm = []
xi_M_0=[]

cociente_f1_f0 = []
cociente_f2_f0 = []

#Fecha para usar en graficos 
fecha_nombre = datetime.today().strftime('%Y%m%d_%H%M%S')
fecha_graf = time.strftime('%Y_%m_%d', time.localtime())


#% Seleccion de carpeta con archivos via interfaz de usuario
root = tk.Tk()
root.withdraw()
if todos==1: #Leo todos los archivos del directorio
    texto_encabezado = "Seleccionar la carpeta con las medidas a analizar:"
    directorio = filedialog.askdirectory(title=texto_encabezado)
    filenames = [f for f in os.listdir(directorio) if f.endswith('.txt')] #todos
    filenames.sort()
    
    fnames_m = []
    path_m = []
    fnames_f = []
    path_f = []
    print('Directorio de trabajo: \n'+ directorio +'\n')
    print('Archivos de muestra en el directorio:\n') 
    for idx_m, m in enumerate(fnmatch.filter(filenames,nombre)):
        print(idx_m,'-',m)
        fnames_m.append(m)
        path_m.append(os.path.join(directorio,m)) 
    
    print('.'*40)
    print('Archivos de fondo en el directorio:\n')

    for idx_f,f in enumerate(fnmatch.filter(filenames, '*_fondo*')):
        print(idx_f,'-',f)
        fnames_f.append(f)
        path_f.append(os.path.join(directorio,f)) 
        
    print(f'\nSon {len(fnames_m)}/{len(fnames_m)+len(fnames_f)} archvos de muestra, {len(fnames_f)}/{len(fnames_m)+len(fnames_f)} de fondo')
        
    print('-'*50)
else:
    print('''setear: 'todos=1' ''')    
    
if len(fnames_m)==0:
    raise Exception(f'No se seleccionaron archivos de muestra.\nIdentificador: {nombre}')

#Creo carpeta para guardar analisis
output_dir = os.path.join(directorio,f'Analisis_{fecha_nombre}') # Directorio donde se guardarán los resultados
if not os.path.exists(output_dir): # Crear el directorio si no existe
    os.makedirs(output_dir)

    
# Parametros a partir de nombre del archivo 
frec_nombre=[]      #Frec del nombre del archivo. Luego comparo con frec ajustada
Idc = []            #Internal direct current en el generador de RF
delta_t = []        #Base temporal 
fecha_m = []        #fecha de creacion archivo, i.e., de la medida 

for i in range(len(fnames_m)):
    frec_nombre.append(float(fnames_m[i].split('_')[0][:-3])*1000)
    Idc.append(float(fnames_m[i].split('_')[1][:-2])/10)
    delta_t.append(1e-6/float(fnames_m[i].split('_')[2][:-3]))
    fecha_m.append(datetime.fromtimestamp(os.path.getmtime(path_m[i])).strftime('%Y-%m-%d- %H:%M'))


#% Templog
if templog:
    try:
        timestamp,temperatura,T_amb =lector_templog_2(directorio) 
        fecha_m_aux=[datetime.fromtimestamp(os.path.getmtime(path)) for path in path_m] 
        delta_t_m = [(int(( t - fecha_m_aux[0]).total_seconds())) for t in fecha_m_aux] #dif en s para los archivos de muestra

        # Calcula las hs de diferencia entre los archivos de muestra y el timestamp 
        diferencia_horaria = (fecha_m_aux[0]-timestamp[0]).total_seconds()//3600
        print(f'{diferencia_horaria} hs de diferencia entre archivos y timestamp')
        date_primer_dato = fecha_m_aux[0]-timedelta(hours=diferencia_horaria) 
        time_m = [(date_primer_dato + timedelta(0,dt)) for dt in delta_t_m]
        temp_m= np.array([temperatura[np.flatnonzero(timestamp==t)[0]] for t in time_m])
        #temp_fondo= T_amb[np.flatnonzero(timestamp==time_m[-1])][0]
        
        cmap = mpl.colormaps['jet']  #'viridis' # Crear un rango de colores basado en las temperaturas y el cmap
        norm = plt.Normalize(temp_m.min(), temp_m.max())

        fig,ax=plt.subplots(figsize=(8,5.5),constrained_layout=True)
        ax.plot(np.arange(len(temperatura)),temperatura,'k-',lw=1,zorder=3)
        delta_0 = (time_m[0] - timestamp[0]).total_seconds()    

        ax.scatter(delta_t_m[0]+delta_0,temp_m[0],c='k')
        for i in range(0,len(temp_m)-1):
            color = cmap(norm(temp_m[i]))
            ax.scatter(delta_t_m[i]+delta_0,temp_m[i],color=color,label=fnames_m[i].split("_")[-1].split(".")[0])
        ax.scatter(delta_0+delta_t_m[-1],temp_m[-1],marker='x',c='k',label='fondo')
        #plt.plot(np.arange(len(T_amb)),T_amb, label= 'RT')
        #plt.legend(ncol=4)
        # plt.legend(loc='center',bbox_to_anchor=(0.5,-0.25),ncol=6,fancybox=True)
        plt.xlim(0,len(T_amb))
        plt.grid()
        plt.xlabel('$\Delta t$ (s) ')    
        plt.ylabel('Temperatura (°C) ') 
        plt.title('Temperatura de archivos de muestra',fontsize=20)
        plt.text(0.75,0.15,f'{frec_nombre[0]/1000:.0f} kHz',fontsize=15,bbox=dict(color='tab:orange',alpha=0.7),transform=ax.transAxes) 
        plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_templog.png'),dpi=200,facecolor='w')
    
    except IndexError:
        print('El horario de los archivos no se encuentra en el templog')
        print(f'''Inicio del templog: {timestamp[0]}
Fin del templog: {timestamp[-1]}
Datetime primer archivo {time_m[0]}''') 

else:
    print('No se requiere archivo con templog')
    temp_m=np.array([float(20) for f in fnames_m])
    cmap = mpl.colormaps['jet']  #'viridis' # Crear un rango de colores basado en las temperaturas y el cmap
    norm = plt.Normalize(temp_m.min(), temp_m.max())
#%% Procesamiento en iteracion sobre archivos 
''' 
Ejecuto medida_cruda()
En cada iteracion levanto la info de los .txt a dataframes.
''' 
k=0     #indice muestras
k_f=0   #indice fondos 
for k in range(len(fnames_m)):
    if un_solo_fondo==1:
        print('Fondo unico:',fnames_f[0],'(1ero en orden alfabetico') 
    else:
        print('1 fondo cada 3 archivos de muestra')
        if (k!=0) and (np.mod(k,3)==0):
            if k_f<len(fnames_f): 
                k_f+=1
            else:
                pass
    df_m = medida_cruda_autom(path_m[k],delta_t[k])# DataFrame
    print('-'*50)
    print(k,k_f)

    print('name',fnames_m[k][:-4],fnames_f[k_f][:-4])
    print('path',path_m[k][-28:-4],path_f[k_f][-32:-4])
    
    df_f = medida_cruda_autom(path_f[k_f],delta_t[k_f])

    '''
    Ajuste sobre señal de referencia (dH/dt) y obtengo params
    Ejecuto ajusta_seno()
    '''
    offset_m , amp_m, frec_m , fase_m = ajusta_seno(df_m['t'],df_m['v_r'])
    offset_f , amp_f, frec_f , fase_f = ajusta_seno(df_f['t'],df_f['v_r'])

    #Genero señal simulada usando params y guardo en respectivos df
    df_m['v_r_ajustada'] = sinusoide(df_m['t'],offset_m , amp_m, frec_m , fase_m)
    df_f['v_r_ajustada'] = sinusoide(df_f['t'],offset_f , amp_f, frec_f , fase_f)

    df_m['residuos'] = df_m['v_r'] - df_m['v_r_ajustada']
    df_f['residuos'] = df_f['v_r'] - df_f['v_r_ajustada']

    # Comparacion de señales y ajustes
    if graficos['Referencias_y_ajustes']==1:
        fig , ax = plt.subplots(2,1, figsize=(8,6),sharex='all')
        
        df_m.plot('t','v_r',label='Referencia',ax=ax[0],title=f'Muestra: {fnames_m[k]}')
        df_m.plot('t','v_r_ajustada',label='Ajuste',ax =ax[0])
        df_m.plot('t','residuos', label='Residuos',ax=ax[0])

        df_f.plot('t','v_r',label='Referencia de fondo',ax=ax[1])
        df_f.plot('t','v_r_ajustada',label='Ajuste',ax =ax[1],title=f'Fondo: {fnames_f[k_f]}')
        df_f.plot('t','residuos', label='Residuos',ax=ax[1])

        fig.suptitle('Comparacion señal de referencias y ajustes',fontsize=20)
        plt.tight_layout()

    '''
    Cortafuegos: Si la diferencia entre frecuencias es muy grande => error
    '''

    text_error ='''
    Incompatibilidad de frecuencias en: 
            {:s}\n
        Muestra:              {:.3f} Hz
        Fondo:                {:.3f} Hz
        En nombre de archivo: {:.3f} Hz
    '''
    incompat = np.array([abs(frec_m-frec_f)/frec_f>0.02,abs(frec_m-frec_nombre[k])/frec_f > 0.05],dtype=bool)
    if incompat.any():
        raise Exception(text_error.format(fnames_m[k],frec_m,frec_f,frec_nombre[k_f]))
    else:
        pass
    
    t_m = df_m['t'].to_numpy() #Muestra
    v_m = df_m['v'].to_numpy()
    v_r_m = df_m['v_r'].to_numpy()
    
    '''
    Resto fondo e interpolo señal
    Ejecuto resta_inter() sobre fem de muestra
    '''
    if graficos['Resta_m-f']==1:
        Resta_m , t_m_1 , v_r_m_1 , figura_m = resta_inter(t_m,v_m,v_r_m,fase_m,frec_m,offset_m,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,'muestra')
    else:
        Resta_m , t_m_1 , v_r_m_1 , figura_m = resta_inter(t_m,v_m,v_r_m,fase_m,frec_m,offset_m,df_f['t'],df_f['v'],df_f['v_r'],fase_f,frec_f,0)
    
    # Grafico las restas 
    if graficos['Resta_mf_y_cf']==1:
        plt.figure(figsize=(10,5))
        plt.plot(t_m_1,Resta_m,'.-',lw=0.9,label='Muestra - fondo')
        # plt.plot(t_c_1,Resta_c,'.-',lw=0.9,label='Calibracion - fondo')
        plt.grid()
        plt.legend(loc='best')
        plt.title('Resta de señales')
        plt.xlabel('t (s)')
        plt.ylabel('Resta (V)')
        plt.show()
    else:
        pass
    '''
    Recorto las señales  para tener N periodos enteros 
    Ejecuto recorte() sobre fem de muestra
    '''
    if graficos['Recorte_a_periodos_enteros_m']==1:
        t_m_3, v_r_m_3 , Resta_m_3, N_ciclos_m, figura_m_3 = recorte(t_m_1,v_r_m_1,Resta_m,frec_m,'muestra')
    else:
        t_m_3, v_r_m_3 , Resta_m_3, N_ciclos_m, figura_m_3 = recorte(t_m_1,v_r_m_1,Resta_m,frec_m,0)
    
    '''
    Ultimo ajuste sobre las señales de referencia 
    Ejecuto ajusta_seno() en fem de campo
    '''
    _,amp_final_m, frec_final_m,fase_final_m = ajusta_seno(t_m_3,v_r_m_3)

    '''
    Promedio los N periodos en 1 
    Ejecuto promediado_ciclos() sobre fem muestra y fem campo
    '''   
    t_f_m , fem_campo_m , R_m , dt_m = promediado_ciclos(t_m_3,v_r_m_3,Resta_m_3,frec_final_m,N_ciclos_m)

    '''
    Integro los ciclos: calcula sumas acumuladas y 
    convierte a fem a campo y magnetizacion
    La integral de la fem_campo_m es proporcional a H
    [C_norm_campo]=[A]*[1/m]+[A/m]=A/m - Cte que dimensionaliza 
    al campo en A/m a partir de la calibracion realizada 
    sobre la bobina del RF
    '''
    C_norm_campo=Idc[k]*pendiente_HvsI+ordenada_HvsI 
    campo_ua0_m = dt_m*cumulative_trapezoid(fem_campo_m,initial=0) #[campo_ua0_c]=V*s
    campo_ua_m = campo_ua0_m - np.mean(campo_ua0_m) #Resto offset
    campo_m  = (campo_ua_m/max(campo_ua_m))*C_norm_campo #[campo_c]=A/m 
    
    '''
    La integral de la fem de la muestra (c/ fondo restado), 
    es proporcional a la M mas una constante'''
    magnetizacion_ua0_m = dt_m*cumulative_trapezoid(R_m,initial=0)#[magnetizacion_ua0_c]=V*s
    magnetizacion_ua_m = magnetizacion_ua0_m-np.mean(magnetizacion_ua0_m)#Resto offset
    
    '''
    Ajuste Lineal sobre ciclo para obtener la polaridad
    '''
    pendiente , ordenada = np.polyfit(campo_m,magnetizacion_ua_m,1) #[pendiente]=m*V*s/A  [ordenada]=V*s
    polaridad = np.sign(pendiente) # +/-1
    pendiente = pendiente*polaridad # Para que sea positiva
    magnetizacion_ua_m = magnetizacion_ua_m*polaridad  #[magnetizacion_ua_c]=V*s


    '''
    Analisis de Fourier 
    Ejecuto fourier_señales_5() sobre Resta_m_3, ie la señal fem sin fondo y recortada a N ciclos

    Recupero:
        - figuras de espectros y señales
        - fem reconstruida a N armonicos impares 
        - defasaje entre fems de H y M para armonico fundamental
        - frecuencia, amplitud y fase del armonico fundamental
        - Espectro (f,amp,phi) de muestra
        - Espectro (f,amp,phi) de referencia (fem de campo)
    '''
#%
    if Analisis_de_Fourier == 1:
        fig_fourier, fig2_fourier, muestra_rec_impar,delta_phi_0,f_0,amp_0,fase_0, espectro_f_amp_fase_m,espectro_ref = fourier_señales_5(t_m_3,Resta_m_3,v_r_m_3,
                                                                                                        delta_t=delta_t[k],polaridad=polaridad,
                                                                                                        filtro=0.05,frec_limite=2*N_armonicos_impares*frec_final_m,
                                                                                                        name=fnames_m[k])
        
        # Guardo graficos fourier: señal/espectro y señal impar/espectro filtrado
        fig_fourier.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_Espectro.png'),dpi=200,facecolor='w')
        fig2_fourier.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_Rec_impar.png'),dpi=200,facecolor='w')
        
        print(f'\nFrecuencia fundamental referencia: {espectro_ref[0]:.2f} Hz')
        print(f'\nFrecuencia fundamental muestra: {f_0:.2f} Hz')
        print(f'Duracion de señal recortada: {(t_m_3[-1]-t_m_3[0]):.2e}')
        print(f'N de periodos enteros: {N_ciclos_m}')
        print(f'Num de samples: {len(Resta_m_3)}')
        
        '''
        Calculo de SAR
        A partir de Julio 23 calculo SAR a partir 
        del analisis armonico sobre señal de fem
        '''
        Hmax=max(campo_m)
        N = len(Resta_m_3)
        sar = mu_0*Hmax*(amp_0*C_Vs_to_Am_magnetizacion)*np.sin(delta_phi_0)/(concentracion*N)
        
        print(f'\nSAR: {sar:.2f} (W/g)')
        print(f'Concentracion: {concentracion/1000} g/L')
        print(f'Fecha de la medida: {fecha_m[k]}')
        print('-'*50)

        tau=np.tan(delta_phi_0)/(2*np.pi*frec_final_m)

        #salvo espectros en ASCII, excepto para el fondo01

        if 'fondo' not in fnames_m[k]:
                 
            spec_out = Table([col for col in espectro_f_amp_fase_m])
        
            encabezado = ['Frecuencia_Hz','Amplitud', 'Fase']
            formato = {'Frecuencia_Hz':'%e','Amplitud':'%e', 'Fase':'%e'} 
            
            spec_out.meta['comments'] = [fnames_m[k][:-4],
                                          f'Frecuencia_campo_Hz_=_{espectro_ref[0]}',
                                          f'Magnitud_campo_Hz_=_{espectro_ref[1]}',
                                          f'Fase_campo_=_{espectro_ref[2]}',
                                          f'Concentracion g/m^3_=_{concentracion}',
                                          f'C_Vs_to_Am_M_A/Vsm_=_{C_Vs_to_Am_magnetizacion}',
                                          f'pendiente_HvsI_1/m_=_{pendiente_HvsI}',
                                          f'ordenada_HvsI_A/m_=_{ordenada_HvsI}\n']
        
            output_spec = os.path.join(output_dir, fnames_m[k][:-4] + '_espectro.txt') # Especificar la ruta completa del archivo de salida
            ascii.write(spec_out,output_spec,names=encabezado,overwrite=True,delimiter='\t',formats=formato)
            

        # Hasta aca lo que tiene que ver con analisis armonico
        '''
        Reemplazo señal recortada con la filtrada en armonicos impares: 
            Resta_m_3 ==> muestra_rec_impar 
        Ejecuto promediado_ciclos() sobre muestra_rec_impar
        '''
        t_f_m , fem_campo_m , fem_mag_m , dt_m = promediado_ciclos(t_m_3,v_r_m_3,muestra_rec_impar,frec_final_m,N_ciclos_m)
        
        magnetizacion_ua0_m_filtrada = dt_m*cumulative_trapezoid(fem_mag_m,initial=0)
        magnetizacion_ua_m_filtrada = magnetizacion_ua0_m_filtrada-np.mean(magnetizacion_ua0_m_filtrada)
    
    else:
        #Sin analisis de Fourier, solamente acomodo la polaridad de la señal de la muestra. 
        t_f_m , fem_campo_m , fem_mag_m , dt_m = promediado_ciclos(t_m_3,v_r_m_3,Resta_m_3*polaridad,frec_final_m,N_ciclos_m)
    '''
    Asigno unidades a la magnetizacion 
    '''
    magnetizacion_m = C_Vs_to_Am_magnetizacion*magnetizacion_ua_m #[magnetizacion_m]=A/m
    magnetizacion_m_filtrada = C_Vs_to_Am_magnetizacion*magnetizacion_ua_m_filtrada #[magnetizacion_m_filtrada]=A/m
    '''
    Ploteo H(t) y M(t) normalizados
    '''
    if graficos['Campo_y_Mag_norm_m']==1: 
        fig , ax =plt.subplots(figsize=(8,5),constrained_layout=True)    
        ax.plot(t_f_m,campo_m/max(campo_m),'tab:red',label='H')
        #ax.plot(t_f_m,campo_ua_m_sf/max(campo_ua_m_sf),label='Campo s/ fase electronica')
        ax.plot(t_f_m,magnetizacion_m/max(magnetizacion_m_filtrada),label='M')
        ax.plot(t_f_m,magnetizacion_m_filtrada/max(magnetizacion_m_filtrada),label='M filt impares')
        # ax.plot(t_f_m_fase,magnetizacion_m_filtrada_fase/max(magnetizacion_m_filtrada_fase),label='M filt corr de fase')
        plt.xlim(0,t_f_m[-1])
        plt.legend(loc='best',ncol=2)
        plt.grid()
        plt.xlabel('t (s)')
        plt.title('Campo y magnetización normalizados de la muestra\n'+fnames_m[k][:-4])
        plt.savefig(fnames_m[k]+'_H_M_norm.png',dpi=200,facecolor='w')    
    '''
    Calculo Coercitividad (Hc) y Remanencia (Mr) 
    '''
    m = magnetizacion_m
    m_filt = magnetizacion_m_filtrada 
    h = campo_m
    Hc = []   
    Mr = [] 
    
    for z in range(0,len(m)-1):
        if ((m_filt[z]>0 and m_filt[z+1]<0) or (m_filt[z]<0 and m_filt[z+1]>0)): #M remanente
            Hc.append(abs(h[z] - m_filt[z]*(h[z+1] - h[z])/(m_filt[z+1]-m_filt[z])))
            
        if((h[z]>0 and h[z+1]<0) or (h[z]<0 and h[z+1]>0)):  #H coercitivo
            Mr.append(abs(m_filt[z] - h[z]*(m_filt[z+1] - m_filt[z])/(h[z+1]-h[z])))

    Hc_mean = np.mean(Hc)
    Hc_mean_kAm = Hc_mean/1000
    Hc_error = np.std(Hc)
    Mr_mean = np.mean(Mr)
    Mr_mean_kAm = Mr_mean/1000
    Mr_error = np.std(Mr)
    print(f'\nHc = {Hc_mean:.2f} (+/-) {Hc_error:.2f} (A/m)')
    print(f'Mr = {Mr_mean:.2f} (+/-) {Mr_error:.2f} (A/m)')
    #%
    '''
    Ploteo ciclo de histeresis individual
    '''
    if graficos['Ciclo_HM_m']==1: 
        
        color = cmap(norm(temp_m[k]))
        fig , ax =plt.subplots(figsize=(7,5.5), constrained_layout=True)    
        ax.plot(campo_m,magnetizacion_m,'.-',label=f'{fnames_m[k].split("_")[-1].split(".txt")[0]}')
        ax.plot(campo_m,magnetizacion_m_filtrada,color=color,label=f'{fnames_m[k].split("_")[-1].split(".txt")[0]} ({N_armonicos_impares} armónicos)')
        ax.scatter(0,Mr_mean,marker='s',c='tab:orange',label=f'M$_r$ = {Mr_mean:.0f} A/m')
        ax.scatter(Hc_mean,0,marker='s',c='tab:orange',label=f'H$_c$ = {Hc_mean:.0f} A/m')
        # ax.plot(campo_m,magnetizacion_m_filtrada_fase,label='Muestra filtrada s/fase')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('H $(A/m)$')
        plt.ylabel('M $(A/m)$')
        plt.title('Ciclo de histéresis de la muestra\n'+fnames_m[k][:-4])
        plt.text(0.75,0.25,f'T = {temp_m[k]} °C\nSAR = {sar:0.1f} W/g',fontsize=13,bbox=dict(color='tab:red',alpha=0.6),
                 va='center',ha='center',transform=ax.transAxes)
        plt.savefig(fnames_m[k][:-4]+ '_ciclo_H_M.png',dpi=200,facecolor='w')
#%
    if 'fondo' not in fnames_m[k]:
        if graficos['Susceptibilidad_M_0']==1:
            try:
                susc_a_M_0=susceptibilidad_M_0(campo_m,magnetizacion_m_filtrada,fnames_m[k][:-4],Hc_mean)
            except UnboundLocalError:
                print('Error al determinar el cruce por M=0')
                susc_a_M_0=0 
        else:
            try:
                susc_a_M_0=susceptibilidad_M_0(campo_m,magnetizacion_m_filtrada,fnames_m[k][:-4],Hc_mean)
            except UnboundLocalError:
                print('Error al determinar el cruce por M=0')
                susc_a_M_0=0
    #plt.close(fig='all')   #cierro todas las figuras
    '''
    Salidas importantes:  
    Tienen tantos elmentos como archivos seleccionados
    '''

    #Ajuste sobre señales de referencia
    Frecuencia_ref_muestra_kHz.append(frec_final_m/1000)#Frecuencia de la referencia en la medida de la muestra
    Frecuencia_ref_fondo_kHz.append(frec_f/1000)        #Frecuencia de la referencia en la medida del fondo
    
    #Analisis armonico
    long_arrays.append(len(Resta_m_3)) 
    Frec_fund.append(f_0)
    Magnitud_1er_arm.append(amp_0)
    Defasaje_1er_arm.append(delta_phi_0)
    Tau.append(tau)                                 #Calculado con magnitud y defasaje
    SAR.append(sar)                                 #Calculado con magnitud y defasaje 
    
    # cociente_f1_f0.append(espectro_f_amp_fase_m[1][1]/espectro_f_amp_fase_m[1][0])
    # cociente_f2_f0.append(espectro_f_amp_fase_m[1][2]/espectro_f_amp_fase_m[1][0])

    #Reconstruccion impar, integracion 
    Ciclos_eje_H.append(campo_m)
    Ciclos_eje_M.append(magnetizacion_m)
    Ciclos_eje_M_filt.append(magnetizacion_m_filtrada)
    Ciclos_eje_M_filt_norm.append(magnetizacion_m_filtrada/max(magnetizacion_m_filtrada))
    Campo_maximo.append(Hmax)                       #Campo maximo en A/m
    Mag_max.append(max(magnetizacion_m_filtrada))
    Coercitividad_kAm.append(Hc_mean/1000)          #Campo coercitivo en kA/m
    Remanencia_Am.append(Mr_mean)             #Magnetizacion remanente en kA/m
    xi_M_0.append(susc_a_M_0) 

    '''
    Exporto ciclos de histeresis en ASCII, primero en V.s, despues en A/m : 
    | Tiempo (s) | Campo (V.s) | Magnetizacion (V.s) | Campo (A/m) |  Magnetizacion (A/m)
    '''
    col0 = t_f_m - t_f_m[0]
    col1 = campo_ua_m
    col2 = magnetizacion_ua_m
    col3 = campo_m/1000 #kA/m
    col4 = magnetizacion_m_filtrada#A/m
    
    ciclo_out = Table([col0, col1, col2,col3,col4])

    encabezado = ['Tiempo_(s)','Campo_(V.s)', 'Magnetizacion_(V.s)','Campo_(kA/m)', 'Magnetizacion_(A/m)']
    formato = {'Tiempo_(s)':'%e' ,'Campo_(V.s)':'%e','Magnetizacion_(V.s)':'%e','Campo_(kA/m)':'%e','Magnetizacion_(A/m)':'%e'} 
    
    ciclo_out.meta['comments'] = [f'Temperatura_=_{temp_m[k]}',
                                  f'Concentracion g/m^3_=_{concentracion}',
                                  f'C_Vs_to_Am_M_A/Vsm_=_{C_Vs_to_Am_magnetizacion}',
                                  f'pendiente_HvsI_1/m_=_{pendiente_HvsI}',
                                  f'ordenada_HvsI_A/m_=_{ordenada_HvsI}',
                                  f'frecuencia_Hz_=_{frec_final_m}\n']
    
    

    # Especificar la ruta completa del archivo de salida
    output_file = os.path.join(output_dir, fnames_m[k][:-4] + '_ciclo_H_M.txt')

    ascii.write(ciclo_out,output_file,names=encabezado,
    overwrite=True,delimiter='\t',formats=formato)

plt.close('all')

#%% PLOTEO TODOS LOS CICLOS CRUDOS

cmap = mpl.colormaps['jet'] #'viridis'
norm = plt.Normalize(temp_m.min(), temp_m.max())# Crear un rango de colores basado en las temperaturas y el cmap

if graficos['Ciclos_HM_m_todos']==1:
    fig = plt.figure(figsize=(9,7),constrained_layout=True)
    ax = fig.add_subplot(1,1,1)

    for i in range(len(fnames_m)): 
        if 'fondo' not in fnames_m[i]:
            color = cmap(norm(temp_m[i]))
            plt.plot(Ciclos_eje_H[i]/1000,Ciclos_eje_M[i],color=color,label=f'{fnames_m[i].split("_")[-1].split(".")[0]:<4s} {temp_m[i]:>5.1f} ºC')
        else:    
            plt.plot(Ciclos_eje_H[-1]/1000,Ciclos_eje_M[-1],'.-',color='k',label='fondo')

# plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),ncol=4,fancybox=True)

# Configurar la barra de colores
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Esto es necesario para que la barra de colores muestre los valores correctos
plt.colorbar(sm, label='Temperatura',ax=ax)  # Agrega una etiqueta adecuada


plt.text(0.15,0.75,f'{frec_nombre[0]/1000:>3.0f} kHz\n{round(np.mean(Campo_maximo)/1e3):>4.1f} kA/m',fontsize=20,bbox=dict(color='tab:orange',alpha=0.7),transform=ax.transAxes)
plt.grid()
plt.xlabel('H (kA/m)',fontsize=15)
plt.ylabel('M (A/m)',fontsize=15)
plt.suptitle('Ciclos de histéresis (sin filtrar)',fontsize=22)
plt.title(fecha_graf +' Sin filtrado',loc='left',fontsize=13)

plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_ciclos_MH_raw.png'),dpi=300,facecolor='w')

#%% PLOTEO TODOS LOS CICLOS FILTRADOS

if Analisis_de_Fourier==1:
    fig = plt.figure(figsize=(9,7),constrained_layout=True)
    ax = fig.add_subplot(1,1,1)

    for i in range(0,len(fnames_m),24):
        if 'fondo' not in fnames_m[i]:
            color = cmap(norm(temp_m[i]))
            plt.plot(Ciclos_eje_H[i]/1000,Ciclos_eje_M_filt[i],'-',color=color,label=f'{fnames_m[i].split("_")[-1].split(".")[0]:<4s} {temp_m[i]:>5.1f} ºC')
        else:
            plt.plot(Ciclos_eje_H[i]/1000,Ciclos_eje_M_filt[i],'-',color='k',label='fondo')
plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),ncol=3,fancybox=True)

# Configurar la barra de colores
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Esto es necesario para que la barra de colores muestre los valores correctos
plt.colorbar(sm, label='Temperatura', ax=ax)  # Agrega una etiqueta adecuada

plt.text(0.15,0.75,f'{frec_nombre[0]/1000:>3.0f} kHz\n{round(np.mean(Campo_maximo)/1e3):>4.1f} kA/m',fontsize=20,bbox=dict(color='tab:orange',alpha=0.7),transform=ax.transAxes)
plt.grid()
plt.xlabel('H (kA/m)',fontsize=15)
plt.ylabel('M (A/m)',fontsize=15)
plt.title(fecha_graf + f'   {N_armonicos_impares} arm impares',loc='left',fontsize=13)
plt.suptitle('Ciclos de histéresis (filtrado impar)',fontsize=22)
#plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_ciclos_MH.png'),dpi=300,facecolor='w')
# plt.show()

#%% PLOTEO TODOS LOS CICLOS FILTRADOS + NORM

if Analisis_de_Fourier==1:
    fig = plt.figure(figsize=(9,7),constrained_layout=True)
    ax = fig.add_subplot(1,1,1)

    for i in range(len(fnames_m)):
        if 'fondo' not in fnames_m[i]:
            color = cmap(norm(temp_m[i]))
            plt.plot(Ciclos_eje_H[i]/1000,Ciclos_eje_M_filt_norm[i],'-',color=color,label=f'{fnames_m[i].split("_")[-1].split(".")[0]:<4s} {temp_m[i]:>5.1f} ºC')
        else:
            plt.plot(Ciclos_eje_H[i]/1000,Ciclos_eje_M_filt_norm[i],'-',color='k',label='fondo')
#plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),ncol=3,fancybox=True)

# Configurar la barra de colores
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Esto es necesario para que la barra de colores muestre los valores correctos
plt.colorbar(sm, label='Temperatura', ax=ax)  # Agrega una etiqueta adecuada

plt.text(0.15,0.75,f'{frec_nombre[0]/1000:>3.0f} kHz\n{round(np.mean(Campo_maximo)/1e3):>4.1f} kA/m',fontsize=20,bbox=dict(color='tab:orange',alpha=0.7),transform=ax.transAxes)
plt.grid()
plt.xlabel('H (kA/m)',fontsize=15)
plt.ylabel('M/M$_{max}$',fontsize=15)
plt.title(fecha_graf + f'   {N_armonicos_impares} arm impares',loc='left',fontsize=13)
plt.suptitle('Ciclos de histéresis (filtrado impar)',fontsize=22)
#plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_ciclos_MH_norm.png'),dpi=300,facecolor='w')
# plt.show()
#%% RESULTADOS.TXT
'''
Guardo ASCII con los datos de todo el procesamiento: 
    |fname|time|Temp|Mr|Hc|Hmax|Mmax|f0|mag0|phi0|dphi0|SAR|Tau|N|xi_M_0|    
'''
SAR_all = ufloat(np.mean(SAR),np.std(SAR))
defasaje_all= ufloat(np.mean(Defasaje_1er_arm),np.std(Defasaje_1er_arm))
Coercitividad_all = ufloat(np.mean(Coercitividad_kAm),np.std(Coercitividad_kAm))
Remanencia_all = ufloat(np.mean(Remanencia_Am),np.std(Remanencia_Am))
xi_all = ufloat(np.mean(xi_M_0),np.std(xi_M_0))
tau_all= ufloat(np.mean(Tau),np.std(Tau))

if templog:
    col1 = [t.strftime("%d/%m/%Y %H:%M:%S") for t in time_m]
else:
    col1= np.zeros_like(fnames_m)

if any('fondo' in fname for fname in fnames_m):
    fnames_m=fnames_m[:-1]
    temp_m=temp_m[:-1]
    Remanencia_Am=Remanencia_Am[:-1]
    Coercitividad_kAm=Coercitividad_kAm[:-1]
    Campo_maximo=Campo_maximo[:-1]
    Mag_max=Mag_max[:-1]
    Frec_fund=Frec_fund[:-1]
    Magnitud_1er_arm=Magnitud_1er_arm[:-1]
    Defasaje_1er_arm=Defasaje_1er_arm[:-1]
    SAR=SAR[:-1]
    Tau=Tau[:-1]
    xi_M_0=xi_M_0[:-1]
    cociente_f1_f0=cociente_f1_f0[:-1]
    cociente_f2_f0=cociente_f2_f0[:-1]
    long_arrays=long_arrays[:-1]
    col1=col1[:-1]

col0 = fnames_m

col2 = temp_m
col3 = Remanencia_Am
col4 = Coercitividad_kAm
col5 = Campo_maximo
col5_bis = Mag_max
col6 = Frec_fund
col7 = Magnitud_1er_arm
col8 = Defasaje_1er_arm
col9 = SAR
col10 = Tau
col11 = long_arrays
col12 = xi_M_0
resultados_out=Table([col0, col1,col2,col3,col4,col5,col5_bis,col6,col7,col8,col9,col10,col11,col12])
encabezado = ['Nombre_archivo',
              'Time_m',
              'Temperatura_(ºC)',
              'Mr_(A/m)',
              'Hc_(kA/m)',
              'Campo_max_(A/m)',
              'Mag_max_(A/m)',
              'f0',
              'mag0',
              'dphi0',
              'SAR_(W/g)',
              'Tau_(s)',
              'N',
              'xi_M_0'] 

formato = {'Nombre_archivo':'%s' ,'Time_m':'%s','Temperatura_(ºC)':'%.1f',
           'Mr_(A/m)':'%.2f','Hc_(kA/m)':'%.2f','Campo_max_(A/m)':'%.2f',
           'Mag_max_(A/m)':'%.2f','f0':'%e','mag0':'%e','dphi0':'%e',
           'SAR_(W/g)':'%.2f','Tau_(s)':'%e','N':'%.0f','xi_M_0':'%.3e'} 

resultados_out.meta['comments'] = ['Configuracion:',
                              f'Concentracion g/m^3_=_{concentracion}',
                              f'C_Vs_to_Am_M_A/Vsm_=_{C_Vs_to_Am_magnetizacion}',
                              f'pendiente_HvsI_1/m_=_{pendiente_HvsI}',
                              f'ordenada_HvsI_A/m_=_{ordenada_HvsI}',
                              f'frecuencia_ref_Hz_=_{frec_final_m}',
                              '\nResultados:',
                              f'tau_s_=_{tau_all:.2e}',
                              f'dphi_rad_=_{defasaje_all:.2f}',
                              f'SAR_W/g_=_{SAR_all:.2f}',
                              f'Hc_kA/m_=_{Coercitividad_all:.2f}',
                              f'Mr_A/m_=_{Remanencia_all:.2f}',
                              f'Suceptibilidad_a_M=0_=_{xi_all:.4f}\n']
output_file2=os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_resultados.txt')
ascii.write(resultados_out,output_file2,names=encabezado,overwrite=True,
delimiter='\t',formats=formato)


#%% dphi | tau | SAR

# Definir el mapa de colores (jet en este caso)
cmap = mpl.colormaps['jet'] #'viridis'
# Normalizar las temperaturas al rango [0, 1] para obtener colores
normalized_temperaturas = (temp_m - temp_m.min()) / (temp_m.max() - temp_m.min())
# Obtener los colores correspondientes a las temperaturas normalizadas
colors = cmap(normalized_temperaturas)

fig, axs = plt.subplots(3, 1, figsize=(7,7), constrained_layout=True,sharex=True)
axs[0].scatter(temp_m, Defasaje_1er_arm,  c=colors,marker='o', label='$\Delta\phi_0$')
axs[0].plot(temp_m, Defasaje_1er_arm,zorder=-1)
axs[0].legend()
axs[0].grid()
axs[0].set_ylabel('$\Delta\phi_0$')

axs[1].scatter(temp_m, Tau,c=colors, marker='s', label=r'$\tau$')
axs[1].plot(temp_m, Tau,zorder=-1)
axs[1].legend()
axs[1].grid()
axs[1].set_ylabel(r'$\tau$ (s)')

axs[2].scatter(temp_m, SAR,c=colors,marker= 'v', label=f'{concentracion/1000:.2f} g/L')
axs[2].plot(temp_m, SAR,zorder=-1)
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel('T (°C)')
axs[2].set_ylabel('SAR (W/g)')

plt.suptitle(r'Defasaje - $\tau$ - SAR'+f'\n{nombre.strip("*")} - {frec_nombre[0]/1000:>3.0f} kHz - {round(np.mean(Campo_maximo)/1e3):>4.1f} kA/m')
#plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_dphi_tau_SAR_vs_T.png'),dpi=200,facecolor='w')

#%% Hc | Mr | xi

fig, axs = plt.subplots(3, 1, figsize=(7,7), constrained_layout=True,sharex=True)
axs[0].scatter(temp_m,Coercitividad_kAm,c=colors, marker='o',label='H$_C$')
axs[0].plot(temp_m,Coercitividad_kAm,zorder=-1)
axs[0].legend()
axs[0].grid()
axs[0].set_ylabel('Campo Coercitivo (kA/m)')

axs[1].scatter(temp_m, xi_M_0,c=colors, marker='s',label='$\chi$ a M=0')
axs[1].plot(temp_m, xi_M_0,zorder=-1)
axs[1].legend()
axs[1].grid()
axs[1].set_ylabel('Susceptibilidad a M=0')

axs[2].scatter(temp_m, Remanencia_Am,c=colors, marker='D',label='M$_R$')
axs[2].plot(temp_m, Remanencia_Am,zorder=-1)
axs[2].legend()
axs[2].grid()
axs[2].set_ylabel('Magnetizacion Remanente')
axs[2].set_xlabel('T (°C)')
plt.suptitle('Campo Coercitivo, Magnetizacion Remanente'+f'\n{nombre.strip("*")} - {frec_nombre[0]/1000:>3.0f} kHz - {round(np.mean(Campo_maximo)/1e3):>4.1f} kA/m')
#plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_Hc_Mr_xi_vs_T.png'),dpi=200,facecolor='w')

#%% PRINTEO PARAMS
print(f'''tau = {tau_all} s
SAR = {SAR_all:.0f} W/g
dphi = {defasaje_all} rad 
        
Coercitivo = {Coercitividad_all:.2f} kA/m
Remanencia = {Remanencia_all:.0f} A/m
Susceptibilidad a M=0 = {xi_all:.e}
      ''')
# %% Cociente magnitudes del 1er y2do armonico

# fig, axs = plt.subplots(1, 1, figsize=(6,4), constrained_layout=True,sharex=True)
# # Gráfico 1
# axs.scatter(temp_m,cociente_f1_f0, marker='o',c=colors)
# axs.plot(temp_m,cociente_f1_f0,label='mag$_1$/mag$_0$',zorder=-1)
# axs.axhline(1,0,1,ls='-.',c='k',zorder=0)
# axs.grid()
# axs.scatter(temp_m, cociente_f2_f0, marker='s',c=colors)
# axs.plot(temp_m,cociente_f2_f0, label='mag$_2$/mag$_0$',zorder=-1)
# axs.legend()
# plt.title('Cociente de magnitudes de armonicos')
# plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_cociente_magnitudes.png'),dpi=200,facecolor='w')


#%%%
end_time = time.time()

print(f'Tiempo de ejecución del script: {(end_time-start_time):6.3f} s.')

#%% PLOTEO PARA PAPER
fig = plt.figure(figsize=(7.2,5.4),constrained_layout=True)
ax = fig.add_subplot(1,1,1)
indices = [0,92] 
for i in indices:
    color = cmap(norm(temp_m[i]))
    plt.plot(Ciclos_eje_H[i]/1000,Ciclos_eje_M_filt[i],'-',color=color,label=f'{fnames_m[i].split("_")[-1].split(".")[0]:<4s} {temp_m[i]:>5.1f} ºC')

#plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),ncol=3,fancybox=True)

# Configurar la barra de colores
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Esto es necesario para que la barra de colores muestre los valores correctos
colorbar = plt.colorbar(sm, ax=ax)  # Agrega una etiqueta adecuada
colorbar.set_label('Temperature', fontsize=15)  # Cambiar el título y ajustar el tamaño de fuente

plt.grid()
plt.xlabel('H (kA/m)',fontsize=15)
plt.ylabel('M (A/m)',fontsize=15)
#plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_ciclos_MH_paper.png'),dpi=400,facecolor='w')
# plt.show()


# %%
