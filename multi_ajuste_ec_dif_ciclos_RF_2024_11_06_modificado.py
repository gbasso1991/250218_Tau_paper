# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:12:20 2020

@author: Pedro

> Importa archivos generados por código en Python y realiza un ajuste y gráfico
de ciclos de histéresis bajo campos RF. Genera gráficos de H vs t, M vs t y M vs H.
Los archivos contienen CINCO columnas:
    
    Tiempo_(s)    Campo_(V.s)    Magnetización_(V.s)    Campo_(kA/m)    Magnetización_(A/m)
    
> Ajustamos con una función seno el campo, y la magnetización usando una
ecuación diferencial basada en Rozensweit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
from lmfit.models import ExpressionModel
from lmfit import Model, Parameter, report_fit
import tkinter as tk
from tkinter import filedialog
import os
import glob
from uncertainties import ufloat
from uncertainties.umath import *
import lmfit
from io import StringIO  # Import necesario para leer los archivos sin caracteres nulos

# Archivo de salida: DataFrame de Pandas
output = pd.DataFrame()

flagseno = 0

# ================= Funciones ============================

def Langevin(x):
    return 1. / np.tanh(x) - 1. / x

def ecdif(x, tau_ed, A_ed, Ms_ed, frec_ed, fase_ed, m0_ed, m_offset_ed):
    """
    d M(H,t) / dt = (1/tau) * ( Meq(H) - M(H,t) )
    Meq(H,t) = Ms * L(aux), donde aux = A * sin(2*pi*frec_ed*t - fase_ed)
    """
    N = 5
    tiempo_N = x
    delta = x[1] - x[0]
    # Se extiende el vector de tiempo (ajústalo según la lógica de tu modelo)
    for k in np.arange(N - 1):
        tiempo_N = np.append(tiempo_N, x + tiempo_N[-1] + delta - x[0])
    m_ec = odeint(dm_dt, m0_ed / Ms_ed, t=tiempo_N, args=(tau_ed, A_ed, frec_ed, fase_ed))
    m_ec_out = Ms_ed * np.array(m_ec).flatten()[-len(x) - 1:-1] + m_offset_ed
    return m_ec_out

def dm_dt(m, t, tau, A, frec_h, fase_h):
    aux = A * np.sin(2 * np.pi * frec_h * t - fase_h)
    meq = Langevin(aux)
    return (1. / tau) * (meq - m)

# ================= SELECCIÓN DE ARCHIVOS =================

busqueda_automatica = 1
filePaths = []
fileNames = []

if busqueda_automatica == 0:
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    files = filedialog.askopenfilename(multiple=True)
else:
    directorio = "G:/Mi unidad/IFLP/Publicaciones/2024-Tau&FFT"
    files = glob.glob(os.path.join(os.getcwd(), "Analisis_20240314_162142", "*_ciclo_H_M*"), recursive=True)

for f in files:
    filePaths.append(f)
    head, tail = os.path.split(f)
    fileNames.append(tail)

# ================= ABRIMOS ARCHIVO CON PANDAS Y AJUSTAMOS =====================

for i in range(len(fileNames)):
    # Leemos el archivo eliminando caracteres nulos
    with open(filePaths[i], "r", encoding='ISO-8859-1', errors='ignore') as archivo:
        lines = archivo.readlines()
    # Reemplazamos cualquier carácter NUL en cada línea
    lines = [line.replace('\x00', '') for line in lines]
    
    df = pd.read_table(StringIO("".join(lines)), 
                       names=['tiempo', 'a', 'b', 'campo', 'magnetizacion'], 
                       sep='\t', engine='python', skiprows=8)

    # Extraemos la temperatura del archivo
    with open(filePaths[i], "r", encoding='ISO-8859-1', errors='ignore') as archivo:
        for linea in archivo:
            if linea.startswith("# Temperatura_=_"):
                temperatura = float(linea.split("_=_")[-1].strip())
                print('Temperatura del ciclo =', temperatura)
                break

    # Ajustes de tiempo y campo
    deltat = 1
    df['tiempo'] = df['tiempo'] * deltat
    df['campo'] = df['campo'] * 10**3
    df = df[['tiempo', 'campo', 'magnetizacion']]
    print(df.head())

    # ================= AJUSTAMOS USANDO LMFIT MODELS ============================
    
    # Modelo para ajustar el campo y la magnetización (función seno)
    gmod = ExpressionModel("offset + amp * sin(2.*pi*frec*x - fase)")
    
    # --- Ajuste del campo ---
    i_offset = df['campo'].mean()
    i_amp = (df['campo'].max() - df['campo'].min()) / 2
    i_frec = 1 / df['tiempo'].max()
    t_max = df['tiempo'][df['campo'].idxmax()]
    i_fase = 2. * np.pi * i_frec * t_max - np.pi/2
    i_fase = i_fase % (2 * np.pi)
    
    gmod.set_param_hint('offset', value=i_offset)
    gmod.set_param_hint('amp', value=i_amp)
    gmod.set_param_hint('frec', value=i_frec)
    gmod.set_param_hint('fase', value=i_fase)
    pars = gmod.make_params()
    
    result_h = gmod.fit(df['campo'], pars, x=df['tiempo'])
    rsquare_h = 1 - result_h.residual.var() / np.var(df['campo'])
    
    if not result_h.result.errorbars:
        result_h.params['amp'].stderr = 0.
        result_h.params['frec'].stderr = 0.
        result_h.params['fase'].stderr = 0.
        result_h.params['offset'].stderr = 0.
    
    # --- Ajuste de la magnetización con función seno ---
    i_offsetm = df['magnetizacion'].mean()
    i_ampm = (df['magnetizacion'].max() - df['magnetizacion'].min()) / 2
    i_frec = result_h.values['frec']
    i_fasem = result_h.values['fase']
    
    gmod.set_param_hint('offset', value=i_offsetm, min=-0.1 * abs(i_ampm), max=0.1 * abs(i_ampm))
    gmod.set_param_hint('amp', value=i_ampm, min=0, max=2 * i_ampm)
    gmod.set_param_hint('fase', value=i_fasem, min=0, max=2 * np.pi)
    gmod.set_param_hint('frec', value=i_frec, vary=False)
    pars = gmod.make_params()
    
    result_m = gmod.fit(df['magnetizacion'], pars, x=df['tiempo'])
    rsquare_m = 1 - result_m.residual.var() / np.var(df['magnetizacion'])
    
    # --- Ajuste de la magnetización con ecuación diferencial ---
    amp_campo = result_h.params['amp'].value
    frec_campo = result_h.params['frec'].value
    fase_campo = result_h.params['fase'].value
    m0 = df['magnetizacion'][0]
    
    T = 300
    mu0 = 4 * np.pi * 1e-7
    muB = 9.2740097e-24
    kB = 1.3806488e-23
    
    i_tau = tan(result_m.params['fase'].value - result_h.params['fase'].value) / (2 * np.pi * result_h.params['frec'].value)
    NmB = 10000
    i_A = mu0 * muB * NmB * amp_campo / (kB * T)
    i_Ms = 3000
    m_offset = result_m.params['offset'].value
    
    model_mag_ecdif = Model(ecdif)
    model_mag_ecdif.set_param_hint('tau_ed', value=i_tau, min=0.01 * i_tau, max=100 * i_tau)
    model_mag_ecdif.set_param_hint('A_ed', value=i_A, min=0, max=i_A * 1e5)
    model_mag_ecdif.set_param_hint('Ms_ed', value=i_Ms, min=0, max=1000e3)
    model_mag_ecdif.set_param_hint('frec_ed', value=frec_campo, vary=False)
    model_mag_ecdif.set_param_hint('fase_ed', value=fase_campo, vary=False)
    model_mag_ecdif.set_param_hint('m0_ed', value=m0, vary=False)
    model_mag_ecdif.set_param_hint('m_offset_ed', value=m_offset, vary=False)
    
    pars_mag_ecdif = model_mag_ecdif.make_params()
    result_mag_ecdif = model_mag_ecdif.fit(df['magnetizacion'], pars_mag_ecdif, x=df['tiempo'])
    rsquare_mag_ecdif = 1 - result_mag_ecdif.residual.var() / np.var(df['magnetizacion'])
    
    print(f'R^2 (modelo seno) = {rsquare_m:.4f}')
    print(f'R^2 (modelo ecuación diferencial) = {rsquare_mag_ecdif:.4f}')
    
    # ================= GRÁFICOS UNIFICADOS ====================
    # Se genera una figura con dos subgráficos:
    # - Izquierdo: Campo y Magnetización vs Tiempo (con Magnetización multiplicada por 100)
    # - Derecho: Magnetización vs Campo (sin modificaciones)
    
    # Configuración global para fondo blanco y gridlines activadas
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True  # activa gridlines por defecto
    plt.rcParams.update({
    'font.size': 20,         # Tamaño general de fuente
    'axes.labelsize': 22,    # Tamaño de etiquetas de los ejes
    'axes.titlesize': 24,    # Tamaño del título del gráfico
    'xtick.labelsize': 20,   # Tamaño de las etiquetas del eje X
    'ytick.labelsize': 20,   # Tamaño de las etiquetas del eje Y
    'legend.fontsize': 20,   # Tamaño de fuente de la leyenda
    'figure.titlesize': 24   # Tamaño del título general de la figura
})

    # Define la figura con el tamaño deseado en pulgadas y dpi adecuado.
    fig = plt.figure(figsize=(25.6, 8.56), dpi=100)  # Esto producirá 2560x856 píxeles
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
    
    # Subgráfico Izquierdo: Campo y Magnetización vs Tiempo
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(df['tiempo'] * 1e6, df['campo'] * 1e-3, 'o', label='Field', color='#1f77b4',
             markerfacecolor='none', markersize=4)
    ax1.plot(df['tiempo'] * 1e6, result_h.best_fit * 1e-3, '-', label='Field Fit', color='#ff7f0e')
    ax1.plot(df['tiempo'] * 1e6, df['magnetizacion'] * 100 * 1e-3, 'o', label='Magnetization ×100', color='#2ca02c',
             markerfacecolor='none', markersize=4)
    ax1.plot(df['tiempo'] * 1e6, result_mag_ecdif.best_fit * 100 * 1e-3, '-', label='Magnetization Fit ×100', color='#d62728')
    if flagseno == 1:
        ax1.plot(df['tiempo'] * 1e6, result_mag_ecdif.init_fit * 100 * 1e-3, 'k--', label='Init Ec. dif. ×100')
        ax1.plot(df['tiempo'] * 1e6, result_m.best_fit * 100 * 1e-3, '-', label='Sine Fit ×100', color='#ff7f0e')
    ax1.set_xlabel('Time ($\mu$s)')
    ax1.set_ylabel('Field (kA/m) & Magnetization ×100 (kA/m)')
    ax1.legend(title='$R^2$: Field %.5f, Magnetization %.5f' % (rsquare_h, rsquare_m))
    ax1.grid(True)
    ax1.spines['top'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')
    # Subgráfico Derecho: Magnetization vs Field
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(df['campo'] * 1e-3, df['magnetizacion'] * 1e-3, 'o', label='Data', markerfacecolor='none',
             markeredgecolor='blue', markersize=4)
    if flagseno == 1:
        ax2.plot(result_h.best_fit * 1e-3, result_m.best_fit * 1e-3, 'o', label='Linear Response Model', color='#ff7f0e')
    ax2.plot(result_h.best_fit * 1e-3, result_mag_ecdif.best_fit * 1e-3, '-', label='Fit', color='#d62728')
    ax2.set_xlabel('Field (kA/m)')
    ax2.set_ylabel('Magnetization (kA/m)')
    ax2.legend()
    ax2.grid(True)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
    plt.tight_layout()
    # Guardamos la figura antes de mostrarla
    plt.savefig(os.path.join(os.path.dirname(files[i]), fileNames[i] + '_fit.jpg'))
    plt.show()
    plt.close()
