import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from biosppy.signals import ecg 
from scipy.signal import find_peaks
import glob
import os
import random
import math 



####### FUNCION BASELINE #############
def señal_referencia(sanos):
    #############Sincronizar las señales########################
    longitud_intervalo = 500
    intervalos_alineados = [] #Vector para los intervalos que se van a seleccionar
    #############Calculo de la muestra ####################
    N=382
    Z=1.64
    p=0.05
    q=0.95
    d=0.05
    n= round((N*pow(Z,2)*p*q)/(pow(d,2)*(N-1)+(pow(Z,2)*p*q))) #Muestra de las señales que se van a usar para la señal baseline

    señales_ecg_aleatorias = random.sample(sanos, n) #Seleccionar señales aleatorias 
    for archivo_csv in señales_ecg_aleatorias:
        df = pd.read_csv(archivo_csv, header=None) #Leer los archivos de las señales aleatorias seleccionadas 
        devII = df.iloc[:, 1] / 1000  # Dividir entre 1000 para pasar a milivolts
        max_index = np.argmax(devII.values)
        inicio_intervalo = max_index - longitud_intervalo // 2
        fin_intervalo = max_index + longitud_intervalo // 2
        intervalo_alineado = devII.iloc[inicio_intervalo:fin_intervalo].values
        if len(intervalo_alineado) == longitud_intervalo:
            intervalos_alineados.append(intervalo_alineado)
    n_señales_utilizadas = len(intervalos_alineados)
    #print("Número de señales utilizadas:", n)

    intervalos_alineados_matrix = np.array(intervalos_alineados)

    ############## BASELINE PULSE##############
    baseline = np.mean(intervalos_alineados_matrix, axis=0) #Calcular el ecg baseline 
    #plt.plot(baseline)
    #plt.title('Baseline pulse')
    #plt.xlabel('Samples')
    #plt.ylabel('Amplitude (mV)')
    #plt.show()
    return baseline

################# FUNCION DE LA CORRELACIÓN MOVIL #######################
def corr_movil(s_referencia, s_muestra, alphacorr):
    alpha = alphacorr #Umbral de correlacion
    paso = 2 #Cada cuanto se va a calcular el indice de correlacion, en este caso cada 2 muestras
    longitud_referencia = len(s_referencia) 
    longitud_ecg= len(s_muestra)
    n_valores_corr = math.floor((longitud_ecg- longitud_referencia + 1) / paso) 
    valores_correlacion = np.zeros(n_valores_corr)
    valores_mayores = []

    for i in range(1, longitud_ecg - longitud_referencia + 1, paso): 
        subseñal_ecg =s_muestra[i:i+longitud_referencia]
        corrcoef_m = np.corrcoef(s_referencia,subseñal_ecg)
        valor_correlacion = corrcoef_m[0,1]

        if valor_correlacion >= alpha:
            valores_mayores.append(valor_correlacion)

        valores_correlacion[i // paso]=valor_correlacion

    return valores_correlacion, valores_mayores

############### Funcion para procesar la base de datos ####################
def procesar_database(sanas, nsanas, arritmias, narritmias, zeta, alphacorr):
    
    baseline = señal_referencia(sanas)#Calcular la señal de referencia 

    valores_mayores_sanas = []
    valores_correlacion_sanas = []
    frecuencias_cardiacas_sanas = []
    proporciones_sanas = []
    clas_señales_sanas =[]
    contador = 0
    valores_mayores_arr = []
    valores_correlacion_arr = []
    frecuencias_cardiacas_arr = []
    proporciones_arr = []
    clas_señales_arr =[]
    fs=500

    def clasificar(vector, p, n, bpm, qrs_ms, ondaP_ms, PR_ms, valor_critico):
        q = 1 - p
        denominador = (p * q) / n
       
        if denominador == 0:
            c = 1
            vector.append(c)

        elif denominador != 0:
            z = (p - 0.0137882489819662595) / np.sqrt(denominador)
            if z >= (-1) * valor_critico and bpm >= 60 and bpm <= 100 and qrs_ms < 110 and ondaP_ms < 120 and PR_ms < 200:
                c = 0
                vector.append(c)

            else:
                c = 1   
                vector.append(c)

        return vector

        q = 1 - p
        denominador = (p * q) / n
        z = (p - 0.0137882489819662595) / np.sqrt(denominador)
        return z

    #Procesar señales sanas################################################
    for archivo_csv in sanas: #Para cada archivo 
        contador += 1
        df=pd.read_csv(archivo_csv,header=None)
        devII = df.iloc[:,1]/1000
        signal=devII.values
        valores_correlacion, valores_mayores = corr_movil(baseline, devII, alphacorr) #Aplicar la funcion de correlacion movil a todas las señales, aquí le estoy dando el valor de alpha
        ecg_analysis =ecg.ecg(signal=signal, sampling_rate=fs, show=False) #funcion de Biosppy
        bpm =np.mean(ecg_analysis['heart_rate']) #Solo se selecciona el parametro de la frecuencia instantanea 
        frecuencias_cardiacas_sanas.append(bpm) #Frecuencias cardíacas 
        #print(f"Para la señal {archivo_csv}, bpm: {bpm}")
        
        #####QRS#####
        out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
        r_peaks = out['rpeaks']
        window_samples = int(0.08 * fs)
        ##### Duración Onda P ########
        ventana = 200
        ventana_samples = int((ventana/1000)*fs)
        duracion_ondas_P = []
        QRS_durations = []  
        for r_peak in r_peaks:
            signal_window = signal[r_peak - window_samples:r_peak + window_samples]#Seleccionar la parte de la señal del pico R +- las muestras
            index_Q = np.argmin(signal_window[:window_samples])
            index_Q_global = r_peak - window_samples + index_Q 
            index_S = np.argmin(signal_window[window_samples:])
            index_S_global = r_peak - window_samples + window_samples + index_S 
            QRS_duration = index_S_global - index_Q_global
            QRS_durations.append(QRS_duration)
            
            ##### Duracion Onda P######
            inicio_ventana = max(0, r_peak - ventana_samples)
            fin_ventana = r_peak
            ventana_signal = signal[inicio_ventana:fin_ventana]
            p_peak_index = np.argmax(ventana_signal)+ inicio_ventana
            ventana_inicio = max(0, p_peak_index - ventana_samples // 4)
            ventana_fin = min(len(signal), p_peak_index + ventana_samples // 4)
            minimos_inicio, _ = find_peaks(-signal[ventana_inicio:p_peak_index])
            minimos_fin, _ =find_peaks(-signal[p_peak_index:ventana_fin])

            if minimos_inicio.size > 0 and minimos_fin.size > 0: 
                inicio_onda_P = ventana_inicio + minimos_inicio[-1]
                fin_onda_P = p_peak_index + minimos_fin[0]
                duracion = (fin_onda_P-inicio_onda_P)/fs*1000
                duracion_ondas_P.append(duracion)
        
        if duracion_ondas_P: 
            ondaP_ms = np.mean(duracion_ondas_P)
            #print(f"{duracion_onda_P_msec}")

        average_QRS_duration = np.mean(QRS_durations)
        qrs_ms = (average_QRS_duration / fs) * 1000
        #print(f"Duración promedio estimada del complejo QRS: {qrs_ms:.2f} ms"

        #####PR######
        intervalo_PR_durations = []
        for i in range(len(r_peaks) - 1):
            # Encontrar el índice del pico P más cercano al pico R actual
            indice_pico_P = out['rpeaks'][i-1]
            # Calcular la diferencia de tiempo entre el pico P y el pico R actual
            intervalo_PR_duration = (r_peaks[i] - indice_pico_P) / fs * 1000  # Duración en milisegundos
            intervalo_PR_durations.append(intervalo_PR_duration)

        promedio_intervalo_PR = np.mean(intervalo_PR_durations)
        PR_ms = abs(promedio_intervalo_PR)
        #print(f"Duración promedio del intervalo PR: {promedio_intervalo_PR:.2f} ms")
   
        if valores_mayores: 
            prom_corr = np.mean(valores_mayores) #Promedio de los valores de correlación que superan a alpha
            valores_mayores_sanas.append(prom_corr)
        else:
            prom_corr =np.mean(valores_correlacion) #Promedio de todos los valores de correlación
            valores_correlacion_sanas.append(prom_corr)
            
        if contador % 100 ==0:
            print(f"Señal {contador}") #Contador solo para indicar en que señal va 

         ######## Proporcion #########
        rK = len(valores_correlacion)
        beta = len(valores_mayores)
        p = beta/rK
        proporciones_sanas.append(p)
        #print(f"Para la señal {archivo_csv}, la proporción es: {p}")

        #########Calcular el estadístico ########
        n= nsanas #Población de las funciones sanas 
        #clasificar(clas_señales_sanas, p, n, bpm, qrs_ms, ondaP_ms, PR_ms, zeta) #Calcula el valor del estadístico y clasificar

        q = 1 - p
        denominador = (p * q) / n
       
        if denominador == 0:
            c = 1
            clas_señales_sanas.append(c)

        elif denominador != 0:
            z = (p - 0.0137882489819662595) / np.sqrt(denominador)
            if z >= (-1) * zeta and bpm >= 60 and bpm <= 100 and qrs_ms < 110 and ondaP_ms < 120 and PR_ms < 200:
                c = 0
                clas_señales_sanas.append(c)

            else:
                c = 1   
                clas_señales_sanas.append(c)

#Se calcula fuera del ciclo for, para que no se este calculando en cada iteracion y solo hasta el final 
    FP = sum(clas_señales_sanas)
    VN = nsanas - FP
    ESP = VN/nsanas
    
    ######### Procesar arritmias ##########################################################################3
    for archivo_csv in arritmias:
            contador += 1
            df=pd.read_csv(archivo_csv,header=None)
            devII = df.iloc[:,1]/1000
            signal=devII.values
            valores_correlacion, valores_mayores = corr_movil(baseline, devII, alphacorr) #Aplicar la funcion de correlacion movil a todas las señales 
            ecg_analysis =ecg.ecg(signal=devII.values, sampling_rate=fs, show=False) #funcion de Biosppy
            bpm =np.mean(ecg_analysis['heart_rate']) #Solo se selecciona el parametro de la frecuencia instantanea 
            frecuencias_cardiacas_arr.append(bpm)
            #print(f"Para la señal {archivo_csv}, bpm: {frecuencia_cardiaca}")
 
            #####QRS#####
            out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
            r_peaks = out['rpeaks']
            window_samples = int(0.08 * fs)
            ##### Duración Onda P ########
            ventana = 200
            ventana_samples = int((ventana/1000)*fs)
            duracion_ondas_P = []
            QRS_durations = []  

            for r_peak in r_peaks:
                signal_window = signal[r_peak - window_samples:r_peak + window_samples]#Seleccionar la parte de la señal del pico R +- las muestras
                index_Q = np.argmin(signal_window[:window_samples])#Aquí esta buscando el valor mínimo de la parte izquieda de la señal que sería el inicio de la onda Q
                index_Q_global = r_peak - window_samples + index_Q 
                index_S = np.argmin(signal_window[window_samples:])
                index_S_global = r_peak - window_samples + window_samples + index_S 
                QRS_duration = index_S_global - index_Q_global
                QRS_durations.append(QRS_duration)


                ##### Duracion Onda P######
                inicio_ventana = max(0, r_peak - ventana_samples)
                fin_ventana = r_peak
                ventana_signal = signal[inicio_ventana:fin_ventana]
                p_peak_index = np.argmax(ventana_signal)+ inicio_ventana
                ventana_inicio = max(0, p_peak_index - ventana_samples // 4)
                ventana_fin = min(len(signal), p_peak_index + ventana_samples // 4)
                minimos_inicio, _ = find_peaks(-signal[ventana_inicio:p_peak_index])
                minimos_fin, _ =find_peaks(-signal[p_peak_index:ventana_fin])

                if minimos_inicio.size > 0 and minimos_fin.size > 0: 
                    inicio_onda_P = ventana_inicio + minimos_inicio[-1]
                    fin_onda_P = p_peak_index + minimos_fin[0]
                    duracion = (fin_onda_P-inicio_onda_P)/fs*1000
                    duracion_ondas_P.append(duracion)
        
            if duracion_ondas_P: 
                ondaP_ms = np.mean(duracion_ondas_P)
                #print(f"{duracion_onda_P_msec}")
            
            average_QRS_duration = np.mean(QRS_durations)
            qrs_ms = (average_QRS_duration / fs) * 1000
            #print(f"Duración promedio estimada del complejo QRS: {qrs_ms:.2f} ms")

            #####PR######
            intervalo_PR_durations = []
            for i in range(len(r_peaks) - 1):
                # Encontrar el índice del pico P más cercano al pico R actual
                indice_pico_P = out['rpeaks'][i-1]

                # Calcular la diferencia de tiempo entre el pico P y el pico R actual
                intervalo_PR_duration = (r_peaks[i] - indice_pico_P) / fs * 1000  # Duración en milisegundos
                intervalo_PR_durations.append(intervalo_PR_duration)

            promedio_intervalo_PR = np.mean(intervalo_PR_durations)
            PR_ms = abs(promedio_intervalo_PR)
            #print(f"Duración promedio del intervalo PR: {promedio_intervalo_PR:.2f} ms")
            
            if valores_mayores: 
                prom_corr = np.mean(valores_mayores)
                valores_mayores_arr.append(prom_corr)
            else:
                prom_corr =np.mean(valores_correlacion)
                valores_correlacion_arr.append(prom_corr)
                
            
            if contador % 100 ==0:
                print(f"Señal {contador}")

             ######## proporcion #########
            rK = len(valores_correlacion)
            beta = len(valores_mayores)
            p = beta/rK
            proporciones_arr.append(p)
            #print(f"Para la señal {archivo_csv}, la proporción es: {p}")

            ####### Valor critico ###### 
            n= narritmias
            #clasificar(clas_señales_arr, p, n, bpm, qrs_ms, ondaP_ms, PR_ms, zeta) #Calcula el valor del estadístico y clasificar
            q = 1 - p
            denominador = (p * q) / n
        
            if denominador == 0:
                c = 1
                clas_señales_arr.append(c)

            elif denominador != 0:
                z = (p - 0.0137882489819662595) / np.sqrt(denominador)
                if z >= (-1) * zeta and bpm >= 60 and bpm <= 100 and qrs_ms < 110 and ondaP_ms < 120 and PR_ms < 200:
                    c = 0
                    clas_señales_arr.append(c)

                else:
                    c = 1   
                    clas_señales_arr.append(c)

    VP = sum(clas_señales_arr)
    SENS = VP/narritmias
    

    return clas_señales_sanas, ESP, clas_señales_arr, SENS










