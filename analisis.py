
import numpy as np
import data 
import corr

umbrales = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
valor_z = [1.64, 1.28, 1.04]



for z in valor_z:
    especificidades = []
    sensibilidades = []

    for umbral in umbrales:
        clasificacion_sanas, ESP, clasificacion_arr, SENS = corr.procesar_database(data.señales_ecg_sanas, 382, data.señales_arrtimias, 463, z, umbral)
        especificidades.append(ESP)
        sensibilidades.append(SENS)
    np.savetxt(f'Sensibilidad_valor_crítico{z}.txt', sensibilidades)
    np.savetxt(f'Especificidad_valor_crítico{z}.txt', especificidades)



