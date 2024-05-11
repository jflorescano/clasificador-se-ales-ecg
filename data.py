import glob 
import os

carpeta = "d:/TESIS_JOANA/FEMALE_SR_NONE_40_60/*.csv"
señales_ecg_sanas = glob.glob(carpeta) #Cargar todas las señales que se encuentren en la carpeta, este es el database
carpeta2 = "d:/TESIS_JOANA/ENFERMAS/*.csv"
señales_arrtimias = glob.glob(carpeta2)
carpeta3 = "d:/TESIS_JOANA/MUESTRA_SANAS/*.csv"
muestra_sanas = glob.glob(carpeta3)
carpeta4 = "d:/TESIS_JOANA/MUESTRA_ARRITMIAS/*.csv"
muestra_arritmias = glob.glob(carpeta4)
