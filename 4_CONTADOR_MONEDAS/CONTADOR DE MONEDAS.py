"""DOCUMENTACION
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
"""
#IMPORTACION DE BIBLIOTECAS
import cv2
import numpy as BibNumpy

var = BibNumpy.ones

# ----------------------------------------------------------------------------------------------------------------------------
# VARIABLES
GaussMatrizSize = int(3) # VARIABLE DE GAUSS
KernelMatrizSize = int(3) # VARIABLE DE KERNEL
Simgma = 0 # VALOR POR DEFECTO 0
CannyMin = 60 # VALOR MINIMO 0
CannyMax = 255 # VALOR MAXIMO 255

# MODELACION NUMPY
ModelKernelMatriz = BibNumpy.ones((KernelMatrizSize,KernelMatrizSize),BibNumpy.uint8) # .unit8 : ENTEROS CON 8 BITS




# ----------------------------------------------------------------------------------------------------------------------------
# FUNCION DE VISUALIZACION
def DefVisualizacion(NameWindows, VarImageSee, PositionX, PositionY, TimeSee):
    cv2.imshow(NameWindows, VarImageSee) # INDICADOR DE VENTANA
    cv2.moveWindow(NameWindows, int(PositionX), int(PositionY)) # POSICION DE VENTANA
    #cv2.resizeWindow(NameWindows, 750, 750) # INDICA EL TAMAÑO DE LA VENTANA, PERO EL CONTENIDO NO ES DINAMICO AL TAMAÑO, NO SE AJUSTA
    cv2.waitKey(int(TimeSee)) # TIEMPO DE APERTURA DE VENTANA (0 PERMANENTE / 1 VIDEOS) (VALOR EN MILESIMAS DE SEGUNDO)
    cv2.destroyAllWindows() # METODO DE CIERRE - AL OPRIMIR CUALQUIER LETRA



# ----------------------------------------------------------------------------------------------------------------------------
# IMPORTACION DE IMAGEN
VarImage_Org = cv2.imread("../0_IMAGENES/MONEDAS_3.jpg") # IMPORTACION DE IMAGEN

# FUNCION DE VISUALISACION
DefVisualizacion("ViewFoto2_1_IMG_ORG",VarImage_Org,0,0,1000)

# ----------------------------------------------------------------------------------------------------------------------------
# IMAGEN ORIGNAL A ESCALA DE GRICES
VarImage_Gris = cv2.cvtColor(VarImage_Org, cv2.COLOR_BGR2GRAY)

# FUNCION DE VISUALISACION
DefVisualizacion("ViewFoto2_1_IMG_GRIS",VarImage_Gris,0,0,1000)

# ----------------------------------------------------------------------------------------------------------------------------
# GAUSS - USO DE HERRAMIENTA GAUSS PARA DIFUMINAR O DESENFOQUE IMAGENES
# LOS VALORES KSIZE INDICAN EL TAMAÑO DE LA MATRIZ, UN VALOR MUY ALTO, GENERA UNA IMAGEN MUY BORROSA
VarImage_Desenfoque = cv2.GaussianBlur(VarImage_Gris, (GaussMatrizSize, GaussMatrizSize), Simgma)

# FUNCION DE VISUALISACION
DefVisualizacion("ViewFoto2_1_IMG_GRIS_(GAUSS)",VarImage_Desenfoque,0,0,1000)

# ----------------------------------------------------------------------------------------------------------------------------
# CANNY - REDUCCION DE RUIDO DE IMAGENES (IMAGEN , RANGO MINIMO, RANGO MAXIMO)
VarImage_Canny=cv2.Canny(VarImage_Desenfoque,CannyMin,CannyMax)

# FUNCION DE VISUALISACION
DefVisualizacion("ViewFoto2_1_IMG_GRIS_(GAUSS+CANNY)",VarImage_Canny,0,0,1000)

# ----------------------------------------------------------------------------------------------------------------------------
# MORFOLOGIA - LIMPIESA DE IMAGEN DE LOS PROCESOS DE ESCALA GRICES + GAUSS + CANNY
# TRANSFORMACION MORFOLOGICA (EROSION, DILATACION, APERTURA, CLAUSURA)
# OBSERVAR DOCUMENTACION morphologyEx
VarImage_MorphologyClosed=cv2.morphologyEx(VarImage_Canny, cv2.MORPH_CLOSE, ModelKernelMatriz)

# FUNCION DE VISUALISACION
DefVisualizacion("ViewFoto2_1_IMG_GRIS_(GAUSS+CANNY+Morphology)",VarImage_MorphologyClosed,0,0,1000)

# ----------------------------------------------------------------------------------------------------------------------------
# GENERACION DE CONTORNO
VarContoursImage_Test_1_ContourNone, VarContoursImage_Test_1_GerarquiaNone = cv2.findContours(VarImage_MorphologyClosed.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)




cv2.drawContours(VarImage_Org, VarContoursImage_Test_1_ContourNone, -1, (255,0,0), 2)
DefVisualizacion("ViewFoto2_1_IMG_ORG",VarImage_Org,0,0,1000)













# MODELADO DE GAUS
"""OBJETIVO: SUAVIASR LOS LIMITES DE LAS IMAGENES DIFUMINANDO LA IMAGEN Y AGREGANDO MAS PIXELES, 
PERMITIENDO UNA MEJOR GENERACION DE CONTORNOS POR IMAGENES DE BAJA CALIDAD O RUIDO, ELIMINACION DE RUIDO DE LA IMAGEN
"""

print("FIN DE PROCESOS")