"""
RECONOCIMIENTO DE CONTORNOS
cv2.findCountours(umbral, mode, method) [1, 2, 3]
1. IMAGEN UMBRALIZADA: IMAGEN EN BLANCO Y NEGRO, SEPARADO DEL OBJETO DEL
ESPACIO DONDE SE ENCUENTRA
2. MODO: SALIDA DE DATOS, COMO LISTA, PROMEDIO, ETC
3. METODO DE ENCONTRAR LOS CONTORNOS LOS PRINCIPALES SON 2 (APROX_NONE Y APROX_SIMPLE )
    APROX_NONE: DEMARCA UN MUCHOS PUNTOS EL CONTORNO = MAS RECURSOS
    APROX_SIMPLE: INDICA EL CONTORNO DE MANERA SIMPLIFICADA PROMEDIANDO E IDENTIFICANDO CON POCOS NUMEROS = MENOS RECURSOS
"""
# OTRA DOCUMENTACION
# BUSCAR DOCUMENTACION SOBRE cvtColor OpenCv


# IMPORTACION DE OPENCV
import cv2

# ADICIONAL
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# _______________________________________________________________________
# IMPORTACION DE IMAGENES ORIGINALES A VARIABLES
VarImageTestOrg_1 = cv2.imread("../0_IMAGENES/IMAGEN_COLOR_1.jpg")
VarImageTestOrg_2 = cv2.imread("../0_IMAGENES/IMAGEN_COLOR_2.jpg")
VarImageTestOrg_3 = cv2.imread("../0_IMAGENES/IMAGEN_COLOR_3.jpg")
VarFotoTestOrg_1 = cv2.imread("../0_IMAGENES/FOTO_COLOR_1.jpg")
VarFotoTestOrg_2 = cv2.imread("../0_IMAGENES/FOTO_COLOR_2.jpg")
VarFotoTestOrg_3 = cv2.imread("../0_IMAGENES/FOTO_COLOR_3.jpg")
# _______________________________________________________________________

# _______________________________________________________________________
# VISUALIZACION DE IMAGEN EN VENTANAS DE PYTHON
print("IMAGEN 1")
cv2.imshow('ViewImage_1', VarImageTestOrg_1)
print("IMAGEN 2")
cv2.imshow('ViewImage_2', VarImageTestOrg_2)
print("IMAGEN 3")
cv2.imshow('ViewImage_3', VarImageTestOrg_3)

# FOTOS
cv2.imshow('ViewFoto_1', VarFotoTestOrg_1)
cv2.imshow('ViewFoto_2', VarFotoTestOrg_2)
cv2.imshow('ViewFoto_3', VarFotoTestOrg_3)

cv2.waitKey(1000)
cv2.destroyAllWindows()
# _______________________________________________________________________

# _______________________________________________________________________
# IMPORTACION DE IMAGENES ORIGINALES CON ESCALA DE GRISES SEGUN FUNCION cv2.cvtColor()
print("CONVERSION DE IMAGENES A ESCALA GRIS")
VarImageTestGris_1 = cv2.cvtColor(VarImageTestOrg_1, cv2.COLOR_BGR2GRAY)
VarImageTestGris_2 = cv2.cvtColor(VarImageTestOrg_2, cv2.COLOR_BGR2GRAY)
VarImageTestGris_3 = cv2.cvtColor(VarImageTestOrg_3, cv2.COLOR_BGR2GRAY)

#FOTOS
VarFotoTestGris_1 = cv2.cvtColor(VarFotoTestOrg_1, cv2.COLOR_BGR2GRAY)
VarFotoTestGris_2 = cv2.cvtColor(VarFotoTestOrg_2, cv2.COLOR_BGR2GRAY)
VarFotoTestGris_3 = cv2.cvtColor(VarFotoTestOrg_3, cv2.COLOR_BGR2GRAY)

# _______________________________________________________________________

# _______________________________________________________________________
# VISUALIZACION DE IMAGEN EN VENTANAS DE PYTHON
print("IMAGEN 1 - GRIS")
cv2.imshow('ViewImage_1_Gray', VarImageTestGris_1)
cv2.waitKey(500)
cv2.destroyAllWindows()
print("IMAGEN 2 - GRIS")
cv2.imshow('ViewImage_2_Gray', VarImageTestGris_2)
cv2.waitKey(500)
cv2.destroyAllWindows()
print("IMAGEN 3 - GRIS")
cv2.imshow('ViewImage_3_Gray', VarImageTestGris_3)
cv2.waitKey(500)
cv2.destroyAllWindows()

cv2.imshow('ViewFoto_1_Gray', VarFotoTestGris_1)
cv2.waitKey(500)
cv2.destroyAllWindows()
cv2.imshow('ViewFoto_2_Gray', VarFotoTestGris_2)
cv2.waitKey(500)
cv2.destroyAllWindows()
cv2.imshow('ViewFoto_3_Gray', VarFotoTestGris_3)
cv2.waitKey(500)
cv2.destroyAllWindows()

# _______________________________________________________________________

# _______________________________________________________________________
# PRIMERO LA IMAGEN DEBE ESTAR EN ESCALA DE GRICES
# AISLAMIENTO DE LA IMAGEN DE SU ENTORNO
# IDENTIFICAR EL TIPO DE UMBRALIZACION (ORIGINAL IMAGEN / THRESH_BINARY / BINARY_INV / TRUNC / TOZERO / TOZERO_INV))
# SE PUEDEN COMBINAR LA UMBRALIZACION
# threshold(IMAGEN EN GRIS, UMBRAL MIN, UMBRAL MAX, TIPO DE UMBRAL)
# !!!!!!!!!! EL METODO DEVUELVE DOS SALIDAS (IMAGEN QUE SE UTILIZO,IMAGEN UMBRALIZADA)
VarUmbralImageTest_1_IMG_ORG,VarUmbralImageTest_1_IMG_UMB=cv2.threshold(VarImageTestGris_1, 100, 255, cv2.THRESH_BINARY)
VarUmbralImageTest_2_IMG_ORG,VarUmbralImageTest_2_IMG_UMB=cv2.threshold(VarImageTestGris_2, 100, 255, cv2.THRESH_BINARY)
VarUmbralImageTest_3_IMG_ORG,VarUmbralImageTest_3_IMG_UMB=cv2.threshold(VarImageTestGris_3, 100, 255, cv2.THRESH_BINARY)

# UMBRALIZACION DE FOTOS
# SE SELECCIONA UNA MISMA FOTO CON DIFERENTES RANGOS DE UMBRAL
VarUmbralFotoTest_1_IMG_ORG,VarUmbralFotoTest_1_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 0, 85, cv2.THRESH_BINARY)
VarUmbralFotoTest_2_IMG_ORG,VarUmbralFotoTest_2_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 85, 170, cv2.THRESH_BINARY)
VarUmbralFotoTest_3_IMG_ORG,VarUmbralFotoTest_3_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 170, 255, cv2.THRESH_BINARY)



print("NOMBRE DE VARIABLE: ",namestr(VarUmbralImageTest_1_IMG_UMB,globals())," / ",type(VarUmbralImageTest_1_IMG_UMB))
print("UMBRALIZACION DE IMAGEN: ",VarUmbralImageTest_1_IMG_UMB)

# _______________________________________________________________________
# VISUALIZACION DE IMAGEN EN VENTANAS DE PYTHON UMBRALIZADA ORIGINAL
print("IMAGEN 1 - UMBRAL ORG")
cv2.imshow('ViewImage_1_IMG_ORG', VarUmbralImageTest_1_IMG_ORG)
print("IMAGEN 2 - UMBRAL ORG")
cv2.imshow('ViewImage_2_IMG_ORG', VarUmbralImageTest_2_IMG_ORG)
print("IMAGEN 3 - UMBRAL ORG")
cv2.imshow('ViewImage_3_IMG_ORG', VarUmbralImageTest_3_IMG_ORG)

cv2.waitKey(1000)
cv2.destroyAllWindows()

# IMAGEN UMBRALIZADA
print("IMAGEN 1 - UMBRAL")
cv2.imshow('ViewImage_1_IMG_UMB', VarUmbralImageTest_1_IMG_UMB)
print("IMAGEN 2 - UMBRAL")
cv2.imshow('ViewImage_2_IMG_UMB', VarUmbralImageTest_2_IMG_UMB)
print("IMAGEN 3 - UMBRAL")
cv2.imshow('ViewImage_3_IMG_UMB', VarUmbralImageTest_3_IMG_UMB)

# FOTO UMBRALIZADA
cv2.imshow('ViewFoto_1_IMG_UMB', VarUmbralFotoTest_1_IMG_UMB)
cv2.imshow('ViewFoto_2_IMG_UMB', VarUmbralFotoTest_2_IMG_UMB)
cv2.imshow('ViewFoto_3_IMG_UMB', VarUmbralFotoTest_3_IMG_UMB)


cv2.waitKey(0)
cv2.destroyAllWindows()


