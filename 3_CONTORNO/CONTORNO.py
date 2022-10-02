"""
GENERACION DE CONTORNO DE ELEMENTOS

1. OBTENCION DE LA IMAGEN ORIGINAL EN ESCALA DE COLORES
2. GENERACION DE LA IMAGEN EN ESCALA DE GRISES A LA IMAGEN ORIGINAL
3. DE LA IMAGEN EN ESCALA DE GRISES SE GENERA IMAGEN UMBRALIZADA SEGUN RANGOS, EN BLANCO Y NEGRO
4. GENERACION DE CONTORNOS

"""
# _______________________________________________________________________
# IMPORTACION DE OPENCV
import cv2


# _______________________________________________________________________
# IMPORTACION DE FOTOS
VarImageTestOrg_1 = cv2.imread("../0_IMAGENES/IMAGEN_COLOR_1.jpg")
VarFotoTestOrg_1 = cv2.imread("../0_IMAGENES/FOTO_COLOR_1.jpg")
VarFotoTestOrg_2 = cv2.imread("../0_IMAGENES/FOTO_COLOR_2.jpg")


# _______________________________________________________________________
# VISUALIZACION DE IMAGENES ORIGINALES EN VENTANAS DE PYTHON
cv2.imshow('ViewImage_1', VarImageTestOrg_1)
cv2.imshow('ViewFoto_1', VarFotoTestOrg_1)
cv2.imshow('ViewFoto_2', VarFotoTestOrg_2)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# _______________________________________________________________________
# VISUALIZACION DE IMAGENES EN VENTANAS DE PYTHON
VarImageTestGris_1 = cv2.cvtColor(VarImageTestOrg_1, cv2.COLOR_BGR2GRAY)
VarFotoTestGris_1 = cv2.cvtColor(VarFotoTestOrg_1, cv2.COLOR_BGR2GRAY)
VarFotoTestGris_2 = cv2.cvtColor(VarFotoTestOrg_2, cv2.COLOR_BGR2GRAY)


# _______________________________________________________________________
# VISUALIZACION DE IMAGENES GRIS EN VENTANAS DE PYTHON
cv2.imshow('ViewImage_1_Gray', VarImageTestGris_1)
cv2.imshow('ViewFoto_1_Gray', VarFotoTestGris_1)
cv2.imshow('ViewFoto_2_Gray', VarFotoTestGris_2)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# _______________________________________________________________________
# UMBRALIZACION
# UMBRALIZACION DE IMAGEN SIMPLE
VarUmbralImageTest_1_IMG_ORG,VarUmbralImageTest_1_IMG_UMB=cv2.threshold(VarImageTestGris_1, 100, 255, cv2.THRESH_BINARY)
# UMBRALIZACION DE FOTO 1 CON UNICO RANGO RANGO
VarUmbralFoto1Test_1_IMG_ORG,VarUmbralFoto1Test_1_IMG_UMB=cv2.threshold(VarFotoTestGris_1, 100, 150, cv2.THRESH_BINARY)
VarUmbralFoto1Test_2_IMG_ORG,VarUmbralFoto1Test_2_IMG_UMB=cv2.threshold(VarFotoTestGris_1, 150, 200, cv2.THRESH_BINARY)
VarUmbralFoto1Test_3_IMG_ORG,VarUmbralFoto1Test_3_IMG_UMB=cv2.threshold(VarFotoTestGris_1, 200, 255, cv2.THRESH_BINARY)
# UMBRALIZACION DE FOTO 2 CON DIFERENTE RANGO
VarUmbralFoto2Test_1_IMG_ORG,VarUmbralFoto2Test_1_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 100, 150, cv2.THRESH_BINARY)
VarUmbralFoto2Test_2_IMG_ORG,VarUmbralFoto2Test_2_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 150, 200, cv2.THRESH_BINARY)
VarUmbralFoto2Test_3_IMG_ORG,VarUmbralFoto2Test_3_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 200, 255, cv2.THRESH_BINARY)


# _______________________________________________________________________
# VISUALIZACION DE IMAGENES UMBRALIZADA ESCALA BLANCO Y NEGRO (NO GRIS)
cv2.imshow('ViewImage_1_IMG_UMB', VarUmbralImageTest_1_IMG_UMB)
cv2.imshow('ViewFoto1_1_IMG_UMB', VarUmbralFoto1Test_1_IMG_UMB)
cv2.imshow('ViewFoto1_2_IMG_UMB', VarUmbralFoto1Test_2_IMG_UMB)
cv2.imshow('ViewFoto1_3_IMG_UMB', VarUmbralFoto1Test_3_IMG_UMB)
cv2.imshow('ViewFoto2_1_IMG_UMB', VarUmbralFoto2Test_1_IMG_UMB)
cv2.imshow('ViewFoto2_2_IMG_UMB', VarUmbralFoto2Test_2_IMG_UMB)
cv2.imshow('ViewFoto2_3_IMG_UMB', VarUmbralFoto2Test_3_IMG_UMB)
cv2.waitKey(1000)
cv2.destroyAllWindows() # PRIMIENDO CUALQUIER BOTON, ELIMINA LA PRESENTACION DE LAS IMAGENES


# _______________________________________________________________________
# GENERACION DE CONTORNOS
# CODIGO: cv.findContours(Imagen.jpg, modo de recuperacion, metodo de aproximacion de contorno)
# (cv.CHAIN_APPROX_SIMPLE / cv.CHAIN_APPROX_NONE)
# DEVUELVE 2 TIPOS DE RESULTADOS ( CONTORNO / GERARQUIA DE CONTORNOS, UNO DENTROS DE OTROS)
# SE DEBE UMBRALIZACION LA IMAGEN

# GENERACION DE CONTORNO SOBRE IMAGENES
print("GENERACION DE CONTORNO")
# BORDE CON PUNTOS SIMPLES
VarContoursImageTest_1_ContourSimplex, VarContoursImageTest_1_GerarquiaSimplex = cv2.findContours(VarUmbralImageTest_1_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print("IMAGEN 1 - CONTORNO 1: ",VarContoursImageTest_1_ContourSimplex)
# BORDE CON PUNTOS MULTIPLES
VarContoursImageTest_1_ContourNone, VarContoursImageTest_1_GerarquiaNone = cv2.findContours(VarUmbralImageTest_1_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

# GENERACION DE CONTORNOS SOBRE FOTOGRAFIA CON DIFERENTE RANGO DE UMBRALIZACION

# CONTORNO DE FOTOGRAFIAS CON CONTORNO SIMPLE
VarContoursFoto1Test_1_ContourSimplex, VarContoursFoto1Test_1_GerarquiaSimplex = cv2.findContours(VarUmbralFoto1Test_1_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
VarContoursFoto1Test_2_ContourSimplex, VarContoursFoto1Test_2_GerarquiaSimplex = cv2.findContours(VarUmbralFoto1Test_2_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
VarContoursFoto1Test_3_ContourSimplex, VarContoursFoto1Test_3_GerarquiaSimplex = cv2.findContours(VarUmbralFoto1Test_3_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# CONTORNO DE FOTOGRAFIAS CON CONTORNO MULTIPLE
VarContoursFoto2Test_1_ContourNone, VarContoursFoto2Test_1_GerarquiaNone = cv2.findContours(VarUmbralFoto2Test_1_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
VarContoursFoto2Test_2_ContourNone, VarContoursFoto2Test_2_GerarquiaNone = cv2.findContours(VarUmbralFoto2Test_2_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
VarContoursFoto2Test_3_ContourNone, VarContoursFoto2Test_3_GerarquiaNone = cv2.findContours(VarUmbralFoto2Test_3_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# AGRUPACION DE RESULTADOS DE CONTORNO IMAGEN
ImageGruopCountor = [
    VarContoursImageTest_1_ContourSimplex,
    VarContoursImageTest_1_ContourNone
]

# AGRUPACION DE RESULTADOS DE COTORNOS
Foto1GroupCountorSimplex = [
    VarContoursFoto1Test_1_ContourSimplex,
    VarContoursFoto1Test_2_ContourSimplex,
    VarContoursFoto1Test_3_ContourSimplex
]
Foto1GroupCountorNone = [
    VarContoursFoto2Test_1_ContourNone,
    VarContoursFoto2Test_2_ContourNone,
    VarContoursFoto2Test_3_ContourNone
]


# _______________________________________________________________________
# DIBUJO DE CONTORNOS
# TODOS LOS CONTORNOS SE ESCRIBE (-1)
# [[[    cv2.drawContours(IMAGEN ORIGINAL PARA DIBUJO DEL CONTORNO, CONTORNO, INDICE DE CONTORNOS, COLOR DE DIBUJO, GROSOR DE CONTORNO)   ]]]
print("DIBUJO DE CONTORNO EN IMAGEN ORIGINAL")
# SOBRE LA IMAGEN ORIGINAL SE DIBUJA EL CONTORNO, ASI QUE SE DEBE LLAMAR OTRA VEZ LA LA IMAGEN
cv2.drawContours(VarImageTestOrg_1, ImageGruopCountor[0] , -1, (255,0,0), 3)
cv2.drawContours(VarImageTestOrg_1, ImageGruopCountor[1] , -1, (0,255,0), 3)

print("DIBUJO DE CONTORNO EN FOTOGRAFIAS 1 ORIGINALES")
cv2.drawContours(VarFotoTestOrg_1, Foto1GroupCountorSimplex[0] , -1, (255,0,0), 3)
cv2.drawContours(VarFotoTestOrg_1, Foto1GroupCountorSimplex[1] , -1, (0,255,0), 3)
cv2.drawContours(VarFotoTestOrg_1, Foto1GroupCountorSimplex[2] , -1, (0,0,255), 3)

print("DIBUJO DE CONTORNO EN FOTOGRAFIAS 2 ORIGINALES")
cv2.drawContours(VarFotoTestOrg_2, Foto1GroupCountorNone[0] , -1, (255,0,0), 2)
cv2.drawContours(VarFotoTestOrg_2, Foto1GroupCountorNone[1] , -1, (0,255,0), 2)
cv2.drawContours(VarFotoTestOrg_2, Foto1GroupCountorNone[2] , -1, (0,0,255), 2)

cv2.imshow('ViewImage_1', VarImageTestOrg_1)
cv2.imshow('ViewFoto_1_SIMPLEX', VarFotoTestOrg_1)
cv2.imshow('ViewFoto_2_NONE', VarFotoTestOrg_2)
cv2.waitKey(0)
cv2.destroyAllWindows() # PRIMIENDO CUALQUIER BOTON, ELIMINA LA PRESENTACION DE LAS IMAGENES







