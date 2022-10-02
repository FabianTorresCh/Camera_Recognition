# _______________________________________________________________________
# IMPORTACION DE OPENCV
import cv2

# _______________________________________________________________________
# IMPORTACION DE FOTOS
VarFotoTestOrg_2 = cv2.imread("../0_IMAGENES/FOTO_COLOR_2.jpg")

# _______________________________________________________________________
# VISUALIZACION DE IMAGENES ORIGINALES EN VENTANAS DE PYTHON
cv2.imshow('ViewFoto_2', VarFotoTestOrg_2)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# _______________________________________________________________________
# VISUALIZACION DE IMAGENES EN VENTANAS DE PYTHON ESCALA GRIS
VarFotoTestGris_2 = cv2.cvtColor(VarFotoTestOrg_2, cv2.COLOR_BGR2GRAY)

# _______________________________________________________________________
# VISUALIZACION DE IMAGENES GRIS EN VENTANAS DE PYTHON
cv2.imshow('ViewFoto_2_Gray', VarFotoTestGris_2)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# _______________________________________________________________________
# UMBRALIZACION
# UMBRALIZACION DE FOTO 2 CON DIFERENTE RANGO
VarUmbralFoto2Test_1_IMG_ORG,VarUmbralFoto2Test_1_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 100, 150, cv2.THRESH_BINARY)
VarUmbralFoto2Test_2_IMG_ORG,VarUmbralFoto2Test_2_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 150, 200, cv2.THRESH_BINARY)
VarUmbralFoto2Test_3_IMG_ORG,VarUmbralFoto2Test_3_IMG_UMB=cv2.threshold(VarFotoTestGris_2, 200, 255, cv2.THRESH_BINARY)

# _______________________________________________________________________
# VISUALIZACION DE IMAGENES UMBRALIZADA ESCALA BLANCO Y NEGRO (NO GRIS)
cv2.imshow('ViewFoto2_1_IMG_UMB', VarUmbralFoto2Test_1_IMG_UMB)
cv2.imshow('ViewFoto2_2_IMG_UMB', VarUmbralFoto2Test_2_IMG_UMB)
cv2.imshow('ViewFoto2_3_IMG_UMB', VarUmbralFoto2Test_3_IMG_UMB)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# _______________________________________________________________________
# GENERACION DE CONTORNOS
VarContoursFoto2Test_1_ContourNone, VarContoursFoto2Test_1_GerarquiaNone = cv2.findContours(VarUmbralFoto2Test_1_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
VarContoursFoto2Test_2_ContourNone, VarContoursFoto2Test_2_GerarquiaNone = cv2.findContours(VarUmbralFoto2Test_2_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
VarContoursFoto2Test_3_ContourNone, VarContoursFoto2Test_3_GerarquiaNone = cv2.findContours(VarUmbralFoto2Test_3_IMG_UMB,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

Foto1GroupCountorNone = [VarContoursFoto2Test_1_ContourNone,VarContoursFoto2Test_2_ContourNone,VarContoursFoto2Test_3_ContourNone]

# _______________________________________________________________________
# DIBUJO DE CONTORNOS
print("DIBUJO DE CONTORNO EN FOTOGRAFIAS 2 ORIGINALES")
cv2.drawContours(VarFotoTestOrg_2, Foto1GroupCountorNone[0] , -1, (255,0,0), 2)
cv2.drawContours(VarFotoTestOrg_2, Foto1GroupCountorNone[1] , -1, (0,255,0), 2)
cv2.drawContours(VarFotoTestOrg_2, Foto1GroupCountorNone[2] , -1, (0,0,255), 2)
cv2.imshow('ViewFoto_2_NONE', VarFotoTestOrg_2)
cv2.waitKey(0)
cv2.destroyAllWindows() # PRIMIENDO CUALQUIER BOTON, ELIMINA LA PRESENTACION DE LAS IMAGENES







