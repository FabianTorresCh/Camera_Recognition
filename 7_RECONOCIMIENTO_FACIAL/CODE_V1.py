# INICIO DE CODIGO

# IMPORTACION DE BIBLIOTECAS
import cv2 as BibOpenCv
import  numpy as BibNumpy

# IMPORTACION DE HAARCASCADE DE RUIDO (ELIMINA LOS CONTORNOS DE ELEMENTOS AGENOS A LA BUSQUEDA)
Haarcascades_Face = BibOpenCv.CascadeClassifier("../0_OPENCV/data/haarcascades/haarcascade_frontalface_default.xml")
print("VARIABLE HAARCASCADE_FACE: ",Haarcascades_Face)

# --------------------------------------------------------------------------------------------------
# VARIABLES DE PROGRAMA
Var_FactorScale = 1.3 # FACTOR PARA REDUCIR PIXELES Y RANGO / EQUIVALE A PORCENTAJE
Var_MinNeighbors = 5 # DEPENDE DEL AMBIENTE DEL VIDEO

# --------------------------------------------------------------------------------------------------
# CAPTURA DE VIDEO
VarCamOnlineOpen = BibOpenCv.VideoCapture(1)

# --------------------------------------------------------------------------------------------------
# CICLO PARA EJECUCION DE PROGRAMA
while True:

    # LECTURA Y ALMACENAMIENTO EN NUEVA VARIABLE
    CamType, VarCamOnlineRead = VarCamOnlineOpen.read()

    # VERIFICACION DE APERTURA Y CONEXION DE CAMARA
    if CamType == False:
        print("NO HAY CONEXION EN CAMARA")
        break

    # CONVERSION A GRIS
    VarCamOnline_Gray = BibOpenCv.cvtColor(VarCamOnlineRead, BibOpenCv.COLOR_BGR2GRAY)

    # VARIABLE DE CARAS
    # PORCENTAGE DE ESCALA, DEBERA REALIZAR TESTEO
    VarFace = Haarcascades_Face.detectMultiScale(VarCamOnline_Gray, Var_FactorScale, Var_MinNeighbors)

    # RECORIDO - BUQUEDA DE PIXEL
    for (Var_X, Var_Y, Var_X_Corner, Var_Y_Corner) in VarFace:
        BibOpenCv.rectangle(VarCamOnlineRead, (Var_X,Var_Y), (Var_X+Var_X_Corner, Var_Y+Var_Y_Corner), (255,0,0), 2)

    BibOpenCv.imshow("Face",VarCamOnlineRead)

    # METIDO DE CIERRE DE PROGRAMA
    if BibOpenCv.waitKey(1)==ord("q"):
        break

# CIERRE DE ELEMENTOS
VarCamOnlineOpen.release()
BibOpenCv.destroyAllWindows()




