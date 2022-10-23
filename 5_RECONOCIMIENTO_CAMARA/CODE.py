# IMPORTACION DE ELEMENTOS
import cv2 as OpenCv
import  numpy as BibNumpy


# --------------------------------------------------------------------------------------------------
# VARIABLES DE PROGRAMA
# LA MATRI DEBE SER UN VALOR IMPAR
Var_GaussMatrizSize = int(1) # VARIABLE DE GAUSS / A TRAVES DE UNA MATRIS INDICA EL TAMAÑO DEL DESENFOQUE PARA PRIMER METODO DE DESENFOQUE GAUSS
Var_KernelMatrizSize = int(7) # VARIABLE DE KERNEL / INDICA EL TAMAÑO DE UNA MATRIZ
Var_Simgma = 0 # VALOR POR DEFECTO 0
Var_CannyMin = 60 # VALOR MINIMO 0
Var_CannyMax = 100 # VALOR MAXIMO 255

# MODELACION NUMPY - A TRAVEZ DE UNA MATRIS SE INDICA EL TAMAÑAO DEL DESENFOQUE PARA EL SEGUNDO METODO DE DESENFOQUE MORFOLOGICO
ModelKernelMatriz = BibNumpy.ones((Var_KernelMatrizSize,Var_KernelMatrizSize),BibNumpy.uint8) # .unit8 : ENTEROS CON 8 BITS

"""NOTA: PARA UN MEJOR SEGUIMIENTO, IMPRIMIR LAS VARIABLES EN LA TERMINAL"""


# --------------------------------------------------------------------------------------------------
# VARIABLE DE CAPTURA DE VIDEO POR CAMARA
"""
SE INDICA EL NUMERO (0) O (1)
0 = CAMARAS INTERNAS, CONECTADAS EN EL PC
1 = CAMARAS EXTERNAS AL PC, COMO CAMARAS DE SEGURIDAD
"""
VarVideoCamara=OpenCv.VideoCapture(0)


# --------------------------------------------------------------------------------------------------
# VERIFICACION DE APERTURA Y CONEXION DE CAMARA
if not VarVideoCamara.isOpened():
    print("NO HAY CONEXION A CAMARA\nPROGRAMA CERRADO")
    exit()


# --------------------------------------------------------------------------------------------------
# CICLO PARA EJECUCION DE PROGRAMA
while True:

    # --------------------------------------------------------------------------------------------------
    # LECTURA Y ALMACENAMIENTO EN NUEVA VARIABLE
    TipoCamara,CamOnline = VarVideoCamara.read()

    # --------------------------------------------------------------------------------------------------
    # APLICACION DE FILTROS

    # 1. TRANSFORMACION A ESCALA GRIS
    CamOnline_Gray = OpenCv.cvtColor(CamOnline, OpenCv.COLOR_BGR2GRAY)

    # 2. USO DE FILTRO GAUSS
    CamaraOnline_Gauss = OpenCv.GaussianBlur(CamOnline_Gray, (Var_GaussMatrizSize, Var_GaussMatrizSize), Var_Simgma)

    # 3. USO DE FILTRO CANNY
    CamaraOnline_Canny = OpenCv.Canny(CamaraOnline_Gauss, Var_CannyMin, Var_CannyMax)

    # ASIGNACION DE VARIABLE GENERAL DE USO
    """PARA LA VARIABLE SE PUEDE CAMBIAR LOS EFECTOS VISUALES SEGUN LOS FILTROS QUE SE INGRESAN, SE PUEDE CAMBIAR POR LA VARIABLE [CamOnline_Gray] PARA UN MAYOR EFECTO EN LOS CONTORNOS"""
    CamOnline_Proces = CamaraOnline_Canny


    # --------------------------------------------------------------------------------------------------
    # UMBRALIZACION
    VarUmbralCamaraOnlineGris_R1_ORG, VarUmbralCamaraOnlineGris_R1_UMB = OpenCv.threshold(CamOnline_Proces, 100, 150,OpenCv.THRESH_BINARY)
    VarUmbralCamaraOnlineGris_R2_ORG, VarUmbralCamaraOnlineGris_R2_UMB = OpenCv.threshold(CamOnline_Proces, 150, 200,OpenCv.THRESH_BINARY)
    VarUmbralCamaraOnlineGris_R3_ORG, VarUmbralCamaraOnlineGris_R3_UMB = OpenCv.threshold(CamOnline_Proces, 200, 255,OpenCv.THRESH_BINARY)

    # GENERACION DE CONTORNO
    VarUmbralCamaraOnlineGris_R1_ContourNone, VarUmbralCamaraOnlineGris_R1_GerarquiaNone = OpenCv.findContours(VarUmbralCamaraOnlineGris_R1_UMB, OpenCv.RETR_LIST, OpenCv.CHAIN_APPROX_NONE)
    VarUmbralCamaraOnlineGris_R2_ContourNone, VarUmbralCamaraOnlineGris_R2_GerarquiaNone = OpenCv.findContours(VarUmbralCamaraOnlineGris_R2_UMB, OpenCv.RETR_LIST, OpenCv.CHAIN_APPROX_NONE)
    VarUmbralCamaraOnlineGris_R3_ContourNone, VarUmbralCamaraOnlineGris_R3_GerarquiaNone = OpenCv.findContours(VarUmbralCamaraOnlineGris_R3_UMB, OpenCv.RETR_LIST, OpenCv.CHAIN_APPROX_NONE)

    # DIBUJO DE CONTORNO EN IMAGEN ORIGINAL
    OpenCv.drawContours(CamOnline, VarUmbralCamaraOnlineGris_R1_ContourNone, -1, (255, 0, 0), 3)
    OpenCv.drawContours(CamOnline, VarUmbralCamaraOnlineGris_R2_ContourNone, -1, (0, 255, 0), 2)
    OpenCv.drawContours(CamOnline, VarUmbralCamaraOnlineGris_R3_ContourNone, -1, (0, 0, 255), 1)


    # --------------------------------------------------------------------------------------------------
    # VENTANA DE VISUALIZACION
    OpenCv.imshow("WEBCAM_ORG",CamOnline)
    OpenCv.imshow("WEBCAM_GRIS",CamOnline_Gray)

    # METIDO DE CIERRE DE PROGRAMA
    if OpenCv.waitKey(1)==ord("q"):
        break

# DETENER Y CERRAR TODOS LOS ELEMENTOS
VarVideoCamara.release()
OpenCv.destroyAllWindows()
