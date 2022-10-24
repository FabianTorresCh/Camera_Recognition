# INICIO DE CODIGO
import cv2 as BibOpenCv
import  numpy as BibNumpy


# --------------------------------------------------------------------------------------------------
# VARIABLES DE PROGRAMA
# LA MATRI DEBE SER UN VALOR IMPAR
Var_GaussMatrizSize = int(5) # VARIABLE DE GAUSS / A TRAVES DE UNA MATRIS INDICA EL TAMAÑO DEL DESENFOQUE PARA PRIMER METODO DE DESENFOQUE GAUSS
Var_KernelMatrizSize = int(7) # VARIABLE DE KERNEL / INDICA EL TAMAÑO DE UNA MATRIZ
Var_Simgma = 1 # VALOR POR DEFECTO 0
Var_CannyMin = 60 # VALOR MINIMO 0
Var_CannyMax = 100 # VALOR MAXIMO 255
Var_UmbralMin = 0 # VARIABLE UMBRALIZACION MINIMA
Var_UmbralMax = 255 # VARIABLE UMBRALIZACION MAXIMA
Var_AreaSize_A6= [677,480] # EN PIXELES
Var_AreaSize_A12= [320,240] # EN PIXELES

# DIAMETRO DE MONEDAS
Var_Coins1_Area = 7220.141
Var_Coins2_Area = 9088.529

# MODELACION NUMPY - A TRAVEZ DE UNA MATRIS SE INDICA EL TAMAÑAO DEL DESENFOQUE PARA EL SEGUNDO METODO DE DESENFOQUE MORFOLOGICO
ModelKernelMatriz = BibNumpy.ones((Var_KernelMatrizSize,Var_KernelMatrizSize),BibNumpy.uint8) # .unit8 : ENTEROS CON 8 BITS

"""NOTA: PARA UN MEJOR SEGUIMIENTO, IMPRIMIR LAS VARIABLES EN LA TERMINAL"""


# --------------------------------------------------------------------------------------------------
# FUNCIONES DE PROGRAMA

# ORDENAR PUNTOS
def DefPointOrder(VarPoint):
    # SE ENCUENTRAN DESORGANIZADO
    MatrixPointOder=BibNumpy.concatenate([VarPoint[0],VarPoint[1],VarPoint[2],VarPoint[3]]).tolist()
    Var_Y_CoordOrder = sorted(MatrixPointOder, key=lambda MatrixPointOder : MatrixPointOder[1])
    Var_X1_CoordOrder = Var_Y_CoordOrder[:2]
    Var_X1_CoordOrder = sorted(Var_X1_CoordOrder,key=lambda Var_X1_CoordOrder:Var_X1_CoordOrder[0])
    Var_X2_CoordOrder = Var_Y_CoordOrder[2:4]
    Var_X2_CoordOrder = sorted(Var_X2_CoordOrder,key=lambda Var_X2_CoordOrder:Var_X2_CoordOrder[0])
    # RETORNO DE DATOS
    return [Var_X1_CoordOrder[0],Var_X1_CoordOrder[1],Var_X2_CoordOrder[0],Var_X2_CoordOrder[1]]
    # NOTIFICACION
    print("FUNCTION EXECUTION POINT ORDER")


# ALINEAMIENTO DE IMAGEN
def DefImageAligned(VarImag, VarWidth, VarHeight):
    VarImagAligned = None
    # IMAGEN ESCALA GRIS
    Image_Gray = BibOpenCv.cvtColor(VarImag,BibOpenCv.COLOR_BGR2GRAY)
    # VARIABLE DE IMAGEN A PROCESAR
    Image_Proces = Image_Gray
    # IMAGEN UMBRALIZADA
    VarUmbralImage_Org, VarUmbralImage_Umb = BibOpenCv.threshold(Image_Proces, Var_UmbralMin, Var_UmbralMax,BibOpenCv.THRESH_BINARY)
    # VISUALIZACION DE IMAGEN UMBRALIZADA
    BibOpenCv.imshow("View_1",VarUmbralImage_Umb)
    # GENERACION DE CONTORNO PARA DEMARCACION DE AREA DE TRABAJO
    VarContourImage_ContourNone = BibOpenCv.findContours(VarUmbralImage_Umb, BibOpenCv.RETR_EXTERNAL, BibOpenCv.CHAIN_APPROX_NONE)[0] # [CHAIN_APPROX_NONE / CHAIN_APPROX_SIMPLE]
    # ORDENAR PUNTOS O MATRICES DE MENOR A MAYOR
    VarContourImage_ContourNone = sorted(VarContourImage_ContourNone, key=BibOpenCv.contourArea,reverse=True)[:1]
    # RECORRIDO DE LOS CONTORNOS
    for Record in VarContourImage_ContourNone:
        # GENERACION DE CURVAS
        GeometryEpsilon=0.01*BibOpenCv.arcLength(Record,True)
        # REDUCCION DE RUDIO EN GENERACION DE CURVA
        GeometryApprox = BibOpenCv.approxPolyDP(Record, GeometryEpsilon, True)
        if len(GeometryApprox) == 4:
            # METODO DE PINTADO DE AREA DE PROCESADO
            VarPointOrder = DefPointOrder(GeometryApprox)
            VarPointS1 = BibNumpy.float32(VarPointOrder)
            VarPointS2 = BibNumpy.float32([[0,0],[VarWidth,0],[VarHeight,0],[VarWidth,VarHeight]])
            # METODO PARA MANTENER PERSPECTIVA SI SE GIRA LA CAMARA
            VarPerspective = BibOpenCv.getPerspectiveTransform(VarPointS1, VarPointS2)
            # ALINEAR IMAGEN SI SE MODIFICA LA POSICION
            VarImagAligned = BibOpenCv.warpPerspective(VarImag,VarPerspective,(VarWidth,VarHeight))
    return VarImagAligned
    # NOTIFICACION
    print("FUNCTION EXECUTION IMAGE ALIGNED")

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

    # TAMAÑO AREA DE TRABAJO
    VarWorkArea =DefImageAligned(VarCamOnlineRead,VarWidth=Var_AreaSize_A6[0],VarHeight=Var_AreaSize_A6[1])

    # VERIFICACION DE IMAGEN
    if VarWorkArea is not None:
        Var_Point=[] # DE APERTURA DE ALMACENAMIENTO DE PUNTOS
        # CONVERSION A GRIS
        VarCamOnline_Gray = BibOpenCv.cvtColor(VarCamOnlineRead, BibOpenCv.COLOR_BGR2GRAY)
        #USO FILTRO GAUSS
        VarCamaraOnline_Gauss = BibOpenCv.GaussianBlur(VarCamOnline_Gray, (Var_GaussMatrizSize, Var_GaussMatrizSize), Var_Simgma)
        # USO FILTRO CANNY
        VarCamaraOnline_Canny = BibOpenCv.Canny(VarCamaraOnline_Gauss, Var_CannyMin, Var_CannyMax)
        # IMAGEN DE PROCESO
        VarCamOnline_Proces = VarCamaraOnline_Canny
        # UMBRALIZACION <<<<>>>> SUMATORIA DE PROCESOS DE UMBRALIZACION
        VarCamOnline_Original_2, VarCamOnline_Threshold_2 = BibOpenCv.threshold(VarCamOnline_Proces, Var_UmbralMin, Var_UmbralMax, BibOpenCv.THRESH_OTSU+BibOpenCv.THRESH_BINARY_INV)
        # VISTA DE UMBRALIZACION
        BibOpenCv.imshow("ImageUmbralizacion",VarCamOnline_Threshold_2)
        # GENERACION DE CONTORNO
        VarCamOnline_ContourSimplex_2 = BibOpenCv.findContours(VarCamOnline_Threshold_2, BibOpenCv.RETR_EXTERNAL, BibOpenCv.CHAIN_APPROX_SIMPLE)[0]
        # DIBUJO DE CONTORNO EN IMAGEN ORIGINAL
        # SE INDICA CON EL AREA DE TRABAJO
        BibOpenCv.drawContours(VarWorkArea, VarCamOnline_ContourSimplex_2, -1, (255, 0, 0), 3)

        # VARIABLE DE ALMACENAMIENTO DE CONTEO DE MONEDAS
        VarCoins_1 = 0.0
        VarCoins_2 = 0.0

        #CICLO DE CONTEO
        for Record_2 in VarCamOnline_ContourSimplex_2:
            VarGeometryCircle_Area = BibOpenCv.contourArea(Record_2)
            VarMoment = BibOpenCv.moments(Record_2)
            if(VarMoment["m00"]==0):
                VarMoment["m00"]=1.0
            VarX_Moment = int(VarMoment["m10"]/VarMoment["m00"])
            VarY_Moment = int(VarMoment["m01"]/VarMoment["m00"])

            # CONTEO DE MONEDAS SEGUN EL TAMAÑO
            # MONEDA 1 - AREA APROXIMADA
            if VarGeometryCircle_Area<9300 and VarGeometryCircle_Area >8800:
                VarFont = BibOpenCv.FONT_HERSHEY_SIMPLEX
                BibOpenCv.putText(VarWorkArea, "MONEDA 1", (VarX_Moment,VarY_Moment),VarFont,0.5,(0,0,255),2)
                VarCoins_1 = VarCoins_1+0.1

            # MONEDA 2 - AREA APROXIMADA
            if VarGeometryCircle_Area<7500 and VarGeometryCircle_Area >6900:
                VarFont = BibOpenCv.FONT_HERSHEY_SIMPLEX
                BibOpenCv.putText(VarWorkArea, "MONEDA 2", (VarX_Moment,VarY_Moment),VarFont,0.5,(0,255,0),2)
                VarCoins_1 = VarCoins_1+0.2

        # SUMATORIA DEL TOTAL DE MONEDAS
        VarSumaCoins = VarCoins_1+VarCoins_2
        # MENSAJE DE VERIFICACION
        print("LA SUMATORIA ES", round(VarSumaCoins,2))
        # VISTA DE PROCESO
        BibOpenCv.imshow("IMAGE_A6", VarWorkArea)
        BibOpenCv.imshow("CAM_UMBRAL", VarCamOnlineRead)

    VarCamOnlineRead


    # METIDO DE CIERRE DE PROGRAMA
    if BibOpenCv.waitKey(1)==ord("q"):
        break

VarCamOnlineOpen.release()
BibOpenCv.destroyAllWindows()


# --------------------------------------------------------------------------------------------------
#


# --------------------------------------------------------------------------------------------------
#

