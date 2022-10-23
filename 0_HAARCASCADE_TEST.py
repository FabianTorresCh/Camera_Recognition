import  cv2

VarFaceCascade = cv2.CascadeClassifier("0_HAARCASCADE/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)


while True:
    _,Img = cap.read()
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    Faces = VarFaceCascade.detectMultiScale(gray,1.1,4)



    for(x,y,w,h) in Faces:
        cv2.rectangle(Img,(x,y), (x+w,y+h),(255,0,0),4)
    cv2.imshow("Img",Img)
    k = cv2.waitKey(30)
    if k== 27:
        break
cap.release()

