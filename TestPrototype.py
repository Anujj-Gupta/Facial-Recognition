import cv2

video=cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(r'C:\Users\ANUJ GUPTA\OneDrive\Desktop\Face Recognition\haarcascade_forntalface_default.xml')

reco = cv2.face.LBPHFaceRecognizer_create()
reco.read("Trainer.yml")

name_list = ["","Anuj","Anmol","Aman"]

while True:
    ret,frame=video.read()

    if(frame is not None):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    
    for (x,y,w,h) in faces:
        serial, conf = reco.predict(gray[y:y+h, x:x+w])
        if conf<100:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, name_list[serial], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, "Unknown", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("Frame",frame)

    k=cv2.waitKey(1) ## if 0 then any key is pressed it will exit

    if k==ord('e'):
        break

video.release()
cv2.destroyAllWindows()
print("Testing Done.....................")