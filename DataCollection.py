import cv2

video=cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(r'C:\Users\ANUJ GUPTA\OneDrive\Desktop\Face Recognition\haarcascade_forntalface_default.xml')

id = input("Enter your ID : ")
# id = int(id)
count = 0

while True:
    ret,frame=video.read()

    if(frame is not None):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    
    for (x,y,w,h) in faces:
        count = count+1
        cv2.imwrite('DataImages/User.'+str(id)+"."+str(count)+".jpg",gray[y:y+h , x:x+w])
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , color=(50,50,255) ,thickness= 1)

    cv2.imshow("Frame",frame)

    k=cv2.waitKey(1) ## if 0 then any key is pressed it will exit

    if count>500:
        break

    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done.....................")