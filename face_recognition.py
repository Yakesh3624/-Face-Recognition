import cv2,os,numpy,imutils

images = []
labels = []
names ={}
id=0
dataset = "face_dataset"
for (subdirs,dirs,files) in os.walk(dataset):
    for subdir in dirs :
        names[id] = subdir
        subjectpath = os.path.join(dataset,subdir)
        for file in os.listdir(subjectpath):
            path = os.path.join(subjectpath,file)
            images.append(cv2.imread(path,0))
            labels.append(id)
        id +=1

model = cv2.face.LBPHFaceRecognizer_create()
model= cv2.face.FisherFaceRecognizer_create()

images = numpy.array(images)
labels = numpy.array(labels)

model.train(images,labels)

haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

while True:
    img = cam.read()[1]
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = haar.detectMultiScale(gray,1.3,4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        face = gray[y:y+h,x:x+w]
        face = imutils.resize(face,130,100)

        res = model.predict(face)
        
        cv2.putText(img,names[res[0]],(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
        
    cv2.imshow("",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
cam.release()
