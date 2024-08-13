import cv2
import face_recognition
import os
from _datetime import datetime
import numpy as np

#step1: load ảnh từ kho ảnh nhận dạng
path = "pic2"
images = []
classNames = []
myList = os.listdir(path)
print(myList) #'Donal Trump.jpg', 'elon musk .jpg', 'Joker.jpg', 'tokuda.jpg'

for cl in myList:
    print(cl)
    curImg = cv2.imread(f"{path}/{cl}") #pic2/Donal Trump.jpg
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) #tách tên, vd tách 2 vế: Donal Trump([0])  ,jpg([1])

print(len(images))
print(classNames)

#step2: encoding
def maHoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnow = maHoa(images)
print("Mã hóa thành công!")
print(len(encodeListKnow))


def thamDu(name):
    with open("thamdu.csv", "r+") as f:
        myDataList = f.readline()
        nameList =[]

        for line in myDataList:
            entry = line.split(',') #tách theo dấu
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name}, {dtString}")



#step3: Khởi động webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() #ret: trả về true false, trả về true: khi trả về đc ảnh. Frame: số khung hình
    framS = cv2.resize(frame, (0,0), None, fx=0.5, fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGRA2BGR)

    #xác định vị trí khuôn mặt trên Cam và encode
    facecurFrame = face_recognition.face_locations(framS)
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame): #lấy từng khuôn mặt vào vị trí khuôn mặt hiện tại
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) #đẩy về index của faceDis min

        if faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            thamDu(name)
        else:
            name = "Unknown"

        #In lên frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x2,y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Nhan dang khuon mat", frame)
    if cv2.waitKey(1) == ord('q'):  #độ trễ 1/1000s, bấm q sẽ thoát
        break
cap.release() #giải phóng camera
cv2.destroyAllWindows() #thoát all cửa sổ


