import cv2
import time  
import mediapipe as mp
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cTime=0
pTime=0
mpHands= mp.solutions.hands
hands= mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

detector=htm.handdetector()


while True :
    success ,img=cap.read()
    img=detector.findhands(img)
    lmlist=detector.findPosition(img,draw=False)
    fingers=[]
    # print(lmlist)
    
    if len(lmlist)!=0:
        if lmlist[8][2]<lmlist[6][2]:
            fingers.append("Index")
        if lmlist[12][2]<lmlist[10][2]:
            fingers.append("Middle")
        if lmlist[16][2]<lmlist[14][2]:
            fingers.append("Ring")
        if lmlist[20][2]<lmlist[17][2]:
            fingers.append("Lady")

        if len(fingers)!=0:
             cv2.putText(img,str(fingers),(10,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)   
        else:
            cv2.putText(img,str("CLOSED"),(10,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

    cv2.putText(img,str(f"Fingers are :{int(len(fingers))}"),(10,140),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)

    if len(lmlist)==0:
        cv2.putText(img,str("NOT FOUND"),(10,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)

        
        
        
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c =img.shape
                cx ,cy =int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                cv2.circle(img,(cx,cy),10,(255,255,38),cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)\
    
    cv2.imshow("Video", img)

    if not success:
        print("error")
        break



    if cv2.waitKey(1)&0xFF==ord('q') :
        print("quitting...")
        break

cap.release()
cv2.destroyAllWindows()
 
