#importing libraries
import cv2,time
#take data from the webcam using video capture of opencv
video=cv2.VideoCapture(0)
first_image=None
while True:
    check,frame=video.read() #reading data
    if check==False:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #converting it into gray scale to increase the accuracy of the feature detection
    gray=cv2.GaussianBlur(gray,(21,21),0) #blur this image to smooth it
    if first_image is None:#checking whether the frame coming is the first frame or not for refrence
        first_image=gray
        continue
    delta_image=cv2.absdiff(first_image,gray) #find difference between frames
    treshold_image=cv2.threshold(delta_image,50,255,cv2.THRESH_BINARY)[1]  #detect motion
    treshold_image=cv2.dilate(treshold_image,None,iterations=2)#how accurate your smoothening will be
   #contour detection
    cntr,_ =cv2.findContours(treshold_image.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    #deine the approximate area
    for contour in cntr:
        if cv2.contourArea(contour)<1000: #checking the area of the contour
            continue
        (x,y,w,h)=cv2.boundingRect(contour)#draw a red rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("abir",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

