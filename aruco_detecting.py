import cv2
import cv2.aruco as aruco

capture = cv2.VideoCapture(0)

def findAruco(img,marker_size=6,total_markers=250,draw=True):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{marker_size}X{marker_size}_{total_markers}')
    arucoDict = aruco.getPredefinedDictionary(key)    
    parameters =  aruco.DetectorParameters()
    bbox,ids,_=aruco.detectMarkers(gray,arucoDict,parameters=parameters)
    print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bbox)
    return bbox,ids

while True:
    ret,img = capture.read()

    bbox,ids=findAruco(img)
    if cv2.waitKey(1) == 27:
        break
    cv2.imshow("img",img)