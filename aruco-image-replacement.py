import cv2
import cv2.aruco as aruco
import numpy as np

capture = cv2.VideoCapture(0)
newImage = cv2.imread("cat.jpg")

def replaceImage(bbox, img, new_image):
    top_left = bbox[0][0][0], bbox[0][0][1]
    top_right = bbox[0][1][0], bbox[0][1][1]
    bottom_left = bbox[0][2][0], bbox[0][2][1]
    bottom_right = bbox[0][3][0], bbox[0][3][1]

    height, width, _ = new_image.shape

    points_1 = np.array([top_left, top_right, bottom_left, bottom_right])
    points_2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix, _ = cv2.findHomography(points_2, points_1)
    imageOut = cv2.warpPerspective(new_image, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, points_1.astype(int), (0, 0, 0))
    imageOut = img + imageOut

    return imageOut


def findAruco(img,marker_size=6,total_markers=250,draw=True):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{marker_size}X{marker_size}_{total_markers}')
    arucoDict = aruco.getPredefinedDictionary(key)    
    parameters =  aruco.DetectorParameters()
    bbox,ids,rejectedImgPoints = aruco.detectMarkers(gray,arucoDict,parameters=parameters)
    if draw:
        #aruco.drawDetectedMarkers(img,bbox)
        pass

    return bbox,ids,rejectedImgPoints


while True:
    ret, img = capture.read()
    bbox, ids, _ = findAruco(img)

    if bbox:
        print("oii")
        frame = replaceImage(np.array(bbox)[0], img, newImage)
        cv2.imshow('Replaced Image', frame)
    else:
        cv2.imshow('Camera Feed', img)

    if cv2.waitKey(1) == 27:
        break