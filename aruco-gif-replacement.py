# import list
import cv2
import cv2.aruco as aruco
import numpy as np

# setting up variables
capture = cv2.VideoCapture(0)
newGif = cv2.VideoCapture("baby-yoda.gif")

detection = False
frameCount = 0

height_marker, width_marker = 100, 100
_, imageGif = newGif.read()
imageGif = cv2.resize(imageGif, (width_marker, height_marker))

# functions
####################################################################################################
def findAruco(img,marker_size=6,total_markers=250):        
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{marker_size}X{marker_size}_{total_markers}')      # dict name constructor
    arucoDict = aruco.getPredefinedDictionary(key)    
    parameters =  aruco.DetectorParameters()
    bbox,ids,rejectedImgPoints = aruco.detectMarkers(gray,arucoDict,parameters=parameters)
    return bbox,ids,rejectedImgPoints

def replaceGif(bbox, img, new_image):
    top_left = bbox[0][0][0], bbox[0][0][1]                 # acess img bbox coordinates
    top_right = bbox[0][1][0], bbox[0][1][1]
    bottom_left = bbox[0][2][0], bbox[0][2][1]
    bottom_right = bbox[0][3][0], bbox[0][3][1]

    height, width, _ = new_image.shape                      # gets values from the new image that will replace current bbox

    points_1 = np.array([top_left, top_right, bottom_left, bottom_right])
    points_2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])          # use array float 32 for adjust precision

    matrix, _ = cv2.findHomography(points_2, points_1)                                 # find homography between both images
    imageOut = cv2.warpPerspective(new_image, matrix, (img.shape[1], img.shape[0]))    # puts new_image in perspective using the homography matrix
    cv2.fillConvexPoly(img, points_1.astype(int), (0, 0, 0))                           # fill img area with black space
    imageOut = img + imageOut                                                          # add the pixel values of the two images together

    return imageOut
####################################################################################################
# end of functions

# main loop
while True:
    ret, img = capture.read()         # reads webcam image
    bbox, ids, _ = findAruco(img)     # call function findAruco over the webcam image

    if detection == False:                                        
        newGif.set(cv2.CAP_PROP_POS_FRAMES,0)                     # sets to display the first frame
        frameCount = 0
    else:
        if frameCount == newGif.get(cv2.CAP_PROP_FRAME_COUNT):    # if current frame is the last one
            newGif.set(cv2.CAP_PROP_POS_FRAMES, 0)                # resets video display
            frameCount = 0
        _, imageGif = newGif.read()                               # restart 
        imageGif = cv2.resize(imageGif, (width_marker, height_marker))


    if bbox:                                                      # if any bounding box was detected
        detection = True
        frame = replaceGif(np.array(bbox)[0], img, imageGif)      # call replace gif, passing current and next image
        cv2.imshow('Replaced Image', frame)                       # display replaced image
    else:
        cv2.imshow('Camera Feed', img)                            # if none bbox, display original web cam image

    if cv2.waitKey(1) == 27:                                      # esc exits
        break

    frameCount += 1