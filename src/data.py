# Dylan Zaragoza
# CSE 60537: Biometrics
# Facial Detection & Recognition

# Imports
import cv2
import os

# Modifiable variables
dataDir = '../data/'                                                                                            # Data directory

face_cascade = cv2.CascadeClassifier('../opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')    # Cascade type (default frontal face)
scaleFactor = 1.1                                                                                               # Detection parameters
minNeighbors = 1                                                                                                   

augDir = 'data/'                                                                                                # Name of augmented images directory
augFlag = 1                                                                                                     # Output augmented images flag
font = cv2.FONT_HERSHEY_PLAIN                                                                                   # Detection score font

# Extract images from data directory
images = []
for file in os.listdir(dataDir):
    file = file.lower()
    if file.endswith('.jpg'):
        images.append(dataDir + file)

# Face detection
if images:
    output = ""
    
    for image in images:
        # Convert to grayscale
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect bound
        faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
        detections = face_cascade.detectMultiScale(gray, scaleFactor, 0)

        # Calculate detection scores and append to output
        scores = []
        p = 20 # proximity in pixels
        output += image.replace('.jpg','') + '\n'
        output += str(len(faces)) + '\n'
        for f in faces:
            neighbors = 0
                
            for d in detections:
                # Check they are not the same window
                if (f[0] != d[0] or f[1] != d[1] or f[2] != d[2] or f[3] != d[3]):
                    # Check Left X proximity
                    if (d[0] >= f[0] - p and d[0] <= f[0] + p):
                        # Check Top Y proximity
                        if (d[1] >= f[1] - p and d[1] <= f[1] + p):
                            # Check Right X proximity
                            if (d[0] + d[2] >= f[0] + f[2] - p and d[0] + d[2] <= f[0] + f[2] + p):
                                # Check Bottom Y proximity
                                if (d[1] + d[3] >= f[1] + f[3] - p and d[1] + d[3] <= f[1] + f[3] + p):
                                    neighbors +=1

            output += str(f[0]) + ' '
            output += str(f[1]) + ' '
            output += str(f[2]) + ' '
            output += str(f[3]) + ' '
            output += str(neighbors) + '\n'
            scores.append(neighbors)
        
        # Save augmented images with detection scores (optional)
        if (augFlag):
            scoreIndex = 0
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 8)
                cv2.putText(img, str(scores[scoreIndex]),(x,y), font, 2,(0,0,255), 1)
                scoreIndex += 1

            if not os.path.exists(augDir):
                os.mkdir(augDir)

            cv2.imwrite(augDir + image.replace(dataDir,''), img)

    # Save output
    f = open(augDir + '/data.txt', 'wb')
    f.write(output)
    f.close()
