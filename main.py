import numpy as np
import cv2
import os
import glob

def FaceCropped(full_path, extra='face', show=False):
    face_cascade = cv2.CascadeClassifier('.../haarcascade_frontalface_default.xml')
    full_path = full_path
    path,file = os.path.split(full_path)


    ff = np.fromfile(full_path, np.uint8)
    img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in faces:
        cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
        result, encoded_img = cv2.imencode(full_path, cropped)
        if result:
            with open(path + '/' + extra + file, mode='w+b') as f:
                encoded_img.tofile(f)
    if show:
        cv2.imshow('Image view', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


```
Save faces of all images  with face recognition
```
