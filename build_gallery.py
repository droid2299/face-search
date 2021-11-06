import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pickle

def get_embeddings(frame , count):
    print("FRAME NO = "+str(count))
    #img = cv2.imread(frame)
    faces = app.get(frame)
    #rimg = app.draw_on(img, faces)
    #cv2.imwrite("./t1_output.jpg", rimg)
    embedding_list = []
    detection_list = []
    if(len(faces) == 0):
      return None
    else:
        for i in range(0 , len(faces)):
            x1 = faces[i]['bbox'][0]
            y1 = faces[i]['bbox'][1]
            x2 = faces[i]['bbox'][2]
            y2 = faces[i]['bbox'][3]
            gender = faces[i]['gender']
            age = faces[i]['age']
            embeddings = faces[i].normed_embedding
            key_point1 = (faces[i]['kps'][0][0] , faces[i]['kps'][0][1])
            key_point2 = (faces[i]['kps'][1][0] , faces[i]['kps'][1][1])
            key_point3 = (faces[i]['kps'][2][0] , faces[i]['kps'][2][1])
            key_point4 = (faces[i]['kps'][3][0] , faces[i]['kps'][3][1])
            embedding_list = [x1 , y1 , x2 , y2 , gender , age , embeddings , key_point1 , key_point2 , key_point3 , key_point4]
            detection_list.append(embedding_list)
            #print(detection_list)
            #print(len(faces))
        return detection_list

def make_pickle(gallery):
    with open('ronaldo_sneakers_local.pkl' , 'wb') as f:
         pickle.dump(gallery , f)

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
cap = cv2.VideoCapture('ron_sneakers_3.mp4')
gallery = {}
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
# Check if camera opened successfully
if (cap.isOpened()== False): 
        print("Error opening video stream or file")
count = 0
while (cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
           break
        embeddings_returned = get_embeddings(frame , count)
        if embeddings_returned == None:
           print("No Face detected")
        else:
           print(embeddings_returned)
           gallery[count] = embeddings_returned
        count += 1

cap.release()
# Closes all the windows currently opened.
#cv2.destroyAllWindows()

make_pickle(gallery)