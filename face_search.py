import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pickle
from scipy import spatial

def calculate_similarity(input_image , gallery):
    similarity_dict = {}
    max_similarity = 0
    max_cos_similarity = 0
    max_cos_similarity2 = 0
    for i in gallery.keys(): 
        each_frame = gallery[i]
        #print(each_frame)
        for j in range(0 , len(each_frame)):

            each_detection = each_frame[j]
            sim = np.dot(input_image[6] , each_detection[6])
            #print(sim)
            cosine_similarity = 1 - spatial.distance.cosine(input_image[6] , each_detection[6])
            #max_similarity = max(max_similarity , sim)
            #max_cos_similarity = max(max_cos_similarity , cosine_similarity)
            if cosine_similarity > 0.50:
                temp_list = [each_detection[0] , each_detection[1] , each_detection[2] , each_detection[3] , cosine_similarity]
                similarity_dict[i] = temp_list
            
    return similarity_dict
'''
def calculate_similarity(input_image , gallery):
    similarity_dict = {}
    max_similarity = 0
    max_cos_similarity = 0
    max_cos_similarity2 = 0
    for frame in gallery: 
        print("Frame: "+str(frame))
        
        for detection in frame:

            each_detection = each_frame[j]
            sim = np.dot(input_image[6] , each_detection[6])
            #print(sim)
            cosine_similarity = 1 - spatial.distance.cosine(input_image[6] , each_detection[6])
            #max_similarity = max(max_similarity , sim)
            #max_cos_similarity = max(max_cos_similarity , cosine_similarity)
            if cosine_similarity > 0.55:
                temp_list = [each_detection[0] , each_detection[1] , each_detection[2] , each_detection[3] , cosine_similarity]
                similarity_dict[i] = temp_list
            
    return similarity_dict
'''

def overlay_video(video_name , similarity_dict):

    cap = cv2.VideoCapture(video_name)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    out= cv2.VideoWriter('overlay.avi', cv2.VideoWriter_fourcc(*'MJPG'),FPS, size)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
           break

        if count in similarity_dict.keys():
            print(count)
            frame = cv2.putText(frame , 'Frame No: '+str(count) , (30 , 30) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,255,0) , 1 , cv2.LINE_AA)

            temp_list = similarity_dict[count]
            x1 = int(temp_list[0])
            y1 = int(temp_list[1])
            x2 = int(temp_list[2])
            y2 = int(temp_list[3])
            conf = int(temp_list[4] * 100)
            frame = cv2.putText(frame , 'Confidence: '+str(conf)+"%" , (30 , 60) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,255,0) , 1 , cv2.LINE_AA)
            frame = cv2.rectangle(frame , (x1,y1) , (x2,y2) , (0,255,0) , 3)
            count += 1

        else:
            print(count)
            frame = cv2.putText(frame , 'Frame No: '+str(count) , (50 , 50) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,255,0) , 1, cv2.LINE_AA)
            count += 1

        out.write(frame)

    cap.release()
    out.release()




with open('new_face_db_ron_tea.pkl', 'rb') as f:
        gallery = pickle.load(f)

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread('cr7_2.jpg')
faces = app.get(img)



input_image = []
if(len(faces) == 0):
      print("No Faces")
else:
      for i in range(0 , len(faces)):
          x1 = faces[i]['bbox'][0]
          y1 = faces[i]['bbox'][1]
          x2 = faces[i]['bbox'][2]
          y2 = faces[i]['bbox'][3]
          gender = faces[i]['gender']
          age = faces[i]['age']
          embeddings = faces[i]['embedding']
          input_image = [x1 , y1 , x2 , y2 , gender , age , embeddings]

if len(input_image) == 0:
   print("No Matches as no faces found")
else:
    similarity_dict = calculate_similarity(input_image , gallery)
print(similarity_dict)

overlay_video('ron_96_final_cut.mp4' , similarity_dict)


