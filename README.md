# Face Search

A repository to detect faces in a video from a reference image. We first create a gallery of the input video which creates embeddings of all the faces detected and stores them in a pickle file which will be called gallery. Next, we generate the embeddings of the reference image which will be used to check the similarity of all the faces in the gallery. If there is a match over a certain threshold, those bounding boxes are overlaid on the video.

## Installation & Usage
### Dependencies:
```sh
pip install -r requirements.txt
```
For GPU acceleration,
```sh
pip uninstall onnxruntime
pip install onnxruntime-gpu
```
### Usage:
#### Step 1:

In the `build_gallery.py` file, change the video path to the path for which we need to build a gallery and the pickle file name which will be out gallery.

#### Step 2:
Run the command:
```sh
python build_gallery.py
```
A pickle file will be generated of the name specified in the **Step 1**. This is our gallery.

#### Step 3:
In the `face_search.py` file, change the path to the pickle file to the path of the file generated in **Step 2** (filename specified in Step 1) on ***line 74***. And the path to the video file used in **Step 1** on ***line 104***. 

### Step 4:
Run the command:
```sh
python face_search.py
```
An output file will be generated of the name `overlay.avi` with the bbox overlaid on the face of where the reference image was found in the imput video.

