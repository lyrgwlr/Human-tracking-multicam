# Human-tracking-multicam
## Introduction 
This is a method that can track peoples among mult-cameras.  
The framework of this work is combining [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3) with [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).  
  
The tracking result of [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3) is not stable enough. The track_id of the same person would change when he goes outside the camera and back or occlusion happended.  
[Deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) is a reid Pytorch framework. I trained a model using [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset on it. The top-5 result reached 95%+ and the top-3 result reached 90%+.  
Therefore, my work is making a little improvement of [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3), and fusing different track_ids that belong to the same person using parts of [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).    
  
## Installation  
1. Download YOLOv3 or tiny_yolov3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).Then convert the Darknet YOLO model to a Keras model. Or use what [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3) had converted https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view?usp=sharing (yolo.h5 model file with tf-1.4.0) , put it into model_data folder.  
  
2.Install the Dependencies followed:

    NumPy
    sklean
    OpenCV
    Pillow
    TensorFlow(>=1.4.0)
  
3. Follow the Installation part of [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to install torchreid.  

4. Download the torchreid model from [Google Drive](https://drive.google.com/open?id=15Ayri_sHtrctJ1Zb8qERjvdi66y6QaI4) or [Baidu Driver](https://pan.baidu.com/s/1Y2eXyPzDmrUgetc1aGUd0A) (password: h09w) and put it into model_data folder.  


