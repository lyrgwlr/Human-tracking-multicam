# Human-tracking-multicam
## Introduction 
This is a method that can track peoples among mult-cameras.  
The framework of this work is combining [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3) with [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).  
  
The tracking result of [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3) is not stable enough. The track_id of the same person would change when he goes outside the camera and back or occlusion happended.  
[Deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) is a reid Pytorch framework. I trained a model using [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset on it. The top-5 result reached 95%+ and the top-3 result reached 90%+.  
<<<<<<< HEAD
Therefore, my work is making a little improvement of [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3), and fusing different track_ids that belong to the same person using parts of [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).   
=======
Therefore, my work is making a little improvement of [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3), and fusing different track_ids that belong to the same person using parts of [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).  
  
## Installation  


>>>>>>> 2777ce1dea79c88564d0e496d6d614123badd323



 

