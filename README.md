# yolov3collage

http://olivain.art/wannabeblog/articles/yolo_image.php

![example](http://olivain.art/blurd/Vnb4YahtVZVu.jpg)

# installation
```
# this script does not work using opencv-4.5.4 !!
python3 -m pip install opencv-python==4.5.3

#install libs
python3 -m pip install pil
python3 -m pip install numpy

#download yolov3
wget https://olivain.art/wannabeblog/misc/coco.names
wget https://olivain.art/wannabeblog/misc/yolov3.cfg
wget https://olivain.art/wannabeblog/misc/yolov3.weights

#download font
curl -O -J -L https://www.fontsquirrel.com/fonts/download/roboto-slab
unzip roboto-slab.zip 

#run the script
python3 yolov3collage.py

```
