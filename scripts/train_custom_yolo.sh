#!/usr/bin/env sh


./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/01/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-ape.cfg -gpus 0 -dont_show -mjpeg_port 8090 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/02/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-benchvise.cfg -gpus 1 -dont_show -mjpeg_port 8091 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/04/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-cam.cfg -gpus 2 -dont_show -mjpeg_port 8092 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/05/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-can.cfg -gpus 3 -dont_show -mjpeg_port 8093 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/06/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-cat.cfg -gpus 4 -dont_show -mjpeg_port 8094 -map
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/10/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-eggbox.cfg -gpus 0 -dont_show -mjpeg_port 8090 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/11/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-glue.cfg -gpus 1 -dont_show -mjpeg_port 8091 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/12/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-holepuncher.cfg -gpus 2 -dont_show -mjpeg_port 8092 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/13/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-iron.cfg -gpus 3 -dont_show -mjpeg_port 8093 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/14/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-lamp.cfg -gpus 4 -dont_show -mjpeg_port 8094 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/15/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-phone.cfg -gpus 5 -dont_show -mjpeg_port 8095 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/09/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-duck.cfg -gpus 6 -dont_show -mjpeg_port 8096 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/08/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-driller.cfg -gpus 7 -dont_show -mjpeg_port 8097 -map
