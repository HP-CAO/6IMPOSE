#!/usr/bin/env sh



./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/10/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-eggbox_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-eggbox_best.weights -gpus 0 -dont_show -mjpeg_port 8090 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/11/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-glue_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-glue_best.weights -gpus 1 -dont_show -mjpeg_port 8091 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/13/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-iron_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-iron_best.weights -gpus 2 -dont_show -mjpeg_port 8092 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/09/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-duck_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-duck_best.weights -gpus 3 -dont_show -mjpeg_port 8093 -map &
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/08/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-driller_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-driller_best.weights -gpus 4 -dont_show -mjpeg_port 8094 -map


#./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/12/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-holepuncher_sup.cfg -gpus 5 -dont_show -mjpeg_port 8098 -map  &
#./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/14/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-lamp_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-lamp_best.weights -gpus 6 -dont_show -mjpeg_port 8099 -map
#./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/15/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-phone_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-phone_best.weights -gpus 7 -dont_show -mjpeg_port 8097 -map



./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/12/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-holepuncher_4gpu_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-holepuncher_sup_best.weights -gpus 0,1,2,3 -dont_show -mjpeg_port 8098 -map
./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/14/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-lamp_4gpu_sup.cfg /home/hongi/devel/darknet/model/darknet_lm_all/yolov4-tiny-lm-lamp_sup_best.weights -gpus 4,5,6,7 -dont_show -mjpeg_port 8098 -map

#./darknet detector train /mnt/scratch1/dataset/blender/blender/blender_linemod/16/preprocessed/darknet/obj.data /home/hongi/devel/pvn/config/yolo_lm_config/yolov4-tiny-lm-all-4gpu.cfg /home/hongi/devel/darknet/pretrained_models/darknet_lm_all/yolov4-tiny-lm-all-4gpu_final.weights -gpus 4,5,6,7 -dont_show -mjpeg_port 8099 -map