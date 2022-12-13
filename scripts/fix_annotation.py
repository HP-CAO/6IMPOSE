import os
import numpy as np

# path_all = [
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/01/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/02/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/04/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/05/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/06/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/08/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/09/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/10/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/11/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/12/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/13/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/14/preprocessed/darknet/obj/',
#     '/mnt/scratch1/dataset/blender/blender/blender_linemod/15/preprocessed/darknet/obj/'
# ]

path_all = [
    '/mnt/scratch1/dataset/blender/blender/blender_linemod/12/preprocessed/darknet/obj/'
]


for path in path_all:
    file_list = os.listdir(path)
    n = 0
    total = len(file_list)
    for file in file_list:
        box = []
        if file.split(".")[1] == 'txt':
            file_path = os.path.join(path, file)
            try:
                ano = np.loadtxt(file_path)
                if len(ano) == 0:
                    n += 1
                else:
                    obj_class = int(0)
                    bbox = f"{obj_class} {ano[1]:.6f} {ano[2]:.6f} {ano[3]:.6f} {ano[4]:.6f}"  # 0 is class (single class for now)
                    with open(file_path, 'w') as F:
                        F.write(bbox)
            except:
                continue

    print(f"{n} out of {total} images are empty. Missing rate{n / total}")