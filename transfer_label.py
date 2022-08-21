import os
import json
import numpy as np
all_json_file = os.listdir("data/train/json文件")
# for j_file in all_json_file:
#     f = open(os.path.join("data/train/json文件", j_file))
#
#     joson_obj = json.load(f)
#     keys = ['__ignore__', 'Dangerous', 'not Dangerous']
#     label = None
#     for i in range(3):
#         if  joson_obj["flags"][keys[i]]:
#             label = np.asarray([i])
            # np.save(os.path.join("data/train/label", j_file.split(".json")[0]), label)
            # print(keys[i])
    # print(joson_obj["flags"])
    # print(joson_obj["imageHeight"])
    # print(joson_obj["imageWidth"])
for j_file in all_json_file:
    # f = open(os.path.join("data/训练集-1979/json文件",j_file))
    # joson_obj = json.load(f)
    # f.close()
    # print(joson_obj["flags"])
    print(j_file)
    label = np.load(os.path.join("data/train/label",j_file.split(".json")[0]+".npy"))
    print(label)