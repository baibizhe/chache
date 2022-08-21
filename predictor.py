import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import time
import math
import onnxruntime
import json
def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)

def post_processing(img, conf_thresh, nms_thresh, output):


    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)

    t3 = time.time()
    
    return bboxes_batch
def sigmoid(x):
    result = []
    for i in x[0][0]:
        result.append(1 / (1 + math.exp(-i)))
    return  result

class Point(object):
    def __init__(self,x1Param = 0.0,y1Param = 0.0,x2Param = 0.0,y2Param = 0.0):
        self.x1 = x1Param
        self.y1 = y1Param
        self.x2 = x2Param
        self.y2 = y2Param


def get_image_tensor(image_path):
    image_src = cv2.imread(image_path)
    image_src = np.moveaxis(image_src, -1, 0)
    # image_result = np.pad(image_src,(1296,2304),constant_values = 0)
    padshape =  [(0, max(sp_i - image_src.shape[i], 0)) for i, sp_i in enumerate((3,1296,2304))]
    data_pad_width = padshape
    all_pad_width = data_pad_width
    if not np.asarray(all_pad_width).any():
        # all zeros, skip padding
        return image_src

    image_final = np.expand_dims(0,np.pad(image_src, all_pad_width, constant_values = 0))

    return image_final
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class predictor():
    def __init__(self):
        self.model_list=[]
        self.fold = 3
        for i in range(self.fold):
            model_name = os.path.join("saved_models","class_model{}.onnx".format(i))
            onnx_model = onnx.load(model_name)
            onnx.checker.check_model(onnx_model)
            onnx_model = onnxruntime.InferenceSession(model_name)
            self.model_list.append(onnx_model)
        print(self.model_list)
        # self.session = onnxruntime.InferenceSession(self.onnx_path_demo)

    def predict(self,image_paths):
        result = []
        result_dict = {0:"__ignore__",1:"Dangerous",2: 'not Dangerous'}
        keys = ['__ignore__', 'Dangerous', 'not Dangerous']
        test_r  = 0
        for image in image_paths:
            image_tensor = get_image_tensor(image)
            image_tensor = np.expand_dims(image_tensor,0)
            print(image_tensor.shape)

            voting_list = []
            for ort_session in self.model_list:
                ort_inputs = {ort_session.get_inputs()[0].name: image_tensor.astype(np.single)}
                ort_outs = sigmoid(ort_session.run(None, ort_inputs))
                ort_outs=np.argmax(ort_outs,0)
                voting_list.append(ort_outs)
            # print(voting_list)
            counts = np.bincount(voting_list)
            voting_result = np.argmax(counts)
            print(result_dict[voting_result]+"   "+image)
            jfile_name = image.split("\\")[3].split(".jpg")[0]
            f = open(os.path.join("data/train/json文件", jfile_name+".json"))
            joson_obj = json.load(f)
            for i in range(3):
                if joson_obj["flags"][keys[i]]:
                    label = i
            print(i,voting_result)




            single_result = {"name": image.split('/')[-1].split('.')[0], "result": result_dict[voting_result]}
            # single_result = {"name": image.split('/')[-1].split('.')[0], "result": "danger"}

            # single_result = {"name": image.split('/')[-1].split('.')[0], "result": "danger"}

        return json.dumps(result)

def detect(session, image_src,image_path):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)


    return boxes

if __name__ == '__main__':

    if len(sys.argv) == 2:
        # 输入图片路径
        image_paths = sys.argv[1]
        img_path=os.listdir(image_paths)
        img_paths=[]
        for image in img_path:
           img_paths.append(os.path.join(image_paths,image))
        
        predict_result=[]
        model=predictor()
        predict_result=model.predict(img_paths)
        print(predict_result)


    else:
        print('Please run this way:\n')
        print('  python predictor.py  <imageFile> ')
