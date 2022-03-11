#%%
import os
import tensorflow as tf
import model_utils, postprocessing_utils, utils

#%%
def yolo_v3():
    hyper_params = utils.get_hyper_params()
    input_shape = (hyper_params["img_size"], hyper_params["img_size"], 3)

    model = model_utils.yolo_v3(input_shape, hyper_params)

    weights_dir = os.getcwd() + "/yolov3_org_weights/weights"
    model.load_weights(weights_dir)
    

    return model


