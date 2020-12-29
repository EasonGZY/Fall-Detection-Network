from detect_main import *
from opt import opt
import os
import pickle
import numpy as np
import datetime
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    yolo_model, pose_model = load_model(opt)
    data_path = "MULT_1fps"# the frames are collected by video clip at 1 FPS
    data_path_out = "Mult_1fps_feature"
    classes=os.listdir(data_path)
    classes.sort()
    start = datetime.datetime.now()
    for classname in classes:
        chutefiles = os.listdir(data_path + '/' + classname)
        chutefiles.sort()
        for chu in chutefiles:
            camfile = os.listdir(data_path + '/' + classname + "/" + chu)
            camfile.sort()
            for cam in camfile:
                im_names = os.listdir(data_path + '/' + classname + '/' + chu + '/' + cam)
                im_names.sort()
                allkp_res = []
                for im_name in im_names:
                    # print(data_path+'/'+chu + '/' + cam + '/' + im_name)
                    img = cv2.imread(data_path + '/' + classname + '/' + chu + '/' + cam + '/' + im_name)
                    keypoint_res = detect_main(im_name, img, yolo_model, pose_model, opt)
                    # print(fall_res)
                    # print(fall_res)
                    # print(keypoint_res)
                    # print(len(keypoint_res))
                    if len(keypoint_res) > 0:
                        np.array(keypoint_res)
                        allkp_res.append(keypoint_res)
                if len(allkp_res) > 0:
                    print(data_path_out + '/' + classname + '/' + chu + '/' + cam)

                    arr = np.zeros((len(allkp_res),) + allkp_res[0][0].shape)
                    for i in range(len(allkp_res)):
                        arr[i] = allkp_res[i][0]
                    #print(arr)
                    write_out = open(data_path_out + '/' + classname + '/' + chu + '/' + cam + '.dat', 'wb')
                    pickle.dump(arr, write_out, protocol=2)
                    write_out.close()
            end = datetime.datetime.now()
            runtime = end - start
            print(start, end, runtime)

    end = datetime.datetime.now()
    runtime = end -start
    print(start, end, runtime)
    print("run time is", runtime)
        # im_path = 'Fall_data_12fps/chute13/cam2/'
        # # im_path = 'err_data/no_person/'
        # im_names = os.listdir(im_path)
        # im_names.sort()
        # print(im_names)













