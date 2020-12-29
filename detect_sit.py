import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from yolo.darknet import Darknet
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from SPPE.src.main_fast_inference import *
from pPose_nms import pose_nms
from fn import vis_frame_fast as vis_frame

from opt import opt
import os
import cv2
from yolo.preprocess import *
from yolo.util import write_results, dynamic_write_results

from dataloader import crop_from_dets, Mscoco


def get_box(prediction, det_inp_dim, im_dim_list, confidence, num_classes, class_num):
    dets = dynamic_write_results(prediction, confidence, num_classes, class_num, nms=True, nms_conf=0.4)
    if isinstance(dets, int) or dets.shape[0] == 0:
        return []
    dets = dets.cpu()
    im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
    scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

    # coordinate transfer
    dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    dets[:, 1:5] /= scaling_factor
    for j in range(dets.shape[0]):
        dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
        dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
    boxes = dets[:, 1:5]
    boxes = boxes.numpy().tolist()
    scores = dets[:, 5:6]
    scores = scores.numpy().tolist()
    # print(scores)
    boxes_out = []
    for i in range(len(boxes)):
        if scores[i][0] >= 0.1:
            boxes_out.append(boxes[i])
    return boxes_out


def compute_overlap(box_hm, box_c):
    S_box_hm = (box_hm[2] - box_hm[0]) * (box_hm[3] - box_hm[1])

    # find the each edge of intersect rectangle
    left_line = max(box_hm[0], box_c[0])
    right_line = min(box_hm[2], box_c[2])
    top_line = max(box_hm[1], box_c[1])
    bottom_line = min(box_hm[3], box_c[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / S_box_hm


def load_model(opt):
    pose_dataset = Mscoco()
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)

    det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights('models/yolo/yolov3-spp.weights')
    det_model.net_info['height'] = opt.inp_dim
    pose_model.cuda()
    pose_model.eval()
    det_model.cuda()
    det_model.eval()

    return det_model, pose_model

    # ImageLoader
    # img, orig_img, im_dim_list = prep_image(im_name, inp_dim)

    # prep_image


def detect_main(im_name, orig_img, det_model, pose_model, opt):
    args = opt
    mode = args.mode
    inp_dim = int(opt.inp_dim)
    dim = orig_img.shape[1], orig_img.shape[0]
    img_ = (letterbox_image(orig_img, (inp_dim, inp_dim)))
    img_ = img_[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    img = [img]
    orig_img = [orig_img]
    im_name = [im_name]
    im_dim_list = [dim]

    #    img.append(img_k)
    #    orig_img.append(orig_img_k)
    #    im_name.append(im_name_k)
    #    im_dim_list.append(im_dim_list_k)

    with torch.no_grad():
        # Human Detection
        img = torch.cat(img)
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        # im_dim_list_ = im_dim_list

    # DetectionLoader

    det_inp_dim = int(det_model.net_info['height'])
    assert det_inp_dim % 32 == 0
    assert det_inp_dim > 32
    # res_n = 0
    with torch.no_grad():
        img = img.cuda()
        prediction = det_model(img, CUDA=True)  # a tensor

        boxes_chair = get_box(prediction, det_inp_dim, im_dim_list, opt.confidence, opt.num_classes, 56)
        # boxes_sofa = get_box(prediction, det_inp_dim, im_dim_list, opt.confidence, opt.num_classes, 57)
        # boxes_bed = get_box(prediction, det_inp_dim, im_dim_list, opt.confidence, opt.num_classes, 59)
        dets = dynamic_write_results(prediction, opt.confidence, opt.num_classes, 0, nms=True, nms_conf=opt.nms_thesh)
        if isinstance(dets, int) or dets.shape[0] == 0:
            # cv2.imwrite('err_result/no_person/'+im_name[0][0:-4]+'_re.jpg', orig_img[0])
            return []
        dets = dets.cpu()
        im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
        scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

        # coordinate transfer
        dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

        dets[:, 1:5] /= scaling_factor
        for j in range(dets.shape[0]):
            dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
            dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]

    boxes_k = boxes[dets[:, 0] == 0]
    if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
        boxes = None
        scores = None
        inps = None
        pt1 = None
        pt2 = None
    else:
        inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
        pt1 = torch.zeros(boxes_k.size(0), 2)
        pt2 = torch.zeros(boxes_k.size(0), 2)
        orig_img = orig_img[0]
        im_name = im_name[0]
        boxes = boxes_k
        scores = scores[dets[:, 0] == 0]

    # orig_img[k], im_name[k], boxes_k, scores[dets[:, 0] == k], inps, pt1, pt2

    # DetectionProcess
    with torch.no_grad():
        if boxes is None or boxes.nelement() == 0:
            pass
        else:
            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

    # self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))

    batchSize = args.posebatch
    # fall_res_all = []

    for i in range(1):
        with torch.no_grad():
            if boxes is None or boxes.nelement() == 0:
                # writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                # res_n = 0
                continue

            # Pose Estimation
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            hm = hm.cpu()
            # writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            fall_res = []
            keypoint_res = []
            # fall_res.append(im_name.split('/')[-1])
            if opt.matching:
                preds = getMultiPeakPrediction(
                    hm, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                result = matching(boxes, scores.numpy(), preds)
            else:
                preds_hm, preds_img, preds_scores = getPrediction(hm, pt1, pt2, opt.inputResH, opt.inputResW,
                                                                  opt.outputResH, opt.outputResW)
                result = pose_nms(boxes, scores, preds_img, preds_scores)
                result = {'imgname': im_name, 'result': result}
            # img = orig_img
            img = vis_frame(orig_img, result)

            for human in result['result']:
                keypoint = human['keypoints']
                kp_scores = human['kp_score']

                keypoint = keypoint.numpy()
                xmax = max(keypoint[:, 0])
                xmin = min(keypoint[:, 0])
                ymax = max(keypoint[:, 1])
                ymin = min(keypoint[:, 1])
                box_hm = [xmin, ymin, xmax, ymax]

                kp_num = 0
                for i in range(len(kp_scores)):
                    if kp_scores[i] > 0.05:
                        kp_num += 1

                if kp_num < 10:
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    fall_res.append([False, xmin, ymin, xmax, ymax])
                    # print("kp_num:"+str(kp_num))
                    continue

                overlap = []
                for box in boxes_chair:
                    overlap.append(compute_overlap(box_hm, box))
                # for box in boxes_sofa:
                #     overlap.append(compute_overlap(box_hm, box))
                # for box in boxes_bed:
                #     overlap.append(compute_overlap(box_hm, box))
                if len(overlap) > 0 and max(overlap) >= 0.6:
                    # res_n = 0
                    fall_res.append([False, xmin, ymin, xmax, ymax])
                    keypoint_res.append(keypoint)
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                    # print("overlap:"+str(overlap))
                    continue

                w = xmax - xmin
                h = ymax - ymin
                # distance = abs((keypoint[15][1] + keypoint[16][1]) / 2 - (keypoint[11][1] + keypoint[12][1]) / 2)
                xhead = (keypoint[1][0] + keypoint[2][0] + keypoint[2][0] + keypoint[3][0] + keypoint[4][0]) / 4
                yhead = (keypoint[1][1] + keypoint[2][1] + keypoint[2][1] + keypoint[3][1] + keypoint[4][1]) / 4
                xfeet = (keypoint[15][0] + keypoint[16][0]) / 2
                yfeet = (keypoint[15][1] + keypoint[16][1]) / 2
                d_ear = (abs(keypoint[3][0] - keypoint[4][0]) ** 2 + abs(keypoint[3][1] - keypoint[4][1]) ** 2) ** 0.5
                r = (w ** 2 + h ** 2) ** 0.5 / d_ear

                if kp_scores[3] > 0.05 and kp_scores[4] > 0.05 and r < 4:
                    fall_res.append([False, xmin, ymin, xmax, ymax])
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    # print("r<4")
                    continue

                    # distance = abs((keypoint[15][1] + keypoint[16][1]) / 2 - (keypoint[11][1] + keypoint[12][1]) / 2)
                # xhead_foot = abs(xfeet - xhead)
                # yhead_foot = abs(yfeet - yhead)
                # dhead_foot = (xhead_foot ** 2 + yhead_foot ** 2) ** 0.5
                # ratio = yhead_foot / dhead_foot

                if min(kp_scores[3], kp_scores[4], kp_scores[15], kp_scores[16]) > 0.05 and yfeet < (
                        keypoint[3][1] + keypoint[4][1]) / 2:
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(img, 'Warning!Fall!', (int(xmin + 10), int(ymax - 10)), font, 1, (0, 255, 0), 2)
                    fall_res.append([True, xmin, ymin, xmax, ymax])

                    # res_n = 2

                elif w / h >= 1.0:
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(img, 'Warning!Fall', (int(xmin + 10), int(ymax - 10)), font, 1, (0, 0, 255), 2)
                    fall_res.append([True, xmin, ymin, xmax, ymax])

                    # res_n = 1

                else:
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    # print("normal")
                    fall_res.append([False, xmin, ymin, xmax, ymax])
                    # res_n = 0
                # cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
    '''
    for box in boxes_chair:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
    for box in boxes_sofa:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
    for box in boxes_bed:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)

    cv2.imwrite('err_result/false/'+im_name[0:-4]+'_re.jpg', img)
    '''
    return keypoint_res


