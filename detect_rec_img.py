import cv2
import numpy as np
import argparse
from itertools import product as product
from math import ceil
from PIL import Image, ImageDraw, ImageFont

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

def puttext_chinese(img, text, point, color):
    pilimg = Image.fromarray(img)  ###[:,:,::-1]  BGRtoRGB
    draw = ImageDraw.Draw(pilimg)  # 图片上打印汉字
    fontsize = int(min(img.shape[:2]) * 0.04)
    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    draw.text(point, text, color, font=font)
    img = np.asarray(pilimg)  ###[:,:,::-1]   BGRtoRGB
    return img

class detect_plate_recognition:
    def __init__(self, confidence_threshold=0.9, top_k=5000, nms_threshold=0.4, keep_top_k=750, vis_thres=0.6):
        self.model = cv2.dnn.readNet('mnet_plate.onnx')
        self.im_height = 640
        self.im_width = 640
        self.scale = np.array([[self.im_width, self.im_height]], dtype=np.float32)
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.min_sizes = [[24, 48], [96, 192], [384, 768]]
        self.steps = [8, 16, 32]
        self.variance = [0.1, 0.2]
        self.clip = False
        self.prior_data = self.generate_priors()  ####PriorBox生成的一堆anchor在强项推理过程中始终是常数是不变量，因此只需要在构造函数里定义一次即可
        self.points_ref = np.float32([[0, 0], [94, 0], [0, 24], [94, 24]])
        self.LPR = lprnet()
    def generate_priors(self):
        feature_maps = [[ceil(self.im_height / step), ceil(self.im_width / step)] for step in self.steps]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.im_width
                    s_ky = min_size / self.im_height
                    dense_cx = [x * self.steps[k] / self.im_width for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.im_height for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = np.asarray(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        return output
    def decode(self, loc, priors, variances):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        # boxes[:, 2:] += boxes[:, :2]
        return boxes   ### xmin,ymin, width, height
    def decode_landm(self, pre, priors, variances):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            ), axis=1)
        return landms
    def resize_image(self, srcimg):
        top, left, newh, neww = 0, 0, self.im_height, self.im_width
        if srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.im_height, int(self.im_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.im_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.im_width - neww - left, cv2.BORDER_CONSTANT, value=0)  # add border
            else:
                newh, neww = int(self.im_height * hw_scale), self.im_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.im_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.im_height - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    def detect_rec(self, srcimg):
        img, newh, neww, top, left = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, mean=(104, 117, 123))
        self.model.setInput(blob)
        loc, conf, landms = self.model.forward(['loc', 'conf', 'landms'])

        boxes = self.decode(loc, self.prior_data, self.variance)
        boxes = boxes * np.tile(self.scale, (1, 2))    ####广播法则
        scores = conf[:, 1]
        landms = self.decode_landm(landms, self.prior_data, self.variance)
        landms = landms * np.tile(self.scale, (1, 4))   ####广播法则

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence_threshold, self.nms_threshold, top_k=self.keep_top_k)
        boxes -= np.array([[left,top,0,0]])  ###还原到原图上，合理使用广播法则
        landms -= np.tile(np.array([[left,top]]), (1, 4))   ####4个关键点坐标是x1,y1,x2,y2,x3,y3,x4,y4排列
        srcim_scale = np.array([[srcimg.shape[1]/neww, srcimg.shape[0]/newh]], dtype=np.float32)
        boxes = boxes * np.tile(srcim_scale, (1, 2))    ###还原到原图上，合理使用广播法则
        landms = landms * np.tile(srcim_scale, (1, 4))  ####4个关键点坐标是x1,y1,x2,y2,x3,y3,x4,y4排列
        # boxes = boxes.astype(np.int32)
        # landms = landms.astype(np.int32)
        for i in indices:
            idx = i[0]
            if scores[idx]<self.vis_thres:
                continue
            xmin, ymin, width, height = boxes[idx, :]
            new_x1, new_y1 = landms[idx, 4] - xmin, landms[idx, 5]- ymin
            new_x2, new_y2 = landms[idx, 6] - xmin, landms[idx, 7] - ymin
            new_x3, new_y3 = landms[idx, 2] - xmin, landms[idx, 3] - ymin
            new_x4, new_y4 = landms[idx, 0] - xmin, landms[idx, 1] - ymin
            # 定义对应的点
            points = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
            M = cv2.getPerspectiveTransform(points, self.points_ref)
            img_box = srcimg[int(ymin):int(ymin + height), int(xmin):int(xmin + width), :]
            processed = cv2.warpPerspective(img_box, M, (94, 24))
            result = self.LPR.rec(processed)
            # cv2.imshow('plate', processed)
            cv2.rectangle(srcimg, (int(xmin), int(ymin)), (int(xmin + width), int(ymin + height)), (0, 0, 255), thickness=1)
            # cv2.putText(srcimg, str(round(scores[idx], 3)), (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            for j in range(4):
                cv2.circle(srcimg, (int(landms[idx, 2*j]), int(landms[idx, 2*j+1])), 10, (255, 0, 0), thickness=-1)
            srcimg = puttext_chinese(srcimg, result, (int(xmin), int(ymin) - 30), (0, 255, 0))
        return srcimg

class lprnet:
    def __init__(self):
        self.img_size = (94, 24)  ###width, height
        self.model = cv2.dnn.readNet('Final_LPRNet_model.onnx')
    def rec(self, img):
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 128, size=self.img_size, mean=127.5)
        self.model.setInput(blob)
        preb = self.model.forward()
        # preb_label = []
        # for j in range(preb.shape[1]):
        #     preb_label.append(np.argmax(preb[:, j], axis=0))
        preb_label = np.argmax(preb, axis=0)
        no_repeat_blank_label = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        return ''.join(list(map(lambda x: CHARS[x], no_repeat_blank_label)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaPL')
    parser.add_argument('--imgpath', default='/home/wangbo/Desktop/data/yolo/license-plate-detect-recoginition/License-Plate-Detector-master/imgs/3.jpg', type=str, help='show detection results')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=1000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    net = detect_plate_recognition(confidence_threshold=args.confidence_threshold, top_k=args.top_k, nms_threshold=args.nms_threshold, keep_top_k=args.keep_top_k, vis_thres=args.vis_thres)
    srcimg = cv2.imread(args.imgpath)
    srcimg = net.detect_rec(srcimg)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()