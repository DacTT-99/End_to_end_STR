from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import time
import argparse
import cv2

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


import cv2
import numpy as np
import craft_utils
import imgproc

from craft import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# detector, detect text 
class detector():
    def __init__(self):
        self.net=CRAFT()
        if args.cuda:
            self.net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        else:
            self.net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
        if args.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        self.net.eval()
    def predict(self, image, text_threshold, link_threshold, low_text, cuda, poly):
        t0 = time.time()
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if args.show_time : 
            print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text

class recognizor():
    def __init__(self):
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = './weights/transformerocr.pth'
        self.config['transformer']['num_decoder_layers'] = 3
        self.config['transformer']['num_encoder_layers'] = 3
        self.config['vocab'] = 'aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        self.detector = Predictor(self.config)
    def predict(self,img):
        return self.detector.predict(img)



def main(args):
    craft = detector()
    tfm_ocr = recognizor()
    image, img_origin = imgproc.loadImage('/home/list_99/Pictures/PDC.png')

    boxs,polys,imgs= craft.predict(image,args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)
    for p in polys:
        a = np.array(p).reshape((-1)).astype(np.int32)
        max_x=max((a[0],a[2],a[4],a[6]))
        min_x=min((a[0],a[2],a[4],a[6]))
        max_y=max((a[1],a[3],a[5],a[7]))
        min_y=min((a[1],a[3],a[5],a[7]))
        box = [min_x,min_y,max_x,max_y]
        crop = img_origin.crop(box)
        ocr_res = tfm_ocr.predict(crop)
        print(box)
        print(ocr_res)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='end to end')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')  ###### <- edit to weight path
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()
    main(args)