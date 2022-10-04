import sys


sys.path.append("./")
from face_parsing_PyTorch.model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    # mask = np.zeros(im.shape[:2], dtype="uint8")
    # # cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
    # # cv2.imshow("Rectangular Mask", mask)
    # parsing_mask = np.array(parsing_anno)
    # masked = cv2.bitwise_and(im, im, mask=parsing_mask)
    # cv2.imshow("Mask Applied to Image", masked)

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    num_of_class = np.max(vis_parsing_anno)
    for pi in range(1, num_of_class + 1):
        vis_parsing_anno_color = np.zeros(im.shape[:2],dtype=np.uint8)
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1]] = 255
        vis_masked = cv2.bitwise_and(vis_im, vis_im, mask=vis_parsing_anno_color)
        # cv2.imshow('vis_masked', vis_masked)
        # cv2.waitKey(2000)
        vis_masked = cv2.cvtColor(vis_masked, cv2.COLOR_RGB2BGR)
        pi_extension = ''
        if len(str(pi)) == 1: 
            pi_extension = f'0{pi}'
        else: 
            pi_extension = str(pi)
        cv2.imwrite(save_path[:-4] + '_' + pi_extension +'.png', vis_masked)


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((32,32), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            print(image_path)
            print(parsing)
            print(np.unique(parsing))
            print(len(np.unique(parsing)))
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate(dspth='./lfw-deepfunneled/Adam_Sandler', cp='79999_iter.pth')
