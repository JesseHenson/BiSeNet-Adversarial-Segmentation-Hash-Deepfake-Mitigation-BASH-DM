import sys
import os
import os.path as osp
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
sys.path.append("./")
from experiments.model import BiSeNet

dspth='./lfw-deepfunneled/Adam_Sandler'
respth='./res/test_res'
cp='79999_iter.pth'

def vis_parsing_maps(im, parsing_anno, stride, save_path):
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
        vis_masked = cv2.cvtColor(vis_masked, cv2.COLOR_RGB2BGR)
        pi_extension = ''
        if len(str(pi)) == 1: 
            pi_extension = f'0{pi}'
        else: 
            pi_extension = str(pi)
        cv2.imwrite(save_path[:-4] + '_' + pi_extension +'.png', vis_masked)


def evaluate():
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    
    if not os.path.exists(respth):
        os.makedirs(respth)
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
            vis_parsing_maps(image, parsing, stride=1, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate()
