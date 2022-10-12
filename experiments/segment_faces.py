import sys
import os
import os.path as osp
from typing import final
from imageio import save
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
sys.path.append("./")
from experiments.model import BiSeNet

dspth='./lfw-deepfunneled/Adam_Sandler'
respth='./res/face_segments'
cp='79999_iter.pth'


def crop_segment(masked_segment):
    grey_img = cv2.cvtColor(masked_segment, cv2.COLOR_BGR2GRAY)	
    ret, thresh = cv2.threshold(grey_img, 127, 255, 0)
    M = cv2.moments(thresh)
    if M["m00"] != 0: 
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cropped_segment = masked_segment[cY-16:cY+16, cX-16:cX+16]
    else: 
        active_pixels = np.stack(np.where(masked_segment))
        top_left = np.min(active_pixels, axis=1).astype(np.int32)
        cropped_segment = masked_segment[top_left[0]:top_left[0]+32,top_left[1]:top_left[1]+32]
    return cropped_segment

def vis_parsing_maps(im, parsing_anno, stride, img_path):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    segment_tensor = parsing_anno.copy().astype(np.uint8)
    segment_tensor = cv2.resize(segment_tensor, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    num_of_segments = np.max(segment_tensor)
    final_img_collect = np.zeros(im.shape[:2],dtype=np.uint8)
    parsing_masks = np.zeros(im.shape[:2], dtype=np.uint8)
    for segment in range(1, num_of_segments + 1):
        parsing_mask = np.zeros(im.shape[:2],dtype=np.uint8)
        index = np.where(segment_tensor == segment)
        if len(index[0]) + len(index[1]) == 0:
            continue
        parsing_mask[index[0], index[1]] = 255
        
        masked_segment = cv2.bitwise_and(vis_im, vis_im, mask=parsing_mask)
        masked_segment = cv2.cvtColor(masked_segment, cv2.COLOR_RGB2BGR)
        
        cropped_segment = crop_segment(masked_segment)
        
        pi_extension = ''
        if len(str(segment)) == 1: 
            pi_extension = f'0{segment}'
        else: 
            pi_extension = str(segment)
        final_path = osp.join(respth,img_path[-8:-4],img_path[:-4] + '_' + pi_extension +'.png')
        if not os.path.exists(f'{respth}/{img_path[-8:-4]}'):
            os.makedirs(f'{respth}/{img_path[-8:-4]}')
        if not os.path.exists(f'{respth}/{img_path[-8:-4]}/ref'):
            os.makedirs(f'{respth}/{img_path[-8:-4]}/ref')
        cv2.imwrite(f'{respth}/{img_path[-8:-4]}/{img_path[:-4]}_{pi_extension}.png', cropped_segment)
        cv2.imwrite(f'{respth}/{img_path[-8:-4]}/ref/{img_path[:-4]}_{pi_extension}.png',masked_segment)
    # if segment has non black pixels add to tensor 
    # return tensor of all images

def crop_imgs(img): 
    pass

def evaluate():
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    # net.cuda()
    
    if not os.path.exists(respth):
        os.makedirs(respth)
        
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512,512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(image_path)
            # print(parsing)
            # print(np.unique(parsing))
            # print(len(np.unique(parsing)))
            vis_parsing_maps(image, parsing, stride=1, img_path=image_path)
            print("success")

if __name__ == "__main__":
    evaluate()
