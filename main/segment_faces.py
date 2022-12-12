import filecmp
import sys
import os
import os.path as osp
from typing import final
from imageio import save
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import cv2
sys.path.append("./")
from main.model import BiSeNet


respth='./results/cropped_segments'
model_path='79999_iter.pth'
dest_path = ''

def pre_process_segment_tensor(im, parsing_anno, stride):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    segment_tensor = parsing_anno.copy().astype(np.uint8)
    segment_tensor = cv2.resize(segment_tensor, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    return im,vis_im,segment_tensor

def mask_segment(im, vis_im, index):
    parsing_mask = np.zeros(im.shape[:2],dtype=np.uint8)
    parsing_mask[index[0], index[1]] = 255
    masked_segment = cv2.bitwise_and(vis_im, vis_im, mask=parsing_mask)
    masked_segment = cv2.cvtColor(masked_segment, cv2.COLOR_RGB2BGR)
    return masked_segment

def crop_segment(masked_segment):
    img_grey = cv2.cvtColor(masked_segment, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(cv2.cvtColor(img[0:32,0:32], cv2.COLOR_BGR2RGB))
    max_useful = 0
    max_point = []
    cropped_segment = masked_segment
    for section_x in range(int(512/16)):
        for section_y in range(int(512/16)):
            percent_useful = sum(sum(im_bw[section_x*16:64+(section_x*16),section_y*16:64+(section_y*16)]/255))/(32*32)
            if percent_useful > max_useful:
                
                max_useful = percent_useful
                max_point = [section_x, section_y]
                cropped_segment = masked_segment[section_x*16:32+(section_x*16),section_y*16:32+(section_y*16)]
    cropped_segment = cv2.resize(cropped_segment,(224,224))
    # print(max_useful, max)
    return cropped_segment, max_point

def write_images(img_path, segment, masked_segment, cropped_segment):
    if not os.path.exists(respth):
        os.makedirs(respth)
    pi_extension = ''
    if len(str(segment)) == 1: 
        pi_extension = f'0{segment}'
    else: 
        pi_extension = str(segment)
    if not os.path.exists(f'{respth}/{img_path[-8:-4]}'):
        os.makedirs(f'{respth}/{img_path[-8:-4]}')
    if not os.path.exists(f'{respth}/{img_path[-8:-4]}/ref'):
        os.makedirs(f'{respth}/{img_path[-8:-4]}/ref')
    print(f'{respth}/{img_path[-8:-4]}/{img_path[-4:]}_{pi_extension}.png')
    cv2.imwrite(f'{respth}/{img_path[-8:-4]}/{pi_extension}.png', cropped_segment)
    cv2.imwrite(f'{respth}/{img_path[-8:-4]}/ref/{pi_extension}.png',masked_segment)
    # if segment has non black pixels add to tensor 
    # return tensor of all images


def vis_parsing_maps(im, parsing_anno, stride, img_path):
    im, vis_im, segment_tensor = pre_process_segment_tensor(im, parsing_anno, stride)
    num_of_segments = np.max(segment_tensor)
    file_crop_loc_dict = {}
    for segment in range(1, num_of_segments + 1):
        index = np.where(segment_tensor == segment)
        if len(index[0]) + len(index[1]) == 0:
            continue
        masked_segment = mask_segment(im, vis_im, index)
        cropped_segment, cropped_segment_location = crop_segment(masked_segment)
        write_images(img_path, segment, masked_segment, cropped_segment)
        file_crop_loc_dict[segment] = cropped_segment_location
    return file_crop_loc_dict
        

def get_net():
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    # net.cuda()
        
    save_pth = osp.join('saved_models', model_path)
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()
    
    return net

def get_transformations():
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    return to_tensor

def get_parsing(net, to_tensor, image_path, dspth):
    img = Image.open(image_path)
    image = img.resize((512,512), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    out = net(img)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return image,parsing

def evaluate(dspth='./lfw-deepfunneled/Adam_Sandler/*_0001.jpg'):
    net = get_net()
    to_tensor = get_transformations()
    with torch.no_grad():
        for image_path in glob.glob(dspth):
            print(image_path)
            image, parsing = get_parsing(net, to_tensor, image_path, dspth)
            file_crop_loc_dict = vis_parsing_maps(image, parsing, stride=1, img_path=image_path)
        
            print("success")
    return file_crop_loc_dict

if __name__ == "__main__":
    evaluate()
