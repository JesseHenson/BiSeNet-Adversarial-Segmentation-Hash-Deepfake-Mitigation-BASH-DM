## ITERATIVE SEGMENT ATTACK METHOD
import filecmp
import sys
import os
import os.path as osp
# from typing import final
# from imageio import save
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import cv2
from stegano import lsb

sys.path.append("./")
from main.model import BiSeNet


model_path='79999_iter.pth'
dest_path = './results/'



def vis_parsing_maps(im, parsing_anno, stride, save_im, save_path):
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

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def write_preprocessed(image, dest_path, extension):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    cv2.imwrite(f'{dest_path}/{extension}.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return 

def simple_attack(vis_im):
    attack_matrix = np.random.random(vis_im.shape) * 100
    adv_im = attack_matrix.astype(int) % 5 + vis_im
    return adv_im

def hash_image_diff(vis_segment_tensor, adv_segment_tensor):
    hash_diff = vis_segment_tensor - adv_segment_tensor
    return hash_diff

def pre_process_segment_tensor(im, parsing_anno, stride):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    segment_tensor = parsing_anno.copy().astype(np.uint8)
    segment_tensor = cv2.resize(segment_tensor, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    return vis_im,segment_tensor

def get_watermark(image):
    adv_image = simple_attack(image)
    return adv_image
        

def get_net():
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
        
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

def get_parsing(net, to_tensor, image_path=None):
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
            image, parsing = get_parsing(net, to_tensor, image_path=image_path)
            vis_im, segment_tensor = pre_process_segment_tensor(image, parsing, stride=1)
            write_preprocessed(vis_im, dest_path, 'original')
            vis_parsing_maps(image, parsing, 1, True, 'results/original_seg.jpeg')
            adv_image = get_watermark(vis_im)
            print(type(adv_image))
            adv_image_path = f'{dest_path}/adv.png'
            adv_image, adv_parsing = get_parsing(net, to_tensor, adv_image_path)
            adv_vis_im, adv_segment_tensor = pre_process_segment_tensor(adv_image, adv_parsing, stride=1)
            write_preprocessed(adv_vis_im, dest_path, 'adv')
            vis_parsing_maps(adv_vis_im, adv_parsing, 1, True, 'results/adv_seg.jpeg')
            hash_diff = hash_image_diff(segment_tensor, adv_segment_tensor)
            steg_msg = lsb.reveal(adv_image_path)
            
            np.set_printoptions(formatter={'int':hex})
            print("success")
            
            print(f'the hash difference is {"".join([hex(x) for x in hash_diff.sum(axis=1).tolist()])} - {"".join([hex(x) for x in hash_diff.sum(axis=0).tolist()])}')
            print(steg_msg)
    return

if __name__ == "__main__":
    evaluate()
