# pull original image 
# pull in the 
import cv2
import glob
import re
import numpy as np
from PIL import Image


def run(file_crop_loc_dict, dspth, adversarial_output_path):
    original_file_name = glob.glob(dspth)[0]
    original_img = cv2.imread(original_file_name)
    original_img = cv2.resize(original_img,(512,512), Image.BILINEAR)
    
    print(original_file_name)
    for image_path in glob.glob(adversarial_output_path):
        adv_img = cv2.imread(image_path)
        file_prefix = re.search(r'file\d{1,2}',image_path)
        segment_number = file_prefix.group()[4:]
        crop_location_x, crop_location_y = file_crop_loc_dict[int(segment_number)]
        # parsing_mask = np.zeros(original_img.shape,dtype=np.uint8)
        # crop_location_x, crop_location_y = [9, 13]
        center = (16+(crop_location_y*16), 16+(crop_location_x*16))
        mask = 255 * np.ones(adv_img.shape, adv_img.dtype)
        print(image_path)
        original_img = cv2.seamlessClone(adv_img, original_img, mask, center, cv2.NORMAL_CLONE)
        # original_img[crop_location_x*16:32+(crop_location_x*16),crop_location_y*16:32+(crop_location_y*16)] = adv_img
        # print(parsing_mask.shape, original_img.shape)
        # cv2.imwrite('parsing_mask.png',parsing_mask)
        # parsing_mask = parsing_mask.astype(np.uint8)
        # masked_segment = cv2.bitwise_xor(original_img, parsing_mask)
        # masked_segment = cv2.cvtColor(masked_segment, cv2.COLOR_RGB2BGR)
    cv2.imwrite('testing_output_2eps.png',original_img)    

        

def crop_segment(masked_segment):
    img_grey = cv2.cvtColor(masked_segment, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(cv2.cvtColor(img[0:32,0:32], cv2.COLOR_BGR2RGB))
    max_useful = 0
    max_point = []
    cropped_segment = masked_segment
    for section_x in range(int(512/16)):
        for section_y in range(int(512/16)):
            percent_useful = sum(sum(im_bw[section_x*16:32+(section_x*16),section_y*16:32+(section_y*16)]/255))/(32*32)
            if percent_useful > max_useful:
                max_useful = percent_useful
                max_point = [section_x, section_y]
                cropped_segment = masked_segment[section_x*16:32+(section_x*16),section_y*16:32+(section_y*16)]
    return cropped_segment, max_point