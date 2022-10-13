from main.get_and_untar_dataset import get_and_untar_dataset
from main.segment_faces import evaluate as segment_faces
from main.generate_adversarial_attack import run as generate_adversarial_image
from main.reconstruct import run as reconstruct

url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
adversarial_output_path = "./results/pictures/*png"
dspth = './lfw-deepfunneled/Adam_Sandler/*_0001.jpg'


path_to_segment = get_and_untar_dataset(url)
# add options for evaulation
file_crop_loc_dict = segment_faces(dspth)
# generate_adversarial_image(dataset='cifar100', model_type="resnet", epochs=100, data_augmentation=True, pictures_path=adversarial_output_path)
reconstruct(file_crop_loc_dict, dspth, adversarial_output_path)
