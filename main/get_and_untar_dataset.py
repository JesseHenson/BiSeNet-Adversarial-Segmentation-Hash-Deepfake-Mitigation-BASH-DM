import tarfile
import shutil
import requests
import glob

dest_tar_path = '/home/jessehenson/Desktop/Segmentation_Adversarial_Watermarking_for_Deepfakes/lfw-deepfunneled.tgz'
dest_extract_path = '/home/jessehenson/Desktop/Segmentation_Adversarial_Watermarking_for_Deepfakes/lfw-deepfunneled'

def extract(tar_url, extract_path='.'):
    print(tar_url)
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])

def get_and_untar_dataset(url):
    if len(glob.glob(dest_extract_path)) > 0:
        print('Dataset tar file already extracted')
    else:
        if len(glob.glob(dest_tar_path)) > 0:
            print('Dataset tar file already found')
        else:
            response = requests.get(url, stream=True)

            with open('lfw-deepfunneled.tgz', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)

            print('The file was saved successfully')
            
        try:
            extract('lfw-deepfunneled' + '.tgz')
            print('dataset tar file extracted')
        except:
            print('file not found')
    return dest_extract_path
    
    

