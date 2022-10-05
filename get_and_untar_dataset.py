import os, tarfile
import shutil
import requests

url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
response = requests.get(url, stream=True)

with open('lfw-deepfunneled.tgz', 'wb') as out_file:
  shutil.copyfileobj(response.raw, out_file)

print('The file was saved successfully')
def extract(tar_url, extract_path='.'):
    print(tar_url)
    tar = tarfile.open(tar_url, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
try:

    extract('lfw-deepfunneled' + '.tgz')
    print('Done.')
except:
    'file not found'