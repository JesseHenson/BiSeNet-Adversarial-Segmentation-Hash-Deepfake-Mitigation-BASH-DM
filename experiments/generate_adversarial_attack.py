"""
Generating Steganography images using Adversarial attacks 
"""



import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED

import os
import keras
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import save_img, load_img, img_to_array
from keras.datasets import cifar10   
import numpy as np
import random, json, time, os
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent
from PIL import Image
import glob
tf.compat.v1.disable_eager_execution()


from stegano import lsb

experiment_time = int(time.time())
strs = "0123456789"
# results need to have image, model, message
segmented_dir = "./res/test_res/"
default_path = "./results/pictures/{}"
default_extension = "png"
palette = 256


def craft_attack(model, x,y=None, epsilon=1., minimal=True):
    attack_params = {"norm": 2,'minimal': minimal,"targeted":True, "eps_step":0.1, "eps":epsilon}
    classifier = KerasClassifier(model=model)
    crafter = ProjectedGradientDescent(classifier)
    crafter.set_params(**attack_params)
    adv_x = crafter.generate(x,y)
    return adv_x

def get_segmented_data():
    filelist = glob.glob(f'{segmented_dir}/*.png')
    # for fname in filelist:
    #     img = Image.open(fname)
    #     img = img.resize()
    x = np.array([np.array(Image.open(fname).resize((32,32))) for fname in filelist])
    y = np.array([np.array(int(fname[-6:-4])) for fname in filelist])
    encoding_base = np.max(y)
    print(encoding_base, x.shape, y.shape)
    return encoding_base, x, y
    # for image in image_set:
    # x_train is the images, y_train is int([-5,-3]
    # x = images
    # y = int(label[-5,-3])
    # take first 200 images as train next 200 as test 
    # return (x_train, y_train), (x_test, y_test)
    # 

def get_dataset(num_classes):

    # get_segmented_data()
    encoding_base,  x_test, y_test = get_segmented_data()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train /= 255
    x_test /= 255

    return x_test, y_test

def load_model(dataset="cifar10",model_type="basic",epochs=1, data_augmentation=True):
    if model_type.find("h5") >-1:
        model_path = model_type
    else:
        model_name = "{}_{}_{}_{}_model.h5".format(dataset, model_type, epochs, data_augmentation)
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_path = os.path.join(save_dir, model_name)
        model = None
    x_test, y_test = get_dataset(18)
    if os.path.isfile(model_path):
        print("Loading existing model {}".format(model_path))
        try:
            model = keras.models.load_model(model_path)
            if isinstance(model.layers[-2], keras.engine.training.Model):
                model = model.layers[-2]
                model.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])
        except Exception as e:
            print(e)
    return model, x_test, y_test


def _encodeString(txt,base):
    return str(int(txt, base))

# TODO: Move this into it's own file 
def _decodeString(n):
    base = len(strs)
    if n < base:
        return strs[n]
    else:
        return _decodeString(n//base) + strs[n%base]

# TODO: Move this into it's own module 
def _decode(dataset, model_type, epochs, extension=None):
    if not extension:
        extension = default_extension
        
    pictures_path = default_path.format(experiment_time)
    model, x_test, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    score = []
    for file in os.listdir(pictures_path):
        if file.endswith(".{}".format(extension)):
            path = "{}/{}".format(pictures_path,file)
            img = img_to_array(load_img(path))/palette
            img_class = np.argmax(model.predict(np.array([img]),verbose=0))
            index = file.index("_truth") -1
            real_class = int(file[index:index+1])
            steg_msg = lsb.reveal(path)
            logger.info("img {} decoded as {} stegano {}".format(file,img_class,steg_msg))
            
            score.append(real_class==img_class)

    decoding_score = np.mean(np.array(score))
    return decoding_score

# TODO: Gather Message
# TODO: Return Model
# TODO: Use Segmented Images 
def _encode(msg,dataset, model_type, epochs, base=10, keep_one=False, quality=100, attack_strength=2.0, extension=None):
    if not extension:
        extension = default_extension
    encoded_msg = _encodeString(msg, base)
    test_size = len(encoded_msg)
    model, x_test, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
    num_classes= 17
    combined = list(zip(x_test, y_test))
    random.shuffle(combined)
    x_test[:], y_test[:] = zip(*combined)
    #keep only correctly predicted inputs
    preds_test = np.argmax(model.predict(x_test,verbose=0), axis=1)
    ground_truth = y_test.argmax(axis=1)
    inds_correct_full = np.where(preds_test == y_test.argmax(axis=1))
    inds_correct = np.where(preds_test == y_test.argmax(axis=1))[0]
    x, y = x_test[inds_correct], y_test[inds_correct]
    x, y = x[:test_size], y[:test_size]

    targets = np.array(to_categorical([int(i) for i in encoded_msg], num_classes), "int32")    
    #print(targets)
    
    # if keep_one:
    #     x = np.repeat(np.array([x[0,:,:,:]]),y.shape[0], axis=0)
    #     y = model.predict(x)
    adv_x = craft_attack(model,x,y=targets, epsilon=attack_strength)
    yadv = np.argmax(model.predict(adv_x), axis=1)
    
    pictures_path = default_path.format(experiment_time)
    os.makedirs(pictures_path, exist_ok =True)
    os.makedirs("{}/ref".format(pictures_path), exist_ok =True)

    for i, adv in enumerate(adv_x):
        # we need ADV_X + ADV_Y returned 
        predicted = yadv[i]
        encoded = np.argmax(targets[i])
        truth = np.argmax(y[i])
        adv_path = "{}/{}_predicted{}_encoded{}_truth{}.{}".format(pictures_path,i,predicted,encoded,truth, extension)
        real_path = "{}/ref/{}.{}".format(pictures_path,i,extension)
        q = int(10-quality/100)
        save_img(adv_path,adv, compress_level=q)
        save_img(real_path,x[i], compress_level=q)

    return model

def run(dataset="cifar10",model_type="basic", epochs = 25, ex_id="SP1"):

    attack_name = "targeted_pgd"
    logger.info("running {} {} {}".format(dataset,model_type, attack_name))
    

    if RANDOM_SEED>0:
        random.seed(RANDOM_SEED)

    quality=100
    extension = "png"
    nb_runs = 1

    experiment_id = "SP1/1"
    
    scores = []
    l = 10
    for i in range(nb_runs):
        logger.info(i)
        msg2 = "".join([strs[random.randint(0,len(strs)-1)] for i in range(l)])
        model = _encode(msg2, dataset, model_type, epochs, quality=quality, attack_strength=1.,extension = extension)
        score = _decode( dataset, model_type, epochs ,extension = extension)
        scores.append(score)

    logger.info("{}:{}".format(experiment_id,np.array(scores).mean()))
    # have one file with one adv image, segmentation model, adv model, message 

    
if __name__ == "__main__":
    run(model_type="basic")