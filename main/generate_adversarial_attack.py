import sys
import os
import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.utils import save_img, load_img, img_to_array
from keras.datasets import cifar10   
import numpy as np
import random, time, os
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent
from PIL import Image
from stegano import lsb
import glob

sys.path.append("./")
from main import logger, RANDOM_SEED

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

experiment_time = int(time.time())
strs = "0123456789"
segmented_dir = "./results/cropped_segments/0001"
default_extension = "png"
palette = 256


def get_segmented_data():
    filelist = glob.glob(f'{segmented_dir}/*.png')
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    file_label = np.array([np.array(int(fname[-6:-4])) for fname in filelist])
    encoding_base = len(filelist)
    return encoding_base, x, file_label,

def _encodeString(txt,base):
    return str(int(txt, base))

def format_dataset():
    encoding_base,  x, file_label = get_segmented_data()
    x = x.astype('float32')
    x /= 255
    return encoding_base, x, file_label

def load_model(dataset, model_type,epochs, data_augmentation):
    if model_type.find("h5") >-1:
        model_path = model_type
    else:
        model_name = "{}_{}_{}_{}_model.h5".format(dataset, model_type, epochs, data_augmentation)
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_path = os.path.join(save_dir, model_name)
        model = None
    if os.path.isfile(model_path):
        print(f"Loading existing model {model_path}")
        try:
            model = keras.models.load_model(model_path)
            if isinstance(model.layers[-2], keras.engine.training.Model):
                model = model.layers[-2]
                model.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])
        except Exception as e:
            print(e)
    return model

def get_labels(model, x):
    y = model.predict(x)
    return y
    
    

def pre_process(model, x, y, test_size):
    combined = list(zip(x, y))
    x[:], y[:] = zip(*combined)
    preds_test = np.argmax(model.predict(x,verbose=0), axis=1)
    inds_correct = np.where(preds_test == y.argmax(axis=1))[0]
    x, y = x[inds_correct], y[inds_correct]
    x, y = x[:test_size], y[:test_size]
    return x,y

def craft_attack(model, x,y=None, epsilon=1., minimal=True):
    attack_params = {"norm": 2,'minimal': minimal,"targeted":True, "eps_step":0.1, "eps":2.}
    classifier = KerasClassifier(model=model)
    crafter = ProjectedGradientDescent(classifier)
    crafter.set_params(**attack_params)
    adv_x = crafter.generate(x,y,mask=x)
    return adv_x


def _encode(encoded_msg, model, x, y, quality, attack_strength, extension, file_label, num_classes, pictures_path):
    test_size = len(encoded_msg)
    q = int(10-quality/100)
    
    x, y = pre_process(model, x, y, test_size)
    targets = np.array(to_categorical([int(i) for i in encoded_msg], num_classes), "int32")    
    adv_x = craft_attack(model,x,y=targets, epsilon=attack_strength)
    adv_y = np.argmax(model.predict(adv_x), axis=1)
    
    os.makedirs(pictures_path, exist_ok =True)
    os.makedirs(f"{pictures_path}/ref", exist_ok =True)

    for i, adv in enumerate(adv_x):
        predicted = adv_y[i]
        encoded = np.argmax(targets[i])
        truth = np.argmax(y[i])
        adv_path = f"{pictures_path}/file{file_label[i]}_i{i}_predicted{predicted}_encoded{encoded}_truth{truth}.{extension}"
        real_path = f"{pictures_path}/ref/{i}.{extension}"
        save_img(adv_path,adv, compress_level=q)
        save_img(real_path,x[i], compress_level=q)

    return model

    
# # TODO: Move this into it's own file 
# def _decodeString(n):
#     base = len(strs)
#     if n < base:
#         return strs[n]
#     else:
#         return _decodeString(n//base) + strs[n%base]

# TODO: Move this into it's own module 
def _decode(model, epochs, file_label, pictures_path, extension=None):
    if not extension:
        extension = default_extension
        
    # pictures_path = default_path.format(experiment_time)
    # model, x_test, y_test = load_model(dataset=dataset, model_type=model_type, epochs=epochs)
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




def run(dataset, model_type, epochs, data_augmentation, pictures_path):

    attack_name = "targeted_pgd"
    logger.info("running {} {}".format(model_type, attack_name))
    random.seed(RANDOM_SEED)
    quality=100
    extension = "png"
    nb_runs = 1
    scores = []
    msg_length = 10
    num_classes= 100

    

    for i in range(nb_runs):
        logger.info(i)
        msg = "".join([strs[random.randint(0,len(strs)-1)] for i in range(msg_length)])
        encoding_base, x, file_label = format_dataset()
        encoded_msg = _encodeString(msg, encoding_base)
        init_model = load_model(dataset=dataset, model_type=model_type, epochs=epochs, data_augmentation=data_augmentation)
        y = get_labels(init_model, x)
        model = _encode(encoded_msg, init_model, x, y, quality=quality, attack_strength=1.,extension = extension, file_label=file_label, num_classes=num_classes, pictures_path='./results/pictures')
        score = _decode(init_model, epochs ,file_label, './results/pictures' ,extension = extension)
        scores.append(score)

    logger.info("{}".format(np.array(scores).mean()))
    # have one file with one adv image, segmentation model, adv model, message 

    
if __name__ == "__main__":
    run(dataset='cifar100', model_type="resnet", epochs=100, data_augmentation=True, pictures_path='./results/pictures')
    
    
