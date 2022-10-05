
# Segmentation_Adversarial_Watermarking_for_Deepfakes
=======
1. Run pipenv install 
2. Run get_and_untar_dataset.py
3. Run experiements/segment_faces.py
4. Run experiments/generate_adversarial_attack.py
5. Open experiments/3_0/pictures

## Known Issues 
1. Image output of segment_faces.py in 512x512, input of SATA is 32x32. 
- Resolution: Since segmented image is mostly black with detected object only, using a max and min to crop around 32x32 of object should give a solid result. Since most of the parts are less than or close to 32x32, most of the object should be shown. 
- Caveat: 
1. We'll need to know exactly location and only place the pixels that are not blackened by segmentation process 
2. We'll need to attack the image, then reset black areas to 0 prior to back classification. This way the classifier only sees the segment + full black background 
3. We'll need to create a method that adds the attacked segment back to the original image without modifying any other piece of the original image. 

## TODO: 
1. Modify the segment_faces.py file to crop the segment image to 32x32 from min to max of the image 
2. Test the segments on classifier
3. Add a function to piece the original image + augmented segments back together. 