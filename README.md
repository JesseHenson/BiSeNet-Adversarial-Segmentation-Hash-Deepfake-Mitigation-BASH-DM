
# BiSeNet Adversarial Segmentation Hash - Deepfake Mitigation (BASH - DM)
1. Run pipenv install 
2. Run get_and_untar_dataset.py
3. Run experiements/segment_attack_cycle.py
4. Open Results
## Basic Application
1. Add an imperceptible watermark to an image. 
2. Produce a hash with the original image used to later authenticate the image against the now watermarked image. 
 

## The Paper 

Deep fakes have grown as a socio-political issue. Many government agencies including the CIA, the FBI and the Congress have put out bulletins explaining its impact on society. While the problem of detecting Deep fakes has had much success, it remains a cat and mouse game, where deep fakes become more sophisticated and detection methods must improve in suite. We propose a different solution to the larger problem of deep fakes and their impact in society. 
To circumvent the need to detect a deep fake, a method of authenticating a piece of media will need to be implemented. Our proposed solution illustrates a method of doing such a thing. To produce authentication of an image, a segmentation model is used to segment an image of a face. This image is then run through an algorithm to add a watermark, which allows for later retrieval and reversal. Finally, the watermarked image is run through the same segmentation model, producing a slightly different segment map. These two segmentation maps are then compared and the hashed difference is output to the console. 
By producing a repeatable watermark for a particular image, the image can be reversed to it’s original form. This gives the producer of the media the ability to authenticate that nothing within the image has been modified. By providing this authentication we solve two very important problems associated with deep fakes. 
First, the problem of societal trust is mitigated. By allowing producers of media to authenticate their own images, the burden of proof lies back on the media producer. Second, the ability for criminals to blackmail a political or social figure is mitigated by the inability of those criminals to authenticate the image. 
We were able to provide one solution that provides full image coverage at a consistent protection. However, using multiple methods or different levels of protection across the image would be more performant. Using a more protective model only in areas likely to be modified by sophisticated deep fakes processes may improve the performance and protection greatly. 

> Paper to come shortly!!!
