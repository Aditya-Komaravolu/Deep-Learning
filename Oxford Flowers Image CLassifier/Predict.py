
#!/usr/bin/python


import sys

arguements_list=sys.argv    
image_path=sys.argv[1]
saved_keras_model_filepath=sys.argv[2]

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json





with open('label_map.json', 'r') as f:
    class_names = json.load(f)



reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath)

reloaded_keras_model.summary()



# TODO: Create the process_image function
def process_image(image):
    image=tf.convert_to_tensor(image)
    image=tf.image.resize(image,(224,224))
    #image=np.expand_dims(image,axis=0)
    image=tf.cast(image,tf.float32)
    image/=255
    image=image.numpy()
    return image

def process_image_1(image):
    image=tf.convert_to_tensor(image)
    image=tf.image.resize(image,(224,224))
    #image=np.expand_dims(image,axis=0)
    image=tf.cast(image,tf.float32)
    image/=255
    image=image.numpy()
    return image


from PIL import Image

im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image_1(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()


from PIL import Image
def predict(image_path,model,top_k=4):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    test_image=process_image(test_image)
    test_image=tf.convert_to_tensor(test_image)
    test_image=np.expand_dims(test_image,axis=0)

    
    log_probs = model.predict(test_image)
    probe,classes= tf.math.top_k(log_probs,k=4)
    probe=probe.numpy().flatten()
    classes=classes.numpy().flatten()
    return probe,classes

probs,classes = predict(image_path,reloaded_keras_model,4) 
print(probs)
print(classes)
flowers = [class_names[str(x+1)] for x in classes]
print(flowers)




