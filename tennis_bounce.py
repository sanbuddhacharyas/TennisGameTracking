import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, Dense, MaxPool2D, Flatten, Layer
from tensorflow.keras import metrics

from tensorflow.keras.applications.resnet50 import preprocess_input

import cv2
import numpy as np 
import random, string
import imutils



from PIL import Image, ImageDraw


class bouncedModel(Model):

    def __init__(self, bounced_network):
        super(bouncedModel, self).__init__()
        self.bounced_network = bounced_network
        self.loss_tracker    = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.bounced_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.bounced_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.bounced_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    

    def _compute_loss(self, data):

        input_image, gt_class = data  
        pred_class            = self.bounced_network(input_image)
        
        class_loss           = tf.keras.losses.CategoricalCrossentropy()(gt_class, pred_class)
        
        return class_loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
    

def bounced_detection(img_size):
    input_tensor        = Input(shape=(img_size[0], img_size[1], 1))
#     input_tensor        = preprocess_input(input_tensor)

    x            = Conv2D(16, 3, strides=(2,2), padding="same", activation='relu', name='Conv1')(input_tensor)
    x            = Conv2D(8,  3, strides=(2,2), padding="same", activation='relu', name='Conv2')(x)
    x            = MaxPool2D((2, 2))(x)
    x            = Flatten()(x)
    x            = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    x            = Dense(30,  activation='relu', kernel_initializer='he_uniform')(x)
    
    classification = Dense(2, activation='softmax', name='Classification')(x)
#     position       = Dense(2, activation='sigmoid', name='bounced')(x)
    
    encoder_model  = Model(inputs=input_tensor, outputs=classification)

    return encoder_model

def create_model(img_size, initial_learning_rate, decay_step):

    # create model
    model = bounced_detection(img_size)
    model = bouncedModel(model)

    #exponentialDecay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_step,
        decay_rate=0.96,
        staircase=True)
    
    #adam optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

    #compile model
    model.compile(optimizer=opt, metrics=['acc'])
    
    return model


def segment_ball_trajectory(blank_img, position, margin=15):

    temp_img  = blank_img.copy()
    PIL_image = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    draw = ImageDraw.Draw(PIL_image)
    draw.line(position, fill ="green", width = 4)
    del draw  

    temp_img = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    x = [i[0] for i in position]
    y = [i[1] for i in position]

    x_max = max(x) + margin
    y_max = max(y) + margin

    x_min = min(x) - margin
    y_min = min(y) - margin

    crop_area = (x_min, y_min, x_max, y_max)

    temp_img = temp_img[y_min:y_max, x_min:x_max]

    if (np.prod(temp_img.shape)!=0) and (temp_img != []):
        return cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY), (x_min, y_min), True

    else:
        return None, (None, None), False

def find_bouncing_point(img, point_detection_model, name=''):

    name = ''.join(random.choices(string.ascii_uppercase +
                                 string.digits, k=9))
    _, h, w, _    = img.shape
    pred         = point_detection_model.predict(img)[0]
    heatmap      = (pred / pred.max() * 255).astype(np.uint8)
    ret, heatmap = cv2.threshold(heatmap, 200, 255, cv2.THRESH_BINARY)
 
    y, x = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)

    image1 = cv2.cvtColor((img[0,:,:,0]*255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    image1 = cv2.circle(image1, (x,y),2,(255,0,0),-1)
    
    cv2.imwrite(f'test_image/{name}.jpg', image1)

    x, y = x / w, y / h

    return x, y


def find_countours(gray):
    
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    return edged, cnts

def keypoint_to_heatmap(heatmap):
    
    contour_area   = []
    contour_center = []
    
    ret, heatmap   = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
    countour, cnts = find_countours(heatmap)

    for c in cnts:
        
        area = cv2.contourArea(c)

        M  = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        contour_area.append(area)
        contour_center.append([cX, cY])

    if len(contour_area)!=0:
        ind = np.argmax(contour_area)
        return (contour_center[ind][0], contour_center[ind][1])

    else:
        return (None, None)

# def find_bouncing_point(all_ball_corrdinate, hit_range=25, reflection=40):
#     total_data = len(all_ball_corrdinate)

#     for i in range(total_data):
#         to_test_index = i + 3
#         start         = i
#         end           = i + 3

#         if to_test_index < total_data:
#             data   = all_ball_corrdinate[start:end]

#             xdata  = [i[0] for i in data]
#             ydata  = [i[1] for i in data]

#             test_x, test_y = all_ball_corrdinate[to_test_index]
            
#             p01   = (xdata[1] - xdata[0]) , (ydata[1] - ydata[0])
#             p12   = (xdata[2] - xdata[1]) , (ydata[2] - ydata[1])
#             trans = (test_x - xdata[2]) , (test_y - ydata[2])
            
#             direction1    = np.arctan2(p01[1], p01[0]) * 180 / np.pi
#             direction2    = np.arctan2(p12[1], p12[0]) * 180 / np.pi
#             direction     = np.arctan2(trans[1], trans[0]) * 180 / np.pi
            
            
#             d12  = (direction1 - direction2)
#             dnew = (direction2 - direction)
            
            
#             if (abs(d12) < hit_range) & (abs(dnew) >= reflection):
                
#                 #   check on two more data
#                 if (to_test_index + 1) < total_data:
#                         new_test_x, new_test_y = all_ball_corrdinate[to_test_index + 1]
#                         new_trans = (new_test_x - test_x) , (new_test_y - test_y)
#                         new_direction = np.arctan2(new_trans[1], new_trans[0]) * 180 / np.pi
                        
#                         if abs(direction - new_direction) < hit_range:
#                             return (xdata[2], ydata[2])

#     return None, None    