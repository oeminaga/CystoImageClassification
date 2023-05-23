#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import tensorflow as tf
import cv2
from collections import defaultdict
import numpy as np
from PIL import Image
import pandas as pd
from PIL.ImageFilter import (
   EDGE_ENHANCE, SHARPEN
)

from plexusnet.architecture import LoadModel

#%%
##############################################
#
#   CHANGE AFTER THIS POINT
#
##############################################
# CONFIGURATION
verbose=1000
# IF verbose = 0, then it will not print anything
# IF verbose = 1, then it will print the progress
# IF verbose = 1000, then it will print the progress, show salincy map and the frame output
# IF verbose = 100, then it will print the progress and show only salincy maps

FOLDER_WITH_IMAGES_IN_CLASSES="./Study" #WHERE YOU STORE THE IMAGES
classes_to_consider=[f for f in os.listdir(FOLDER_WITH_IMAGES_IN_CLASSES) if f!=".DS_Store"]
#####
#
#   STUDY FOLDER
#       \_ CLASS 1
#       \_ CLASS 2
#       \_ CLASS 3
#       \_ CLASS 4
#
# OR YOU CAN JUST PROVIDE A LIST OF CLASSES [CLASS 1, CLASS 2, CLASS 3, CLASS 4]
#####
source = "/PCIe/DevelopmentSet_CystoNet/" #WHERE YOU STORE THE VIDEOS

consider_folder = "ExternalValidation" #THE SUBFOLDER OF THE SOURCE
outcomes = defaultdict(list)
color_space={"positive": (0, 255, 0), "negative": (0, 255, 0)} #COLOR CODE FOR A POSITIVE OR NEGATIVE FRAME , format (Red, Green, Blue)
MEAN =[0,0,0]
STD= [1,1,1]
directory_to_store_outcome="./ExternalValidationExternalValidationCVAT_MobileNet_2nd_HEATMAPS" #WHERE YOU WANT TO STORE THE OUTCOME
print("classes_to_consider",classes_to_consider) #CHECK IF THE CLASSES ARE CORRECT AS THEY CAN BE DIFFERENCES BECAUSE OF THE OPERATING SYSTEM.
directory_with_the_ground_truth="/home/eminaga/ExternalValidation/ExternalValidation_2022_06_15"
#%%
#HERE DEFINE YOUR MODEL
model = LoadModel("./weights_mobilenet_v3/model.h5")
model.summary() 
##############################################
#
#   DO NOT CHANGE AFTER THIS POINT
#
##############################################
import keras.backend as K

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def visualize_class_activation_map(model, original_img,target_class = 1):
        width, height, _ = original_img.shape

        #Reshape to the network input shape (3, w, h).
        img = original_img#np.array((np.float32(original_img))
        
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-2].get_weights()[0]
        final_conv_layer = get_output_layer(model, "conv_pw_13")
        get_output = K.function([model.layers[0].input], \
                    [final_conv_layer.output,model.layers[-1].output])
        [conv_outputs, predictions] = get_output(np.array([img]))
        conv_outputs=conv_outputs[0, :, :, :]
        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        for i, w in enumerate(class_weights[:, target_class]):
                t=conv_outputs[ :, :,i]
                cam += w * t#(t-np.min(conv_outputs[ :, :,i]))/(np.max(conv_outputs[ :, :,i])-np.min(conv_outputs[ :, :,i]))
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        #heatmap[np.where(cam < 0.2)] = 0
        super_imposed_img = cv2.addWeighted(heatmap, 0.5, original_img, 0.5, 0)
        return super_imposed_img,predictions

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.applyColorMap((heatmap*255).astype("uint8"), cv2.COLORMAP_JET)

    heatmap=cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    super_imposed_img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
    return super_imposed_img

def AddInformationToVideoFrame(frame, labels):
    for i,label in enumerate(labels):
        prediction = labels[label]
        image_data = cv2.putText(frame, f'P: {str(label)} ({prediction:.2f})', (5,20*(1+i)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1, cv2.LINE_AA)
    return image_data
def Sharpness(bgr):
            kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
            bgr = cv2.filter2D(src=bgr, ddepth=-1, kernel=kernel)
            return bgr
 
def get_ROI(img):
            """
            get ROI of the image
            """

            img = cv2.GaussianBlur(img, (45, 45),sigmaX=20)#45
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray= cv2.copyMakeBorder(gray.copy(),10,10,10,10,cv2.BORDER_CONSTANT,value=0)

            _,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)

            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            x,y,w,h=0,0,0,0
            
            for c in cnts:
                x_,y_,w_,h_ = cv2.boundingRect(c)
                if w<w_:
                    x,y,w,h = x_,y_,w_,h_
            
            return x,y,w,h

from tqdm import tqdm
for folder in os.listdir(source):
    if folder !=consider_folder and consider_folder != None:
        continue
    files = [f for f in os.listdir(f"{source}/{folder}") if f.endswith("mp4")]
    for fl in tqdm(files):
        fil=fl.split(".")[0]
        annotation=pd.read_csv(f"{directory_with_the_ground_truth}/{fil}-cys.csv")
        cap = cv2.VideoCapture(f"{source}/{folder}/{fl}")
        ret, frame = cap.read()
        fps_=cap.get(cv2.CAP_PROP_FPS)
        counter=1

        result = {}
        result["FrameId"]=[]
        for k in classes_to_consider:
            result[k]=[]
        while(1):
            ret, frame = cap.read()
            if frame is None:
                break
            frm=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img=Image.fromarray(frm)
            img = img.filter(SHARPEN)
            frm = np.array(img.filter(EDGE_ENHANCE))
            
            x,y,w,h=get_ROI(np.array(frm).astype(np.uint8))
            x_clip =w//4
            y_clip = h//6
            x_start = x + x_clip
            y_start = y + y_clip
            h = w - 2* w//4
            w = w - 2*w//4
            
            frm=frm[y_start:h+y_start,x_start:w+x_start]

            if (frm.shape[0]==0 or frm.shape[1]==0):
                for k in classes_to_consider:
                    result[k].append(-1)
                result["FrameId"].append(counter)
                counter+=1
                continue
            
            #frmb = sr.upsample(frm)
            if verbose>1000:
                frmb = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'{fl}-{folder}_C',frmb)


            frm=cv2.resize(frm,(300,300))#(224,224))
            heatmap,output=visualize_class_activation_map(model, frm, target_class=6)
            
            result["FrameId"].append(counter)
            for k, p in zip(classes_to_consider,  output[0]):
                result[k].append(p)

            preds=np.argsort(output,axis=1,)[:,-2:]

            labels = {}

            for p in preds[0]:
                label=classes_to_consider[p]
                labels[label]=output[0,p]
            labels={k: v for k, v in sorted(labels.items(), key=lambda item: item[1], reverse=True)}

            if verbose>100:
                cv2.imshow(f'{fl}-{folder}-B',heatmap)
            
            frame = cv2.rectangle(frame, (x_start,y_start), (x_start+w, y_start+h), (255, 0, 0), 2)
            frame=AddInformationToVideoFrame(frame, labels)

            
            if verbose>100:
                lab=annotation.iloc[counter-1]["true_label"]
                frame=cv2.putText(frame, lab, (x_start,y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(f'{fl}-{folder}',frame)
                k = cv2.waitKey(round(fps_)) & 0xff

                if k == 27:
                    break
            
            counter+=1
        f_x=fl.split(".")[0]
        fl_csv = f"{f_x}.csv"
        pd.DataFrame(result).to_csv(f"{directory_to_store_outcome}/{fl_csv}", index=False)
        cap.release()
        if verbose>100:
            cv2.destroyAllWindows()