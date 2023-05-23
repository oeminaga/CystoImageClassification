'''
Copyright by Okyaz Eminaga 2023
'''
import pygame
from pygame.locals import *
import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from collections import defaultdict
from models import ModelBasic
import pandas as pd
import yaml
import os
from datetime import datetime
from vidgear.gears import WriteGear
from numba import njit
# function 
def ConvertPixelTo2DCart(pixel_coordinate, height, width):
    to_consider=min([height, width])
    center = to_consider/2
    xPix, yPix =pixel_coordinate

    yPix = height- yPix
    xClip = (width/center+0.1)*(2*(xPix/ width)-1)#-0.1#-(center[0]/width) #- 1.0
    yClip =(height/center+(0.1*height/width))*(2*(yPix/ height)-1)#+(0.1*height/width)#-(center[1]/height)
    return [xClip,yClip]
def ConvertPixelVerticiesToWorldCoordinate(verticies, height, width):
    verticies_world_coordinate = []
    for vertex in verticies:
        
        coordinates =ConvertPixelTo2DCart(vertex[0:2],  height, width)
        
        if len(vertex)==3:
            coordinates.append(vertex[2])
        verticies_world_coordinate.append(coordinates)
    return verticies_world_coordinate

# class
class Function:
    def __init__(self, name, color, thickness, edges) -> None:
        self.name = name
        self.color = color
        self.thickness = thickness
        self.edges = edges
        self.results = defaultdict(list)
        self.IsRunning = False
    def execute(self, frame,index):
        if self.IsRunning is False:
            self.IsRunning =True
        self.frame=frame
        self.index =index
        pass
    def stop(self):
        #self.IsRunning=False
        pass
    def start(self):
        pass
    def Draw(self):
        pass
    def get_result(self):
        pass
class VideoStreamManagement:
    def __init__(self) -> None:
        with open('./config.yml', 'r') as file:
            self.configuration = yaml.safe_load(file)
        if not os.path.exists(self.configuration["to_store"]):
            os.makedirs(self.configuration["to_store"], exist_ok=True)
        self.video_filename=""
        self.result_filename = ""
        self.screen_shot_filename_template= ""
        self.path_to_save = self.configuration["to_store"]
        self.case_id = ""
        self.case_path = ""
        self.case_type = ""
        self.model_infos=self.configuration["model"]
        
    def new(self):
        print('Enter the case id (No space!):')
        self.case_id = input()
        run =True
        while run:
            print('Enter the procedure type(O or C):')
            self.case_type = str(input()).upper()
            if self.case_type in ["O","C"]:
                run=False
        self.DateTime =datetime.now().strftime("%m%d%Y")
        self.case_path= f"{self.path_to_save}/{self.case_id}_{self.case_type}_{self.DateTime}/"
        
        os.makedirs(self.case_path, exist_ok=True)
        for itm in ["Images", "Videos", "AI"]:
            os.makedirs(f"{self.case_path}/{itm}", exist_ok=True)
        self.screen_shot_filename_template= f"{self.case_path}/Images/{self.case_id}_{self.case_type}_{self.DateTime}"
        self.video_filename= f"{self.case_path}/Videos/{self.case_id}_{self.case_type}_{self.DateTime}"
        self.result_filename= f"{self.case_path}/AI/{self.case_id}_{self.case_type}_{self.DateTime}"
    def close(self):
        pass
class ModePrediction(Function):
    def __init__(self, name, filename, color, thickness, edges, model:ModelBasic, preprocess = None, input_size=(300,300),TypeOfDetectionProblem="single_frame_classification",ShowOnlyPositiveAlert=True,PlaySound=False,postprocess=None, resolution=[1920,1080]) -> None:
        '''
        TypeOfDetectionProblem: single_frame_classification, sequence_frame_classification, single_frame_object_detection, single_frame_segmentation
        '''
        self.name = name
        self.color = color
        self.thickness = thickness
        self.edges = edges
        self.results = defaultdict(list)
        self.model = model
        self.IsRunning=False
        self.TypeOfDetectionProblem= TypeOfDetectionProblem #TypeOfDetectionProblem
        self.stopped=False
        self.filename=filename
        self.preprocess =preprocess
        self.postprocess = postprocess
        self.input_size=input_size
        self.labels =[False]
        self.ShowOnlyPositiveAlert = ShowOnlyPositiveAlert
        #pygame.mixer.init()
        #pygame.mixer.set_num_channels(8)
        #self.voice = pygame.mixer.Channel(5)
        #self.alert_sound = pygame.mixer.Sound("BEEPAppl_Detector 1 (ID 2251)_BSB.wav")
        #self.PlaySound = PlaySound
        #self.resolution=resolution
    def execute(self, frame, index):
        self.IsRunning=True
        self.frame=frame
        self.stopped=False
        
        if self.stopped == False and self.IsRunning:
            
            if self.preprocess:
                frm, self.rectangle=self.preprocess(frame)
            else:
                frm=frame
                
            if (frm.shape[0]==0 or frm.shape[1]==0):
                self.results["Pred"].append(-1)
                self.results["Label"].append(False)
                self.results["FrameID"].append(index)
                if self.TypeOfDetectionProblem=="single_frame_object_detection":
                    self.results["Coordinate"].append(-1)
                self.color = (0,255,0)
                self.thickness = 2
                
                return
            frm=cv2.resize(frm.astype("uint8"),self.input_size)
            if self.TypeOfDetectionProblem=="single_frame_object_detection":
                predict_, label_, coordinates = self.model.Prediction(frm)
                for predict_, label, coordinte in zip(predict_, label_,coordinates):
                    self.results["Pred"].append(predict_)
                    self.results["Label"].append(label)
                    self.results["FrameID"].append(index)
                    self.results["Coordinate"].append(coordinte)
                self.coordinates=coordinates
                self.labels = label_
            if self.TypeOfDetectionProblem in ["single_frame_classification", "sequence_frame_classification"]:
                predict_, label_, heatmaps_= self.model.Prediction(frm)
                for predict_, label in zip(predict_, label_):
                    self.results["Pred"].append(predict_)
                    self.results["Label"].append(label)
                    self.results["FrameID"].append(index)
                self.labels = label_
            if self.TypeOfDetectionProblem in ["single_frame_segmentation"]:
                #TODO: Add this function
                print("Not implemented")
                if self.postprocess:
                    pass
                pass
    def Draw(self):
        if self.frame is None:
            return
        height,width, _=self.frame.shape
        if self.TypeOfDetectionProblem in ["single_frame_classification", "sequence_frame_classification"]:
            rectangle=self.rectangle
            if rectangle[2]==0 or rectangle[3]==0:
                return
            vertices= ((rectangle[0],rectangle[1],0),
                    (rectangle[0],rectangle[1]+rectangle[3],0),
                    (rectangle[0]+rectangle[2],rectangle[1]+rectangle[3],0),
                    (rectangle[0]+rectangle[2], rectangle[1],0)
                    )
            for label in self.labels:
                if label:
                        #if self.PlaySound:
                        #    if not self.voice.get_busy():
                        #        self.voice.play(self.alert_sound)
                        color = (255,0,0)
                        thickness = 6
                else:
                        #if self.PlaySound:
                        #    self.voice.stop()
                        color = (0,255,0)
                        thickness = 2
                vertices=ConvertPixelVerticiesToWorldCoordinate(vertices, height, width)
                glLineWidth(thickness)
                glBegin(GL_LINES)

                colors=(
                    color,
                    color,
                    color,
                    color
                )
                if self.ShowOnlyPositiveAlert:
                    if label:
                        for edge in self.edges:
                            for vertex in edge:
                                glColor3fv(colors[vertex])
                                glVertex(vertices[vertex])
                else:
                    for edge in self.edges:
                            for vertex in edge:
                                glColor3fv(colors[vertex])
                                glVertex(vertices[vertex])
                glEnd()
                glFlush()
        
        if self.TypeOfDetectionProblem=="single_frame_object_detection":
            for (coordinate, label) in zip(self.coordinates, self.labels):
                vertices=ConvertPixelVerticiesToWorldCoordinate(coordinate, height, width)
                if label:
                    color = (255,0,0)
                    thickness = 6
                else:
                    color = (0,255,0)
                    thickness = 2
                glLineWidth(thickness)
                glBegin(GL_LINES)
                
                colors=(
                color,
                color,
                color,
                color
                )
                for edge in self.edges:
                    for vertex in edge:
                        glColor3fv(colors[vertex])
                        glVertex(vertices[vertex])
                glEnd()
                glFlush()
        return super().Draw()
    def stop(self):
        self.stopped=True
        return super().stop()
    def get_result(self):
        self.IsRunning=False
        pd.DataFrame(self.results).to_csv(f"{self.filename}.csv", index=False)
class VideoRecoder(Function):
    def __init__(self, filename, shape, fps, name="VideoRecoder", color=None, thickness=None, edges=None) -> None:
        self.filename = filename
        self.shape= shape
        self.fps = fps
        self.out =WriteGear(f"{self.filename}.mp4")
        #self.out.write(frame)
        #self.out.close()
        #cv2.VideoWriter_fourcc(*'XVID')
        #self.out = cv2.VideoWriter(f"{self.filename}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps,self.shape)
        self.IsRunning=True
        self.stopped = False
        super().__init__(name, color, thickness, edges)
    def execute(self, frame, index=-1):
        self.out.write(frame)
        return True
    def stop(self): 
        self.stopped=True       
        return super().stop()
    def start(self):
        self.stopped=False
        return super().start()
    def get_result(self):
        self.IsRunning=False
        self.out.close()#release()
        return True
class ScreenShot(Function):
    def __init__(self, name="ScreenShot", filename="", color=None, thickness=None, edges=None) -> None:
        self.filename=filename
        super().__init__(name, color, thickness, edges)
    def execute(self, frame, index):
        cv2.imwrite(f"{self.filename}_{index}.png", frame)
        return super().execute(frame, index)
    def get_result(self):
        return super().get_result()
class ImageLoader:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 0
        self.height = 0
        self.img_data = 0
        self.Texture = glGenTextures(1)

    def load(self, image):
        im = image
        tx_image = cv2.flip(im, 0)
        tx_image = cv2.flip(tx_image, 1)
        tx_image = Image.fromarray(tx_image)
        self.width = tx_image.size[0]
        self.height = tx_image.size[1]
        self.img_data = tx_image.tobytes('raw', 'BGR', 0, -1)

        
        glBindTexture(GL_TEXTURE_2D, self.Texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.img_data)

    def draw(self):
        glEnable(GL_TEXTURE_2D)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslate(self.x, self.y, 0)
        glBegin(GL_QUADS)
        glVertex(0, 0, 0)
        glTexCoord2f(0, 0)
        glVertex(self.width, 0, 0)
        glTexCoord2f(0, 1)
        glVertex(self.width, self.height, 0)
        glTexCoord2f(1, 1)
        glVertex(0, self.height, 0)
        glTexCoord2f(1, 0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glFlush()
