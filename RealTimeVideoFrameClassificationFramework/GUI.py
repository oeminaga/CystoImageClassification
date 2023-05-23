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
from collections import namedtuple
from typing import Union
from numba import njit
import numpy as np
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import time
from utils import Function, ImageLoader, ConvertPixelVerticiesToWorldCoordinate
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def DrawText(x, y, text, font, width, height):
    textSurface = font.render(text, True, (255, 255, 66, 255), (0, 0, 0, 0))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    y_ = height - (y + textSurface.get_height())
    glWindowPos2d(x, y_)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, textData)
    glFlush()


def get_ROI(img):
    """
    docstring
    """
    img = cv2.GaussianBlur(img, (45, 45), sigmaX=20)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray.copy(), 10, 10, 10,
                              10, cv2.BORDER_CONSTANT, value=0)

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_TC89_KCOS)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    x, y, w, h = 0, 0, 0, 0

    for c in cnts:
        x_, y_, w_, h_ = cv2.boundingRect(c)
        if w < w_:
            x, y, w, h = x_, y_, w_, h_

    return x, y, w, h


@njit
def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    else:
        return -1


@njit
def Draw(verticies, edges, colors):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv(colors[vertex])
            glVertex(verticies[vertex])
    glEnd()
    glFlush()


class Button(object):
    def __init__(self, name="", type_name="button", position=(0, 0), size=(10, 10), IsClickedColor=(1.0, 1.0, 0), ReleaseClickColor=(1.0, 1.0, 1.0), Command: Function = None, *args, **kwargs) -> None:
        self.command = Command
        self.name = name
        self.type_name = type_name
        self.position = position
        self.size = size
        self.arguments = args
        self.StatusExecute = False
        self.IsClickedColor = IsClickedColor
        self.ReleaseClickColor = ReleaseClickColor
        self.window_shape = None

    def Draw(self):
        self.vertices = (
            (self.position[0], self.position[1], 0),
            (self.position[0], self.position[1]+self.size[1], 0),
            (self.position[0]+self.size[0], self.position[1]+self.size[1], 0),
            (self.position[0]+self.size[0], self.position[1], 0)
        )
        w_height, w_width, _ = self.window_shape
        self.vertices = ConvertPixelVerticiesToWorldCoordinate(
            self.vertices, w_height, w_width)
        self.edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        )
        if self.StatusExecute:
            self.colors = (
                self.IsClickedColor,
                self.IsClickedColor,
                self.IsClickedColor,
                self.IsClickedColor
            )
        else:
            self.colors = (
                self.ReleaseClickColor,
                self.ReleaseClickColor,
                self.ReleaseClickColor,
                self.ReleaseClickColor
            )
        Draw(self.vertices, self.edges, self.colors)

    @njit
    def OnClick(self, mous_coordination):
        a = Rectangle(self.position[0], self.position[0],
                      self.position[0]+self.size[0], self.position[1]+self.size[1])
        b = Rectangle(mous_coordination[0], mous_coordination[1],
                      mous_coordination[0]+10, mous_coordination[1]+10)
        x = area(a, b)
        if x > 0:
            self.StatusExecute = not self.StatusExecute


class WindowVisualization:
    def __init__(self, buttons: list, Commands: list, video_source: str or int) -> None:
        se_ = cv2.VideoCapture(video_source)
        self.fps_ = se_.get(cv2.CAP_PROP_FPS)
        self.width = int(se_.get(3))
        self.height = int(se_.get(4))
        se_.release()
        se_ = None
        # cv2.VideoCapture(video_source)
        self.cap = CamGear(source=video_source).start()
        self.IsStarted = False
        self.buttons = buttons

        pygame.init()
        display = (1920, 1080)  # (self.width,self.height)
        self.screen = pygame.display.set_mode(
            display, DOUBLEBUF | OPENGL | FULLSCREEN)
        pygame.display.set_caption("AI-based Framework for Cystoscopy")

        self.im_loader = ImageLoader(0, 0)
        glClearColor(0.7, 0, 0, 1)
        self.Commands = Commands

    def GetScreenShot(self):
        screen = pygame.display.get_surface()
        size = screen.get_size()
        buffer = glReadPixels(0, 0, *size, GL_RGB, GL_UNSIGNED_BYTE)
        screen_surf = pygame.image.fromstring(buffer, size, "RGB")
        screen_surf = pygame.transform.flip(screen_surf, False, True)
        data = pygame.image.tostring(screen_surf, 'RGB')
        image = Image.frombytes('RGB', size, data)
        return np.array(image).astype(np.uint8)

    def Run(self):
        counter = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    for btn in self.buttons:
                        btn.command.get_result()

                    for cmd in self.Commands:
                        cmd.get_result()
                    self.cap.stop()
                    pygame.quit()
                    quit()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            try:
                glOrtho(0, self.width, self.height, 0., 0, 1.)
            except:
                pass
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            glDisable(GL_DEPTH_TEST)
            image = self.cap.read()

            if image is not None:
                self.im_loader.load(image)
                counter += 1
            else:
                for btn in self.buttons:
                    btn.command.get_result()
                for cmd in self.Commands:
                    cmd.get_result()
                self.cap.stop()  # release()
                pygame.quit()
                quit()
                break
            glColor3f(1, 1, 1)
            self.im_loader.draw()

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()

            gluPerspective(45, (self.width / self.height),
                           0.1, 10.0)  # (width / height)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            glTranslate(0.0, 0.0, -5)

            glEnable(GL_DEPTH_TEST)
            mouse = pygame.mouse.get_pos()

            for i in range(len(self.buttons)):
                if pygame.mouse.get_pressed()[0] == 1:
                    self.buttons[i].OnClick(mouse)
                if self.buttons[i].StatusExecute:
                    self.buttons[i].command.execute(image, counter)
                    self.buttons[i].command.Draw()
                self.buttons[i].window_shape = image.shape
                self.buttons[i].Draw()

            for cmd in self.Commands:
                if cmd.name == "VideoRecoder":
                    frm_ = self.GetScreenShot()
                    frm_ = cv2.cvtColor(frm_, cv2.COLOR_RGBA2BGR)
                    cmd.execute(frm_, counter)
                else:
                    cmd.execute(image, counter)
                cmd.Draw()
            pygame.display.flip()
