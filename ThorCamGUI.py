# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:31:39 2021

@author: HIROKI-TAKAHASHI
"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import sys
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from pyqtgraph.Qt.QtWidgets import *
from astropy.modeling import models,fitting #used for data fitting
from astropy.modeling.models import custom_model #used for data fitting
import math
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

@custom_model #custom model for 1D Gaussian with an offset
def offsetted_gaussian(x,amplitude=1.,sigma=1,mean=1, offset =1):
    return (amplitude*np.exp(-.5 * (x-mean)**2/(sigma**2))+offset)

def beamprofile(array):
        #create list of coordinates on the CCD sensor
        xlist = np.linspace(0,1440,1440)*3.45*10**(-3)
        ylist = np.linspace(0,1080,1080)*3.45*10**(-3)
        
        #process the array for model fitting and plotting
        
        #integrate the image along horizontal axis
        horizontal_projection = array.sum(axis=0)
        #integrate the image along horizontal axis
        vertical_projection= array.sum(axis=1)
        #normalize the data
        normalizedH = horizontal_projection/np.max(horizontal_projection)
        normalizedV = vertical_projection/np.max(vertical_projection)

        #Gaussian fit to the data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            g_init =offsetted_gaussian(amplitude=1.,sigma=1,mean=1440/2*3.45*10**(-3), offset =.1)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init,ylist,normalizedV)
            w_y =round(np.sqrt((2*g.sigma.value)**2),4) #vertical beam waist

            gx_init = offsetted_gaussian(amplitude=1.,sigma=1,mean=1440/2*3.45*10**(-3), offset =.1)
            fit_gx = fitting.LevMarLSQFitter()
            gx = fit_g(gx_init,xlist,normalizedH)
            w_x =round(np.sqrt((2*gx.sigma.value)**2),4) #horizontal beam waist
        return w_x,w_y, gx, g, normalizedH,normalizedV

class ThorCamWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCameras()
        self.w_x =0 #initialize beam waist
        self.w_y=0 #initialize beam waist
        self.listH=[] #initialize list for horizontal projection
        self.listV=[] #initialize list for vertical projection
        self.fitH=offsetted_gaussian(amplitude=1.,sigma=1,mean=1440/2*3.45*10**(-3), offset =.1) #initialize the fit
        self.fitV=offsetted_gaussian(amplitude=1.,sigma=1,mean=1440/2*3.45*10**(-3), offset =.1) #initialize teh fit
        if self.camera:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.updateImage)
            self.timer.start(10) # update image at every 10ms
    
    def initUI(self):
        self.setWindowTitle('Thorlab Camera Interface')
        self.resize(1920,1080)
        
        self.labl=QtGui.QLabel(self) #text shown on the window
        self.labl.move(1650,200)
        self.labl.setText(" w_x: \n w_y: ")
        self.labl.setStyleSheet("QLabel { color : white; }")
        
        #text holder for the beam waist value
        self.w_x_text = QtGui.QLabel(self) 
        self.w_x_text.move(1690,200)
        self.w_x_text.setText("not measured yet")
        self.w_x_text.setStyleSheet("QLabel { color : white; }")
        
        self.w_y_text = QtGui.QLabel(self)
        self.w_y_text.move(1690,220)
        self.w_y_text.setText("not measured yet")
        self.w_y_text.setStyleSheet("QLabel { color : white; }")

        self.gaussButton()
        self.projBtn()
        self.button()
        
        self.plot = pg.PlotItem()
        self.plot.setAspectLocked(True)
        self.setCentralWidget(self.plot)
        self.plot.showAxis('left')
        self.img = pg.ImageItem(border='w')
        self.plot.addItem(self.img)
        self.show()

    
    def initCameras(self):
        self.sdk = TLCameraSDK()
        self.available_cameras = self.sdk.discover_available_cameras()
        if len(self.available_cameras) < 1:
            print("no cameras detected")
            self.camera = None
            return
        self.camera = self.sdk.open_camera(self.available_cameras[0])
        self.camera.exposure_time_us = 110  # set exposure to .11 ms
        self.camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        self.camera.image_poll_timeout_ms = 1000  # 1 second polling timeout
        self.camera.roi = (0, 0, 1440, 1080)  # set roi to the whole region of the CCD camera
        self.camera.arm(2)
        self.camera.issue_software_trigger()
    
    def updateImage(self):
        # get a frame from the camera and update the image
        frame = self.camera.get_pending_frame_or_null()
        if frame is not None:
            self.image_buffer = np.copy(frame.image_buffer)
            self.img.setImage(np.transpose(self.image_buffer))
    
    def closeEvent(self, event):
        # override the closing behaviour
        self.sdk.__del__()
        self.timer.stop()
        event.accept()

    def button(self):
    	qbtn = QPushButton('Get Beam Waist', self)
    	qbtn.resize(200,100)
    	qbtn.move(1650,800)
    	qbtn.clicked.connect(self.buttnpress)
    	
    def buttnpress(self):
        self.w_x , self.w_y, self.fitH, self.fitV, self.listH, self.listV=beamprofile(self.image_buffer)
        self.w_x_text.setText(str(self.w_x)+"mm")
        self.w_y_text.setText(str(self.w_y)+"mm")

    def gaussButton(self):
        gbtn = QPushButton('Gaussian Fit window', self)
        gbtn.resize(200,100)
        gbtn.move(1650,650)
        gbtn.clicked.connect(self.openGaussianWindow)
        
    def openGaussianWindow(self):
        self.w_x , self.w_y, self.fitH, self.fitV, self.listH, self.listV=beamprofile(self.image_buffer)
        openGauss = gaussianWindow(self.listH,self.listV, self.fitH, self.fitV)
        openGauss.gaussianPlot()

    def projBtn(self):
        pbtn = QPushButton('Vertical/Horizontal\n Projection', self)
        pbtn.resize(200,100)
        pbtn.move(1650,500)
        pbtn.clicked.connect(self.openProj)
        
    def openProj(self):
        self.w_x , self.w_y, self.fitH, self.fitV, self.listH, self.listV=beamprofile(self.image_buffer)
        openProj = projectionWindow(self.listH,self.listV, self.fitH, self.fitV)
        openProj


class gaussianWindow(QWidget):
    def __init__(self,listH,listV,fitH,fitV):
        super().__init__()

        layout = QVBoxLayout()
        self.setWindowTitle('Thorlab Camera Interface')
        self.resize(500,500)
        self.setLayout(layout)

        self.listVertical = listV
        self.listHorizontal = listH
        self.horizontalFit = fitH
        self.verticalFit=fitV

        self.w0_y = np.sqrt((2*fitV.sigma.value)**2)
        self.w0_x = np.sqrt((2*fitH.sigma.value)**2)
        self.xlist = np.linspace(0,1440,1440)*3.45*10**(-3)
        self.ylist = np.linspace(0,1080,1080)*3.45*10**(-3)

        self.show()

    def checkbox(self):
        self.fitOn = QCheckBox("Show Curve Fit")
        self.fitOn.setChecked(False)
        self.fitOn.resize(100,50)
        self.fitOn.move(800,400)
        self.fitOn.stateChanged.connect(self.gaussianPlot)
        self.show()

    def gaussianPlot(self):
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        plt.plot(self.ylist,self.listVertical,'x')
        plt.title('Vertical Projection')
        plt.plot(self.ylist,self.verticalFit(self.ylist),'r')
        x1,y1 = [self.verticalFit.mean.value-self.w0_y,self.verticalFit.mean.value+self.w0_y],[1/math.e**2+self.verticalFit.offset, 1/math.e**2+self.verticalFit.offset]
        plt.plot(x1,y1)
        plt.xlim([0,3.73])
        plt.xlabel('Coordinate on CCD in mm')
        plt.ylabel('Normalized intensity')
        plt.legend(['data','Gaussian Fit','beam width'])
        
        #plotting Gaussian fit of horisontal projection
        plt.subplot(1,2,2)
        plt.title('Horizontal Projection')
        plt.plot(self.xlist,self.listHorizontal,'x')
        plt.plot(self.xlist,self.horizontalFit(self.xlist),'r')
        x2,y2 = [self.horizontalFit.mean.value-self.w0_x,self.horizontalFit.mean.value+self.w0_x],[1/math.e**2+self.horizontalFit.offset, 1/math.e**2+self.horizontalFit.offset]
        plt.plot(x2,y2)
        plt.xlim([0,5])
        plt.xlabel('Coordinate on CCD in mm')
        plt.ylabel('Normalized intensity')
        plt.legend(['data','Gaussian Fit','beam width'])
        plt.show()
        
        

class projectionWindow(gaussianWindow):
    def __init__(self,listH,listV,fitH,fitV):
        super().__init__(listH,listV, fitH, fitV)
        self.showProjection()
        
    def showProjection(self):
        w0_y = np.sqrt((2*self.verticalFit.sigma.value)**2)
        w0_x = np.sqrt((2*self.horizontalFit.sigma.value)**2)
        
        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        plt.plot(self.ylist,self.listVertical,'x')
        plt.title('Vertical Projection')
        plt.xlim([0,3.73])
        plt.xlabel('Coordinate on CCD in mm')
        plt.ylabel('Normalized intensity')
        plt.legend(['data'])
        
        #plotting Gaussian fit of horisontal projection
        plt.subplot(1,2,2)
        plt.title('Horizontal Projection')
        plt.plot(self.xlist,self.listHorizontal,'x')
        plt.xlim([0,5])
        plt.xlabel('Coordinate on CCD in mm')
        plt.ylabel('Normalized intensity')
        plt.legend(['data'])
        plt.show()
        
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = ThorCamWindow()
    sys.exit(app.exec_())
    
