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
from scipy.optimize import curve_fit

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

def GaussFunc(x, A=1, x0=0, s=1, d=0):
    return A*np.exp(-(x-x0)**2/s**2)+d

def check_concave(x, y):
    # This function checks if the given dataset is concave or not.
    ind = np.argsort(x)
    yy = y[ind]
    xx = x[ind]
    # make a line through (xx[0], yy[0]) and (xx[-1], yy[-1])
    line = (yy[-1]-yy[0])/(xx[-1]-xx[0])*xx+(yy[0]*xx[-1]-yy[-1]*xx[0])/(xx[-1]-xx[0])
    d = yy-line
    if np.mean(d)>0:
        return True
    else:
        return False

def EstimateGaussIni(x, y):
     # estimate initial parameters for fit
     concave = check_concave(x, y)
     if concave:
         miny = np.min(y)
         maxy = np.max(y)
         imax = np.argmax(y)
         x0 = x[imax]
         iyh = np.nonzero(y > 0.6*maxy + 0.4*miny)[0]
         s = (x[iyh[-1]]-x[iyh[0]])/2
         p = np.array([maxy-miny, x0, s, miny])
     else:
         miny = np.min(y)
         maxy = np.max(y)
         imin = np.argmin(y)
         x0 = x[imin]
         iyh = np.nonzero(y < 0.6*miny + 0.4*maxy)[0]
         s = (x[iyh[-1]]-x[iyh[0]])/2
         p = np.array([miny-maxy, x0, s, maxy])
     return p
 
def GaussFit(x, y):
    p0 = EstimateGaussIni(x, y)
    popt, pcov = curve_fit(GaussFunc, x, y, p0)
    yfit = GaussFunc(x, popt[0], popt[1], popt[2], popt[3])
    return popt, yfit
                     
class ThorCamWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.createMenu()
        self.initCameras()
        # initialize fit data to None
        self.fit_h = None
        self.fit_v = None
        if self.camera:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.updateImage)
            self.timer.start(10) # update image at every 10ms
    
    def initUI(self):
        self.setWindowTitle('Thorlab Camera Interface')
        self.resize(1500,1000)
        self.maincontainer = QtGui.QWidget() 
        self.hbox = QtGui.QHBoxLayout()
        self.vbox = QtGui.QVBoxLayout()
        self.hbox2 = QtGui.QHBoxLayout() # horizoantl layout in vbox
        self.img = pg.ImageItem(border='w')
        # plots
        self.plot2D = pg.PlotWidget(self) # camera image
        self.plot_h = pg.plot() # horizontal projection
        self.data_h = self.plot_h.plot() # create PlotDataItem for horizontal projection
        self.plot_v = pg.plot() # vertical projection
        self.data_v = self.plot_v.plot() # create PlotDataItem for vertical projection
        self.plot2D.addItem(self.img)
        # buttons
        self.fit_button = QtGui.QPushButton("Fit", self)
        self.fit_button.clicked.connect(self.fitButtonClicked)
        self.stop_button = QtGui.QPushButton("Stop", self)
        self.stop_button.setStyleSheet('QPushButton {color: red;}')
        self.stop_button.clicked.connect(self.stopButtonClicked)
        self.hbox2.addStretch()
        self.hbox2.addWidget(self.fit_button)
        self.hbox2.addStretch()
        self.hbox2.addWidget(self.stop_button)
        self.hbox2.addStretch()
        # arrange widgets
        self.vbox.addWidget(self.plot_h)
        self.vbox.addWidget(self.plot_v)
        self.vbox.addLayout(self.hbox2)
        self.hbox.addWidget(self.plot2D)
        self.hbox.addLayout(self.vbox)
        self.maincontainer.setLayout(self.hbox)
        self.setCentralWidget(self.maincontainer)
        self.show()
        
    def createMenu(self):
        # create menubar
        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu('File')
        tools_menu = menu_bar.addMenu('Tools')
        help_menu = menu_bar.addMenu('Help')
        
        # Define actions
        exit_act = QtGui.QAction('Exit', self)
        exit_act.setShortcut('Çtrl+Q')
        exit_act.triggered.connect(self.close)
        
        settings_act = QtGui.QAction('Settings', self)
        settings_act.triggered.connect(self.openSettingsDialog)
        about_act = QtGui.QAction('About', self)
        about_act.triggered.connect(self.aboutDialog)
        
        file_menu.addSeparator()
        file_menu.addAction(exit_act)
        tools_menu.addAction(settings_act)
        help_menu.addAction(about_act)
    
    def openSettingsDialog(self):
        #self.ps_dialog = SettingsDialog(self)
        pass
        
    def aboutDialog(self):
        QtGui.QMessageBox.about(self, "ThorCamGui", "This software displays the image of a Thorlab scientific camera.")
    
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
        #old_roi = self.camera.roi  # store the current roi
        self.camera.arm(2)
        self.camera.issue_software_trigger()
    
    def updateImage(self):
        # get a frame from the camera and update the image
        frame = self.camera.get_pending_frame_or_null()
        if frame is not None:
            self.image_buffer = np.copy(frame.image_buffer)
            self.img.setImage(self.image_buffer)
            self.proj_h, self.proj_v = self.getProjs(self.image_buffer)
            self.data_h.setData(self.proj_h)
            self.data_v.setData(self.proj_v)
    
    def getProjs(self, image):
        proj_h = image.sum(axis=1)
        proj_v = image.sum(axis=0)
        return (proj_h, proj_v)
    
    def stopButtonClicked(self):
        if self.timer.isActive():
            self.timer.stop()
            self.stop_button.setText('Start')
        else:
            self.timer.start()
            self.stop_button.setText('Stop')
    
    def fitButtonClicked(self):
        xh = np.arange(0, len(self.proj_h))
        xv = np.arange(0, len(self.proj_v))
        popt_h, yfit_h = GaussFit(xh, self.proj_h)
        popt_v, yfit_v = GaussFit(xv, self.proj_v)
        if self.fit_h == None:
            self.fit_h = pg.PlotDataItem(xh, yfit_h, pen={'color': 'r', 'width': 1})
            self.plot_h.addItem(self.fit_h)
        else:
            self.fit_h.setData(xh, yfit_h)
        
        if self.fit_v == None:
            self.fit_v = pg.PlotDataItem(xv, yfit_v, pen={'color': 'r', 'width': 1})
            self.plot_v.addItem(self.fit_v)
        else:
            self.fit_v.setData(xv, yfit_v, pen={'color': 'r', 'width': 1})
        
    
    def closeEvent(self, event):
        # override the closing behaviour
        self.camera.dispose()
        self.sdk.dispose()
        self.timer.stop()
        event.accept()
        
if __name__ == '__main__':        
    app = QtGui.QApplication(sys.argv)
    win = ThorCamWindow()
    sys.exit(app.exec_())
    