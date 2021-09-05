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

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None


class ThorCamWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.createMenu()
        self.initCameras()
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
        self.img = pg.ImageItem(border='w')
        self.plot = pg.PlotWidget(self)
        self.plot_h = pg.plot()
        self.data_h = self.plot_h.plot()
        self.plot_v = pg.plot()
        self.data_v = self.plot_v.plot()
        self.plot.addItem(self.img)
        self.vbox.addWidget(self.plot_h)
        self.vbox.addWidget(self.plot_v)
        self.hbox.addWidget(self.plot)
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
            proj_h, proj_v = self.getProjs(self.image_buffer)
            self.data_h.setData(proj_h)
            self.data_v.setData(proj_v)
    
    def getProjs(self, image):
        proj_h = image.sum(axis=0)
        proj_v = image.sum(axis=1)
        return (proj_h, proj_v)
    
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
    