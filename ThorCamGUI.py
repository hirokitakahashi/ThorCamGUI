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


class ThorCamWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCameras()
        if self.camera:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.updateImage)
            self.timer.start(10) # update image at every 10ms
    
    def initUI(self):
        self.setWindowTitle('Thorlab Camera Interface')
        self.resize(1000,1000)
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
        #old_roi = self.camera.roi  # store the current roi
        self.camera.arm(2)
        self.camera.issue_software_trigger()
    
    def updateImage(self):
        # get a frame from the camera and update the image
        frame = self.camera.get_pending_frame_or_null()
        if frame is not None:
            self.image_buffer = np.copy(frame.image_buffer)
            self.img.setImage(self.image_buffer)
    
    def closeEvent(self, event):
        # override the closing behaviour
        self.sdk.__del__()
        self.timer.stop()
        event.accept()
        
if __name__ == '__main__':        
    app = QtGui.QApplication(sys.argv)
    win = ThorCamWindow()
    sys.exit(app.exec_())
    