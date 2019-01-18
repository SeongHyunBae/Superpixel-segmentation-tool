import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QImage, QPainter, QPalette, QPixmap
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import glob
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import img_as_ubyte

form_class = uic.loadUiType("superpixel_annotation.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.label = Label()
        self.label.release_complete.connect(self.pixel_clicked)
        self.pushButton.clicked.connect(self.save_clicked)
        self.pushButton_2.clicked.connect(self.reset_clicked)
        self.pushButton_3.clicked.connect(self.open_clicked)
        self.pushButton_4.clicked.connect(self.apply_clicked)
        self.pushButton_5.clicked.connect(self.next_clicked)
        self.pushButton_6.clicked.connect(self.previous_clicked)
        # sigma
        self.doubleSpinBox.setRange(0.0, 100.0)
        self.doubleSpinBox.setValue(5.0)
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox.valueChanged.connect(self.double_spin_box_changed)
        # num segments
        self.spinBox.setRange(0, 10000)
        self.spinBox.setValue(100)
        self.spinBox.valueChanged.connect(self.spin_box_changed)
        # R
        self.spinBox_2.setRange(0, 255)
        self.spinBox_2.setValue(255)
        self.spinBox_2.valueChanged.connect(self.spin_box_2_changed)
        # G
        self.spinBox_3.setRange(0, 255)
        self.spinBox_3.setValue(0)
        self.spinBox_3.valueChanged.connect(self.spin_box_3_changed)
        # B
        self.spinBox_4.setRange(0, 255)
        self.spinBox_4.setValue(0)
        self.spinBox_4.valueChanged.connect(self.spin_box_4_changed)

        self.paths = []
        self.save_path = None
        self.index = 0
        self.index_max = 0
        self.numSegments = 100
        self.sigma = 5.0
        self.R = 255
        self.G = 0
        self.B = 0
        self.image = None
        self.segments = None
        self.mask = None
        self.pre_seg = -1
        self.pre_LR = -1
        
    def pixel_clicked(self):
        global press_pt
        global release_pt
        global LR
        
        seg = self.segments[press_pt[1], press_pt[0]]

        if self.pre_seg != seg or self.pre_LR != LR:
            condition = (self.segments == seg)
            
            if LR == 0:
                self.mask[condition] = (self.R, self.G, self.B)
            elif LR == 1:
                self.mask[condition] = (0, 0, 0)

            self.draw_image()
            self.pre_seg = seg

    def apply_clicked(self):
        if self.image is not None:
            self.draw_image()

    def reset_clicked(self):
        if self.image is not None:
            self.mask = np.zeros_like(self.image)
            self.draw_image()

    def save_clicked(self):
        if self.image is not None:
            if self.save_path is None:
                self.save_path = QFileDialog.getExistingDirectory(self, "Select Label Directory")
            save_image = cv2.cvtColor(self.mask, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.save_path + '/' + self.paths[self.index].split('/')[-1], save_image)

    def open_clicked(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        paths = glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png')
        paths.sort()

        if len(paths) > 0:
            self.paths = paths
            self.index = 0
            self.index_max = len(paths)
            self.label_3.setText(self.paths[self.index].split('/')[-1])
            self.update_image()
            self.draw_image()

    def next_clicked(self):
        if self.image is not None:
            self.index += 1
            if self.index == self.index_max:
                self.index = 0
            self.label_3.setText(self.paths[self.index].split('/')[-1])
            self.update_image()
            self.draw_image()

    def previous_clicked(self):
        if self.image is not None:
            self.index -= 1
            self.label_3.setText(self.paths[self.index].split('/')[-1])
            self.update_image()
            self.draw_image()

    def spin_box_changed(self):
        self.numSegments = self.spinBox.value()

    def spin_box_2_changed(self):
        self.R = self.spinBox_2.value()

    def spin_box_3_changed(self):
        self.G = self.spinBox_3.value()

    def spin_box_4_changed(self):
        self.B = self.spinBox_4.value()

    def double_spin_box_changed(self):
        self.sigma = self.doubleSpinBox.value()

    def draw_image(self):
        float_image = img_as_float(self.image)
        self.segments = slic(float_image, n_segments = self.numSegments, sigma = self.sigma)
        boundaries = mark_boundaries(self.image, self.segments)
        cv_image = img_as_ubyte(boundaries)

        result = cv2.addWeighted(cv_image, 0.7, self.mask, 0.3, 0)

        height, width, channels = np.shape(result)
      
        totalBytes = result.nbytes
        bytesPerLine = int(totalBytes / height)
        qimg = QtGui.QImage(result.data, result.shape[1], result.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.label.resize(width, height)
        self.label.setPixmap(pixmap)
        self.label.show()

    def update_image(self):
        self.image = io.imread(self.paths[self.index])
        self.mask = np.zeros_like(self.image)

class Label(QtWidgets.QLabel):
    release_complete = pyqtSignal()
    pressed = 0

    def mouseMoveEvent(self, event):
        global press_pt
        if self.pressed == 1:
            press_pt = event.x(), event.y()
            self.release_complete.emit()
   
    def mousePressEvent(self, event):
        global press_pt
        global LR

        self.pressed = 1
        press_pt = event.x(), event.y()

        if event.button() == Qt.LeftButton:
            LR = 0

        elif event.button() == Qt.RightButton:
            LR = 1

        self.release_complete.emit()
    
    def mouseReleaseEvent(self, event):
        global release_pt
        global LR

        self.pressed = 0
        release_pt = event.x(), event.y()

        if event.button() == Qt.LeftButton:
            LR = 0

        elif event.button() == Qt.RightButton:
            LR = 1

if __name__ == "__main__":
    press_pt = None
    release_pt = None
    LR = -1

    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
