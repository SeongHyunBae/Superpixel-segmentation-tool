import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QImage, QPainter, QPalette, QPixmap
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
        self.label.left_press_signal.connect(self.left_press_callback)
        self.label.right_press_signal.connect(self.right_press_callback)
        self.label.left_release_signal.connect(self.left_release_callback)
        self.label.right_release_signal.connect(self.right_release_callback)
        self.label.move_signal.connect(self.move_callback)

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

        self.left_press = -1
        self.right_press = -1
        self.candidate = []
        self.delete_candidate = []

    def left_press_callback(self):
        global press_pt

        h, w, _ = self.image.shape

        if press_pt[1] > 0 and press_pt[1] < h and press_pt[0] > 0 and press_pt[0] < w:
            seg = self.segments[press_pt[1], press_pt[0]]

            if seg not in self.candidate:
                self.candidate.append(seg)
                self.left_press = 1

    def right_press_callback(self):
        global press_pt

        h, w, _ = self.image.shape

        if press_pt[1] > 0 and press_pt[1] < h and press_pt[0] > 0 and press_pt[0] < w:
            seg = self.segments[press_pt[1], press_pt[0]]

            if seg not in self.delete_candidate:
                self.delete_candidate.append(seg)
                self.right_press = 1

    def left_release_callback(self):
        global release_pt

        h, w, _ = self.image.shape

        if release_pt[1] > 0 and release_pt[1] < h and release_pt[0] > 0 and release_pt[0] < w:
            seg = self.segments[release_pt[1], release_pt[0]]

            if seg not in self.candidate:
                self.candidate.append(seg)

        for i in self.candidate:
            condition = (self.segments == i)
            self.mask[condition] = (self.R, self.G, self.B)

        self.draw_image()
        self.left_press = 0

    def right_release_callback(self):
        global release_pt

        h, w, _ = self.image.shape

        if release_pt[1] > 0 and release_pt[1] < h and release_pt[0] > 0 and release_pt[0] < w:
            seg = self.segments[release_pt[1], release_pt[0]]

            if seg not in self.delete_candidate:
                self.delete_candidate.append(seg)

        for i in self.delete_candidate:
            condition = (self.segments == i)
            self.mask[condition] = (0, 0, 0)

        self.draw_image()
        self.right_press = 0

    def move_callback(self):
        global move_pt

        if self.left_press == 1:
            h, w, _ = self.image.shape

            if move_pt[1] > 0 and move_pt[1] < h and move_pt[0] > 0 and move_pt[0] < w:
                seg = self.segments[move_pt[1], move_pt[0]]

                if seg not in self.candidate:
                    self.candidate.append(seg)

        if self.right_press == 1:
            h, w, _ = self.image.shape

            if move_pt[1] > 0 and move_pt[1] < h and move_pt[0] > 0 and move_pt[0] < w:
                seg = self.segments[move_pt[1], move_pt[0]]

                if seg not in self.delete_candidate:
                    self.delete_candidate.append(seg)

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

        self.candidate = []
        self.delete_candidate = []

    def update_image(self):
        self.image = io.imread(self.paths[self.index])
        self.mask = np.zeros_like(self.image)

class Label(QtWidgets.QLabel):
    move_signal = pyqtSignal()
    left_press_signal = pyqtSignal()
    right_press_signal = pyqtSignal()
    left_release_signal = pyqtSignal()
    right_release_signal = pyqtSignal()

    def mouseMoveEvent(self, event):
        global move_pt

        move_pt = event.x(), event.y()
        self.move_signal.emit()
   
    def mousePressEvent(self, event):
        global press_pt

        press_pt = event.x(), event.y()

        if event.button() == Qt.LeftButton:
            self.left_press_signal.emit()

        elif event.button() == Qt.RightButton:
            self.right_press_signal.emit()

    def mouseReleaseEvent(self, event):
        global release_pt

        release_pt = event.x(), event.y()

        if event.button() == Qt.LeftButton:
            self.left_release_signal.emit()

        elif event.button() == Qt.RightButton:
            self.right_release_signal.emit()

if __name__ == "__main__":
    move_pt = None
    press_pt = None
    release_pt = None

    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
