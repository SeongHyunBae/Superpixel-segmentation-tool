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
import copy
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
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
        self.pushButton_7.clicked.connect(self.red_clicked)
        self.pushButton_8.clicked.connect(self.blue_clicked)
        self.pushButton_9.clicked.connect(self.green_clicked)
        self.pushButton_10.clicked.connect(self.white_clicked)
        self.pushButton_11.clicked.connect(self.yello_clicked)
        self.pushButton_12.clicked.connect(self.black_clicked)
        self.pushButton_13.clicked.connect(self.redo_clicked)
        self.pushButton_14.clicked.connect(self.undo_clicked)

        self.radioButton.clicked.connect(self.radio_button_clicked)
        self.radioButton_2.clicked.connect(self.radio_button_clicked)
        self.radioButton_3.clicked.connect(self.radio_button_clicked)
        self.radioButton_4.clicked.connect(self.radio_button_clicked)
        self.radioButton_5.clicked.connect(self.radio_button_clicked)
        self.radioButton_6.clicked.connect(self.radio_button_clicked)
        self.radioButton_6.setChecked(True)
        
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(1000)
        self.horizontalSlider.setValue(100)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.sliderReleased.connect(self.slider_1_changed)
        self.horizontalSlider.valueChanged.connect(self.slider_1_value_changed)
        self.label_1.setText('scale : ' + str(100))

        self.horizontalSlider_4.setMinimum(0)
        self.horizontalSlider_4.setMaximum(100)
        self.horizontalSlider_4.setValue(5)
        self.horizontalSlider_4.setTickInterval(1)
        self.horizontalSlider_4.sliderReleased.connect(self.slider_1_changed)
        self.horizontalSlider_4.valueChanged.connect(self.slider_1_value_changed)
        self.label_11.setText('sigma : ' + str(5./10.))

        self.horizontalSlider_5.setMinimum(0)
        self.horizontalSlider_5.setMaximum(1000)
        self.horizontalSlider_5.setValue(50)
        self.horizontalSlider_5.setTickInterval(1)
        self.horizontalSlider_5.sliderReleased.connect(self.slider_1_changed)
        self.horizontalSlider_5.valueChanged.connect(self.slider_1_value_changed)
        self.label_13.setText('min_size : ' + str(50))

        self.horizontalSlider_2.setMinimum(1)
        self.horizontalSlider_2.setMaximum(5000)
        self.horizontalSlider_2.setValue(250)
        self.horizontalSlider_2.setTickInterval(1)
        self.horizontalSlider_2.sliderReleased.connect(self.slider_2_changed)
        self.horizontalSlider_2.valueChanged.connect(self.slider_2_value_changed)
        self.label_2.setText('n_segments : ' + str(250))

        self.horizontalSlider_6.setMinimum(1)
        self.horizontalSlider_6.setMaximum(30)
        self.horizontalSlider_6.setValue(10)
        self.horizontalSlider_6.setTickInterval(1)
        self.horizontalSlider_6.sliderReleased.connect(self.slider_2_changed)
        self.horizontalSlider_6.valueChanged.connect(self.slider_2_value_changed)
        self.label_14.setText('compactness : ' + str(10))

        self.horizontalSlider_7.setMinimum(0)
        self.horizontalSlider_7.setMaximum(100)
        self.horizontalSlider_7.setValue(1)
        self.horizontalSlider_7.setTickInterval(1)
        self.horizontalSlider_7.sliderReleased.connect(self.slider_2_changed)
        self.horizontalSlider_7.valueChanged.connect(self.slider_2_value_changed)
        self.label_15.setText('sigma : ' + str(1/10.))

        self.horizontalSlider_3.setMinimum(0)
        self.horizontalSlider_3.setMaximum(10)
        self.horizontalSlider_3.setValue(3)
        self.horizontalSlider_3.setTickInterval(1)
        self.horizontalSlider_3.sliderReleased.connect(self.slider_3_changed)
        self.horizontalSlider_3.valueChanged.connect(self.slider_3_value_changed)
        self.label_10.setText('kernel_size : ' + str(3))

        self.horizontalSlider_8.setMinimum(0)
        self.horizontalSlider_8.setMaximum(100)
        self.horizontalSlider_8.setValue(6)
        self.horizontalSlider_8.setTickInterval(1)
        self.horizontalSlider_8.sliderReleased.connect(self.slider_3_changed)
        self.horizontalSlider_8.valueChanged.connect(self.slider_3_value_changed)
        self.label_16.setText('max_dist : ' + str(6))

        self.horizontalSlider_9.setMinimum(1)
        self.horizontalSlider_9.setMaximum(100)
        self.horizontalSlider_9.setValue(50)
        self.horizontalSlider_9.setTickInterval(1)
        self.horizontalSlider_9.sliderReleased.connect(self.slider_3_changed)
        self.horizontalSlider_9.valueChanged.connect(self.slider_3_value_changed)
        self.label_17.setText('ratio : ' + str(50/100.))

        self.horizontalSlider_10.setMinimum(0)
        self.horizontalSlider_10.setMaximum(5000)
        self.horizontalSlider_10.setValue(250)
        self.horizontalSlider_10.setTickInterval(1)
        self.horizontalSlider_10.sliderReleased.connect(self.slider_4_changed)
        self.horizontalSlider_10.valueChanged.connect(self.slider_4_value_changed)
        self.label_18.setText('markers : ' + str(250))

        self.horizontalSlider_11.setMinimum(10)
        self.horizontalSlider_11.setMaximum(100)
        self.horizontalSlider_11.setValue(1000)
        self.horizontalSlider_11.setTickInterval(1)
        self.horizontalSlider_11.sliderReleased.connect(self.slider_4_changed)
        self.horizontalSlider_11.valueChanged.connect(self.slider_4_value_changed)
        self.label_19.setText('compactness : ' + str(1000/100000.))

        self.horizontalSlider_12.setMinimum(1)
        self.horizontalSlider_12.setMaximum(100)
        self.horizontalSlider_12.setValue(5)
        self.horizontalSlider_12.setTickInterval(1)
        self.horizontalSlider_12.sliderReleased.connect(self.slider_5_changed)
        self.horizontalSlider_12.valueChanged.connect(self.slider_5_value_changed)
        self.label_23.setText('radius : ' + str(5))

        self.horizontalSlider_13.setMinimum(1)
        self.horizontalSlider_13.setMaximum(100)
        self.horizontalSlider_13.setValue(5)
        self.horizontalSlider_13.setTickInterval(1)
        self.horizontalSlider_13.sliderReleased.connect(self.slider_6_changed)
        self.horizontalSlider_13.valueChanged.connect(self.slider_6_value_changed)
        self.label_22.setText('thickness : ' + str(5))

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

        self.checkBox.stateChanged.connect(self.check_box_changed)
        self.checkBox_2.stateChanged.connect(self.check_box_2_changed)
        self.hide = 0
        self.mask_on = 0

        self.paths = []
        self.save_path = None
        self.index = 0
        self.index_max = 0
        
        self.method = 5
        self.felzenszwalb_scale = 100
        self.felzenszwalb_sigma = 0.5
        self.felzenszwalb_min_size = 50

        self.slic_n_segments = 250
        self.slic_compactness = 10
        self.slic_sigma = 1

        self.quickshift_kernel_size = 3
        self.quickshift_max_dist = 6
        self.quickshift_ratio = 0.5

        self.watershed_markers = 1000
        self.watershed_compactness = 0.001

        self.radius = 5
        self.thickness = 5

        self.R = 255
        self.G = 0
        self.B = 0
        self.image = None
        self.segments = None
        self.mask = None
        self.pre_mask = None
        self.redo_mask = None
        self.undo = 0
        self.line = False
        self.line_start = None
        self.temp_mask = None

        self.just_mask = 0
        self.left_press = -1
        self.right_press = -1
        self.candidate = []
        self.delete_candidate = []

    def slider_1_value_changed(self):
        self.felzenszwalb_scale = self.horizontalSlider.value()
        self.felzenszwalb_sigma = float(self.horizontalSlider_4.value())/10.
        self.felzenszwalb_min_size = self.horizontalSlider_5.value()
        self.label_1.setText('scale : ' + str(self.felzenszwalb_scale))
        self.label_11.setText('sigma : ' + str(self.felzenszwalb_sigma))
        self.label_13.setText('min_size : ' + str(self.felzenszwalb_min_size))

    def slider_2_value_changed(self):
        self.slic_n_segments = self.horizontalSlider_2.value()
        self.slic_compactness = self.horizontalSlider_6.value()
        self.slic_sigma = self.horizontalSlider_7.value()/10.
        self.label_2.setText('n_segments : ' + str(self.slic_n_segments))
        self.label_14.setText('compactness : ' + str(self.slic_compactness))
        self.label_15.setText('sigma : ' + str(self.slic_sigma))

    def slider_3_value_changed(self):
        self.quickshift_kernel_size = self.horizontalSlider_3.value()
        self.quickshift_max_dist = self.horizontalSlider_8.value()
        self.quickshift_ratio = self.horizontalSlider_9.value()/100.
        self.label_10.setText('kernel_size : ' + str(self.quickshift_kernel_size))
        self.label_16.setText('max_dist : ' + str(self.quickshift_max_dist))
        self.label_17.setText('ratio : ' + str(self.quickshift_ratio))

    def slider_4_value_changed(self):
        self.watershed_markers = self.horizontalSlider_10.value()
        self.watershed_compactness = self.horizontalSlider_11.value()/100000.
        self.label_18.setText('markers : ' + str(self.watershed_markers))
        self.label_19.setText('compactness : ' + str(self.watershed_compactness))

    def slider_5_value_changed(self):
        self.radius = self.horizontalSlider_12.value()
        self.label_23.setText('radius : ' + str(self.radius))

    def slider_6_value_changed(self):
        self.thickness = self.horizontalSlider_13.value()
        self.label_22.setText('thickness : ' + str(self.thickness))

    def slider_1_changed(self):
        self.felzenszwalb_scale = self.horizontalSlider.value()
        self.felzenszwalb_sigma = float(self.horizontalSlider_4.value())/10.
        self.felzenszwalb_min_size = self.horizontalSlider_5.value()
        self.label_1.setText('scale : ' + str(self.felzenszwalb_scale))
        self.label_11.setText('sigma : ' + str(self.felzenszwalb_sigma))
        self.label_13.setText('min_size : ' + str(self.felzenszwalb_min_size))
        if self.image is not None:
            self.draw_image()

    def slider_2_changed(self):
        self.slic_n_segments = self.horizontalSlider_2.value()
        self.slic_compactness = self.horizontalSlider_6.value()
        self.slic_sigma = self.horizontalSlider_7.value()/10.
        self.label_2.setText('n_segments : ' + str(self.slic_n_segments))
        self.label_14.setText('compactness : ' + str(self.slic_compactness))
        self.label_15.setText('sigma : ' + str(self.slic_sigma))
        if self.image is not None:
            self.draw_image()

    def slider_3_changed(self):
        self.quickshift_kernel_size = self.horizontalSlider_3.value()
        self.quickshift_max_dist = self.horizontalSlider_8.value()
        self.quickshift_ratio = self.horizontalSlider_9.value()/100.
        self.label_10.setText('kernel_size : ' + str(self.quickshift_kernel_size))
        self.label_16.setText('max_dist : ' + str(self.quickshift_max_dist))
        self.label_17.setText('ratio : ' + str(self.quickshift_ratio))
        if self.image is not None:
            self.draw_image()

    def slider_4_changed(self):
        self.watershed_markers = self.horizontalSlider_10.value()
        self.watershed_compactness = self.horizontalSlider_11.value()/100000.
        self.label_18.setText('markers : ' + str(self.watershed_markers))
        self.label_19.setText('compactness : ' + str(self.watershed_compactness))
        if self.image is not None:
            self.draw_image()

    def slider_5_changed(self):
        self.radius = self.horizontalSlider_12.value()
        self.label_23.setText('radius : ' + str(self.radius))

    def slider_6_changed(self):
        self.thickness = self.horizontalSlider_13.value()
        self.label_22.setText('thickness : ' + str(self.thickness))

    def radio_button_clicked(self):
       
        if self.radioButton.isChecked():
            self.method = 0
            self.line = False
            if self.image is not None:
                self.draw_image()
        elif self.radioButton_2.isChecked():
            self.method = 1
            self.line = False
            if self.image is not None:
                self.draw_image()
        elif self.radioButton_3.isChecked():
            self.method = 2
            self.line = False
            if self.image is not None:
                self.draw_image()
        elif self.radioButton_4.isChecked():
            self.method = 3
            self.line = False
            if self.image is not None:
                self.draw_image()
        elif self.radioButton_5.isChecked():
            self.method = 4
            self.line = False
            if self.image is not None:
                self.draw_image()
        else:
            self.method = 5
            if self.image is not None:
                self.draw_image()

    def check_box_changed(self):
        if self.image is not None:
            if self.checkBox.isChecked() == True:
                self.hide = 1
                self.draw_image()
            else:
                self.hide = 0
                self.draw_image()

    def check_box_2_changed(self):
        if self.image is not None:
            if self.checkBox_2.isChecked() == True:
                self.mask_on = 1
                self.draw_image()
            else:
                self.mask_on = 0
                self.draw_image()

    def left_press_callback(self):
        global press_pt

        h, w, _ = self.image.shape

        if press_pt[1] > 0 and press_pt[1] < h and press_pt[0] > 0 and press_pt[0] < w:
            if self.segments is not None:
                seg = self.segments[press_pt[1], press_pt[0]]

                if seg not in self.candidate:
                    self.candidate.append(seg)

            else:
                pt = press_pt

                if self.line == False:
                    cv2.circle(self.mask, (pt[0], pt[1]), self.radius, (self.R, self.G, self.B), -1)
                    self.draw_image()
            
                else:
                    self.line_start = pt
                    self.temp_mask = copy.copy(self.mask)
                
            self.left_press = 1

    def right_press_callback(self):
        global press_pt

        h, w, _ = self.image.shape

        if press_pt[1] > 0 and press_pt[1] < h and press_pt[0] > 0 and press_pt[0] < w:
            if self.segments is not None:
                seg = self.segments[press_pt[1], press_pt[0]]

                if seg not in self.delete_candidate:
                    self.delete_candidate.append(seg)
                    self.right_press = 1
            else:
                pt = press_pt
                cv2.circle(self.mask, (pt[0], pt[1]), self.radius, (0, 0, 0), -1)
                self.draw_image()
                self.right_press = 1

    def left_release_callback(self):
        global release_pt

        h, w, _ = self.image.shape

        if self.segments is not None:
            if release_pt[1] > 0 and release_pt[1] < h and release_pt[0] > 0 and release_pt[0] < w:
                
                seg = self.segments[release_pt[1], release_pt[0]]

                if seg not in self.candidate:
                    self.candidate.append(seg)

            self.pre_mask = copy.copy(self.mask)
            for i in self.candidate:
                condition = (self.segments == i)
                self.mask[condition] = (self.R, self.G, self.B)

            self.just_mask = 1
            self.draw_image()
            self.just_mask = 0
            self.left_press = 0

        else:
            if release_pt[1] > 0 and release_pt[1] < h and release_pt[0] > 0 and release_pt[0] < w:
                
                pt = release_pt
                self.pre_mask = copy.copy(self.mask)
                if self.line == False:
                    cv2.circle(self.mask, (pt[0], pt[1]), self.radius, (self.R, self.G, self.B), -1)

                else:
                    cv2.line(self.mask, (self.line_start[0], self.line_start[1]), (pt[0], pt[1]), (self.R, self.G, self.B), self.thickness)
                
            self.draw_image()
            self.left_press = 0

    def right_release_callback(self):
        global release_pt

        h, w, _ = self.image.shape

        if self.segments is not None:
            if release_pt[1] > 0 and release_pt[1] < h and release_pt[0] > 0 and release_pt[0] < w:
                seg = self.segments[release_pt[1], release_pt[0]]

                if seg not in self.delete_candidate:
                    self.delete_candidate.append(seg)
            self.pre_mask = copy.copy(self.mask)
            for i in self.delete_candidate:
                condition = (self.segments == i)
                self.mask[condition] = (0, 0, 0)

            self.just_mask = 1
            self.draw_image()
            self.just_mask = 0
            self.right_press = 0

        else:
            if release_pt[1] > 0 and release_pt[1] < h and release_pt[0] > 0 and release_pt[0] < w:
                
                pt = release_pt
                self.pre_mask = copy.copy(self.mask)
                cv2.circle(self.mask, (pt[0], pt[1]), self.radius, (0, 0, 0), -1)
                self.draw_image()
                self.right_press = 0

    def move_callback(self):
        global move_pt

        if self.left_press == 1:
            h, w, _ = self.image.shape

            if move_pt[1] > 0 and move_pt[1] < h and move_pt[0] > 0 and move_pt[0] < w:
                if self.segments is not None:

                    seg = self.segments[move_pt[1], move_pt[0]]

                    if seg not in self.candidate:
                        self.candidate.append(seg)

                else:
                    pt = move_pt
                    if self.line == False:
                        cv2.circle(self.mask, (pt[0], pt[1]), self.radius, (self.R, self.G, self.B), -1)
                        self.draw_image()
                    else:
                        cv2.line(self.mask, (self.line_start[0], self.line_start[1]), (pt[0], pt[1]), (self.R, self.G, self.B), self.thickness)
                        self.draw_image()
                        self.mask = copy.copy(self.temp_mask)

        if self.right_press == 1:
            h, w, _ = self.image.shape

            if move_pt[1] > 0 and move_pt[1] < h and move_pt[0] > 0 and move_pt[0] < w:
                if self.segments is not None:
                    seg = self.segments[move_pt[1], move_pt[0]]

                    if seg not in self.delete_candidate:
                        self.delete_candidate.append(seg)

                else:
                    pt = move_pt
                    cv2.circle(self.mask, (pt[0], pt[1]), self.radius, (0, 0, 0), -1)
                    self.draw_image()

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

    def red_clicked(self):
        self.R = 255
        self.G = 0
        self.B = 0
        self.spinBox_2.setValue(255)
        self.spinBox_3.setValue(0)
        self.spinBox_4.setValue(0)

    def blue_clicked(self):
        self.R = 0
        self.G = 0
        self.B = 255
        self.spinBox_2.setValue(0)
        self.spinBox_3.setValue(0)
        self.spinBox_4.setValue(255)

    def green_clicked(self):
        self.R = 0
        self.G = 255
        self.B = 0
        self.spinBox_2.setValue(0)
        self.spinBox_3.setValue(255)
        self.spinBox_4.setValue(0)

    def white_clicked(self):
        self.R = 255
        self.G = 255
        self.B = 255
        self.spinBox_2.setValue(255)
        self.spinBox_3.setValue(255)
        self.spinBox_4.setValue(255)

    def yello_clicked(self):
        self.R = 255
        self.G = 255
        self.B = 0
        self.spinBox_2.setValue(255)
        self.spinBox_3.setValue(255)
        self.spinBox_4.setValue(0)

    def black_clicked(self):
        self.R = 0
        self.G = 0
        self.B = 0
        self.spinBox_2.setValue(0)
        self.spinBox_3.setValue(0)
        self.spinBox_4.setValue(0)

    def redo_clicked(self):
        if self.undo == 1:
            self.mask = copy.copy(self.redo_mask)
            self.draw_image()
            self.undo = 0
            
    def undo_clicked(self):
        self.undo = 1
        self.redo_mask = copy.copy(self.mask)
        self.mask = copy.copy(self.pre_mask)
        self.draw_image()

    def spin_box_2_changed(self):
        self.R = self.spinBox_2.value()

    def spin_box_3_changed(self):
        self.G = self.spinBox_3.value()

    def spin_box_4_changed(self):
        self.B = self.spinBox_4.value()

    def draw_image(self):
        float_image = img_as_float(self.image)

        if self.method == 0:
            if self.just_mask == 0:
                self.segments = felzenszwalb(float_image, scale=self.felzenszwalb_scale, sigma=self.felzenszwalb_sigma, min_size=self.felzenszwalb_min_size)
        
        elif self.method == 1:
            if self.just_mask == 0:
                self.segments = slic(float_image, n_segments=self.slic_n_segments, compactness=self.slic_compactness, sigma=self.slic_sigma)
        
        elif self.method == 2:
            if self.just_mask == 0:
                self.segments = quickshift(float_image, kernel_size=self.quickshift_kernel_size, max_dist=self.quickshift_max_dist, ratio=self.quickshift_ratio)
        
        elif self.method == 3:
            if self.just_mask == 0:
                gradient = sobel(rgb2gray(float_image))
                self.segments = watershed(gradient, markers=self.watershed_markers, compactness=self.watershed_compactness)
        
        elif self.method == 4:
            self.segments = None
            self.line = False
        
        else:
            self.segments = None
            self.line = True

        if self.segments is not None:
            boundaries = mark_boundaries(self.image, self.segments)
            cv_image = img_as_ubyte(boundaries)

            if self.hide == 0:
                if self.mask_on == 0:
                    result = cv2.addWeighted(cv_image, 0.5, self.mask, 0.5, 0)
                else:
                    result = self.mask
            else:
                if self.mask_on == 0:
                    result = cv2.addWeighted(self.image, 0.5, self.mask, 0.5, 0)
                else:
                    result = self.mask
        else:
            if self.mask_on == 0:
                result = cv2.addWeighted(self.image, 0.5, self.mask, 0.5, 0)
            else:
                result = self.mask
        
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
        if self.save_path is not None:
            label = cv2.imread(self.save_path + '/' + self.paths[self.index].split('/')[-1])
            if label is not None:
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                self.mask = label
                self.pre_mask = copy.copy(self.mask) 
            else:
                self.mask = np.zeros_like(self.image)
                self.pre_mask = copy.copy(self.mask)        
        else:
            self.mask = np.zeros_like(self.image)
            self.pre_mask = copy.copy(self.mask) 

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
