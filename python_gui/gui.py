import matplotlib.pyplot as plt
from ctypes import*
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pyqtgraph as pg

modulation_dll = cdll.LoadLibrary("C:\\magisterka\\git\\digital_modulations\\main2.dll")
amplitude = c_float(1)
freq = c_float(1)
cos_factor_idx = c_int(1)
modulation_dll.init_func(amplitude, freq, cos_factor_idx)

class MainWindow(QMainWindow):
    factor = 0
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.factor = 0
        self.setWindowTitle("Digital modulations")
        self.graphWidget = pg.PlotWidget()

        self.setCentralWidget(self.graphWidget)

        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        button_action = QAction("Button", self)
        button_action.setStatusTip("This is button")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)
        toolbar.addAction(button_action)

        self.setStatusBar(QStatusBar(self))

    def onMyToolBarButtonClick(self, s):
        print("click", s)
        self.factor += 1
        n_samples = 180 * 8 * self.factor
        type_for_probki = c_float * n_samples
        wsk_probki = type_for_probki.in_dll(modulation_dll,"modulated_data")
        hamming = np.array(wsk_probki[:])
        print(hamming)
        self.graphWidget.clear()
        self.graphWidget.plot(hamming)
        amplitude2 = c_float(6)
        freq2 = c_float(15)
        cos_factor_idx2 = c_int(2)
        n_samples = 2 * 8
        modulation_dll.init_func(amplitude2, freq2, cos_factor_idx2)


app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec_()
