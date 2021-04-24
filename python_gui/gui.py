import matplotlib.pyplot as plt
from ctypes import*
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
import sys
import pyqtgraph as pg
from timeit import default_timer as timer


class modulated_data_window(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("Modulated_data")
        self.setGeometry(600, 600, 600, 600)
        self.label = QLabel("Modulated data")
        self.setLayout(layout)
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setLabel("left", "Amplitude [V]")
        self.graphWidget.setLabel("bottom", "N")

        layout.addWidget(self.graphWidget)

    def show_mod_data(self, data):

        self.graphWidget.clear()
        self.graphWidget.plot(data)

class modulation_c():
    def __init__(self):
        super().__init__()
        self.amplitude = c_float(1)
        self.frequency = c_float(1)
        self.cos_factor_idx = c_int(2)
        self.n_bits_c = c_int(0)
        self.n_bits_py = 0
        self.mod_type = ""
        self.bit_stream = 0
        self.n_samples_factor = 0
        self.modulation_dll = cdll.LoadLibrary("D:\\magisterka\\git\\digital_modulations\\main2.dll")
        self.mod_data_window = modulated_data_window()
        self.n_samples = 0
        self.demodulated_data = []

    def set_mod_parameters(self, amp, freq, cos_fac_idx, n_bits, mod_type, bit_stream):
        if mod_type == "qpsk" or mod_type == "qpskc":
            self.n_samples_factor = 2
        elif mod_type == "16qam" or mod_type == "16qamc":
            self.n_samples_factor = 4
        else:
            self.n_samples_factor = 1
        print(mod_type)

        self.amplitude = c_float(amp)
        self.frequency = c_float(freq)
        self.cos_factor_idx = c_int(cos_fac_idx)
        self.n_bits_c = c_int(n_bits)
        self.n_bits_py = n_bits
        self.bit_stream = bit_stream
        self.mod_type = create_string_buffer(mod_type.encode('utf-8'))

    def modulate(self, amp, freq, cos_fac_idx, n_bits, mod_type, bit_stream):

        self.set_mod_parameters(amp, freq, cos_fac_idx, n_bits, mod_type, bit_stream)
        self.modulation_dll.cuda_dummy_free()
        start = timer()

        status = self.modulation_dll.init_func(self.amplitude, self.frequency, self.cos_factor_idx, self.n_bits_c, byref(self.bit_stream),
                                          self.mod_type)
        end = timer()
        print("time = ", end - start)
        if status == 0:
            n_samples = int(180 * cos_fac_idx / self.n_samples_factor) * n_bits
            self.n_samples = n_samples
            type_for_probki = c_float * n_samples

            modulated_data_ptr = POINTER(type_for_probki).in_dll(self.modulation_dll, "modulated_data").contents
            modulated_data = np.array(modulated_data_ptr[:])

            self.demodulated_data = []
            if mod_type == "bpsk" or mod_type == "bpskc":
                for i in range(n_bits):
                    self.demodulated_data.append(int(modulated_data[360 * i + 180] == amp))
            elif mod_type == "qpsk" or mod_type == "qpskc":
                for i in range(int(n_bits / 2)):
                    if modulated_data[360 * i + 45] == -1 * amp:
                        self.demodulated_data.append(1)
                        self.demodulated_data.append(1)
                    elif modulated_data[360 * i + 45] == 1 * amp:
                        self.demodulated_data.append(0)
                        self.demodulated_data.append(0)
                    elif modulated_data[360 * i + 270] < 0 and  modulated_data[360 * i + 359] < 0:
                        self.demodulated_data.append(1)
                        self.demodulated_data.append(0)
                    else:
                        self.demodulated_data.append(0)
                        self.demodulated_data.append(1)
            elif mod_type == "am":
                for i in range(n_bits):
                    self.demodulated_data.append(int(modulated_data[360 * i + 180] == -amp))

            elif mod_type == "fm" or mod_type == "fmc":
                for i in range(n_bits):
                    self.demodulated_data.append(int(modulated_data[360 * i + 45] !=
                                                     - modulated_data[360 * i + 90]))
            else:
                self.demodulated_data.append(0)

            # print("modulated data")
            # for i in range(n_bits):
            #     print(self.demodulated_data[i])

            self.mod_data_window.show_mod_data(modulated_data)
            self.mod_data_window.show()

            self.modulation_dll.memory_free()
        else:
            print("Modulation exit with error code = ", status)



class MainWindow(QMainWindow):
    factor = 0
    modulation_type = ""

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.factor = 0
        self.setWindowTitle("Digital modulations")
        self.setGeometry(600, 600, 600, 600)

        self.start_button = QtWidgets.QPushButton(self)
        self.start_button.setText("Start modulation")
        self.start_button.move(300, 350)
        self.start_button.clicked.connect(self.start_clicked)

        self.file_button = QtWidgets.QPushButton(self)
        self.file_button.setText("Use input file")
        self.file_button.move(300, 300)
        self.file_button.clicked.connect(self.show_file_dialog)

        self.setStatusBar(QStatusBar(self))

        self.freq_spinbox = QtWidgets.QDoubleSpinBox(self)
        self.freq_spinbox.setMinimum(1)
        self.freq_spinbox.move(200, 100)

        self.freq_label = QtWidgets.QLabel(self)
        self.freq_label.setText("Frequency")
        self.freq_label.move(140, 100)

        self.amp_spinbox = QtWidgets.QDoubleSpinBox(self)
        self.amp_spinbox.setMinimum(1)
        self.amp_spinbox.move(200, 150)

        self.amp_label = QtWidgets.QLabel(self)
        self.amp_label.setText("Amplitude")
        self.amp_label.move(140, 150)

        self.file = QtWidgets.QFileDialog(self)
        self.file.setFileMode(QFileDialog.AnyFile)
        self.file.move(400, 400)

        self.text_edit = QTextEdit(self)
        self.text_edit.move(400, 200)

        self.text_input_label = QtWidgets.QLabel(self)
        self.text_input_label.setText("Input bits")
        self.text_input_label.move(400, 170)

        checkbox_location = 20
        checkbox_width = 20
        self.bpsk_checkbox = QCheckBox("bpsk", self)
        self.bpsk_checkbox.move(20, checkbox_location)
        self.bpsk_cuda_checkbox = QCheckBox("bpskc", self)
        self.bpsk_cuda_checkbox.move(20, checkbox_location + checkbox_width * 1)
        self.qpsk_checkbox = QCheckBox("qpsk", self)
        self.qpsk_checkbox.move(20, checkbox_location + checkbox_width * 2)
        self.qpsk_cuda_checkbox = QCheckBox("qpskc", self)
        self.qpsk_cuda_checkbox.move(20, checkbox_location + checkbox_width * 3)
        self.am_checkbox = QCheckBox("am", self)
        self.am_checkbox.move(20, checkbox_location + checkbox_width * 4)
        self.am_cuda_checkbox = QCheckBox("amc", self)
        self.am_cuda_checkbox.move(20, checkbox_location + checkbox_width * 5)
        self.fm_checkbox = QCheckBox("fm", self)
        self.fm_checkbox.move(20, checkbox_location + checkbox_width * 6)
        self.fm_cuda_checkbox = QCheckBox("fmc", self)
        self.fm_cuda_checkbox.move(20, checkbox_location + checkbox_width * 7)
        self._16qam_checkbox = QCheckBox("16qam", self)
        self._16qam_checkbox.move(20, checkbox_location + checkbox_width * 8)
        self._16qam_cuda_checkbox = QCheckBox("16qamc", self)
        self._16qam_cuda_checkbox.move(20, checkbox_location + checkbox_width * 9)

        self.mod_checkbox = QButtonGroup()
        self.mod_checkbox.addButton(self.bpsk_checkbox, 1)
        self.mod_checkbox.addButton(self.qpsk_checkbox, 2)
        self.mod_checkbox.addButton(self.am_checkbox, 3)
        self.mod_checkbox.addButton(self.fm_checkbox, 4)
        self.mod_checkbox.addButton(self.bpsk_cuda_checkbox, 5)
        self.mod_checkbox.addButton(self.qpsk_cuda_checkbox, 6)
        self.mod_checkbox.addButton(self.fm_cuda_checkbox, 7)
        self.mod_checkbox.addButton(self.am_cuda_checkbox, 8)
        self.mod_checkbox.addButton(self._16qam_checkbox, 7)
        self.mod_checkbox.addButton(self._16qam_cuda_checkbox, 8)
        self.modulation = modulation_c()

    def show_file_dialog(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "C:\\magisterka\\git\\digital_modulations\\")

        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()
                self.text_edit.setText(data)
                print(self.text_edit.toPlainText())

    def start_clicked(self):
        self.get_modulation_type()

        amplitude = self.amp_spinbox.value()
        freq = self.freq_spinbox.value()
        cos_factor_idx = 2
        bits = []
        input_text = self.text_edit.toPlainText()
        for i in input_text :
            if i == "0" or i == "1":
                bits.append(int(i))
        n_bits = len(bits)
        seq = c_int * n_bits
        bit_stream = seq(*bits)

        self.modulation.modulate(amplitude, freq, cos_factor_idx, n_bits, self.modulation_type, bit_stream)

    def get_modulation_type(self):

        for button in self.mod_checkbox.buttons():
            if button.isChecked():
                self.modulation_type = button.text()


app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec_()

