from ctypes import*
import numpy as np
from PyQt5.QtWidgets import *
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

class demodulated_data_window(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowTitle("Constellation")
        self.setGeometry(600, 600, 600, 600)
        self.label = QLabel("Constellation")
        self.setLayout(layout)
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setLabel("left", "Im")
        self.graphWidget.setLabel("bottom", "Re")

        layout.addWidget(self.graphWidget)

    def show_mod_data(self, data_y, data_x):

        self.graphWidget.clear()
        self.graphWidget.plot(data_x, data_y, pen=None, symbol='o')
        self.graphWidget.setRange(rect=None, xRange=[-1,1], yRange=[-1,1])
        self.graphWidget.plot(data_x, data_y, pen=None, symbol='o')
        self.show()

class demodulation_c():
    def __init__(self):
        super().__init__()
        self.mod = ""
        self.amplitude = 1
        self.frequency = 1
        self.n_bits = 0
        self.demod_data = []
        self.constellation_data_x = []
        self.constellation_data_y = []
        self.demodulated_window = demodulated_data_window()

    def set_demodulation_params(self, mod, amp, freq, n_bits, demod_data):
        self.mod = mod
        self.amplitude = amp
        self.frequency = freq
        self.n_bits = n_bits
        self.demod_data = demod_data

    def create_constellation_2_points(self):

        for i in range(self.n_bits):
            white_noise_y = np.random.normal(0, 0.004, 2)
            white_noise_x = np.random.normal(0, 0.004, 2)
            self.constellation_data_y.append(white_noise_y[0])
            if self.demod_data[i] == 1:
                self.constellation_data_x.append(1 + white_noise_x[0])
            else:
                self.constellation_data_x.append(-1 + white_noise_x[0])

    def create_constellation_4_points(self):
        for i in range(int(self.n_bits/2)):
            white_noise = np.random.normal(0, 0.002, 2)
            if self.demod_data[i*2] == 0:
                if self.demod_data[(i*2)+1] == 0:
                    self.constellation_data_y.append(-1 + white_noise[0])
                    self.constellation_data_x.append(-1 + white_noise[1])
                else:
                    self.constellation_data_y.append(1 + white_noise[0])
                    self.constellation_data_x.append(-1 + white_noise[1])
            else:
                if self.demod_data[(i*2) + 1] == 0:
                    self.constellation_data_y.append(-1 + white_noise[0])
                    self.constellation_data_x.append(1 + white_noise[1])
                else:
                    self.constellation_data_y.append(1 + white_noise[0])
                    self.constellation_data_x.append(1 + white_noise[1])

    def create_constellation_16_points(self):
        tmp = []
        for i in range(int(self.n_bits / 2)):
            tmp.append(str(self.demod_data[i * 2]) + str(self.demod_data[(i * 2) + 1]))
        for j in range(int(self.n_bits / 4)):
            white_noise = np.random.normal(0, 0.004, 2)
            i = j*2
            if tmp[i] == "00":
                self.constellation_data_y.append(-0.75 + white_noise[0])
            elif tmp[i] == "01":
                self.constellation_data_y.append(-0.25 + white_noise[0])
            elif tmp[i] == "10":
                self.constellation_data_y.append(0.25 + white_noise[0])
            elif tmp[i] == "11":
                self.constellation_data_y.append(0.75 + white_noise[0])
            if tmp[i + 1] == "00":
                self.constellation_data_x.append(-0.75 + white_noise[1])
            elif tmp[i + 1] == "01":
                self.constellation_data_x.append(-0.25 + white_noise[1])
            elif tmp[i + 1] == "10":
                self.constellation_data_x.append(0.25 + white_noise[1])
            elif tmp[i + 1] == "11":
                self.constellation_data_x.append(0.75 + white_noise[1])


    def get_constellation(self):
        self.constellation_data_x = []
        self.constellation_data_y = []
        if self.mod == "bpsk_c" or self.mod == "bpsk_cuda"\
                or self.mod == "bfsk_c" or self.mod == "bfsk_cuda"\
                or self.mod == "bask_c" or self.mod == "bask_cuda":
            self.create_constellation_2_points()
        elif self.mod == "qpsk_c" or self.mod == "qpsk_cuda":
            self.create_constellation_4_points()
        elif self.mod == "16qam_c" or self.mod == "16qam_cuda":
            self.create_constellation_16_points()

        self.demodulated_window.show_mod_data(self.constellation_data_y, self.constellation_data_x)


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
        self.modulation_dll = cdll.LoadLibrary("D:\\magisterka\\git\\digital_modulations\\cmake-build-debug\\digital_modulations_lib.dll")
        self.mod_data_window = modulated_data_window()
        self.n_samples = 0
        self.demodulation_c = demodulation_c()

    def add_out_text(self, out_text):
        self.out_text = out_text

    def set_mod_parameters(self, amp, freq, cos_fac_idx, n_bits, mod_type, bit_stream):
        if mod_type == "qpsk_c" or mod_type == "qpsk_cuda":
            self.n_samples_factor = 2
        elif mod_type == "16qam_c" or mod_type == "16qam_cuda":
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
        self.modulation_dll.alloc_memory(self.n_bits_c)
        end_time = 0
        for i in range(1000):
            start = timer()

            status = self.modulation_dll.modulate(self.amplitude,
                                              self.frequency,
                                              self.cos_factor_idx,
                                              self.n_bits_c,
                                              byref(self.bit_stream),
                                              self.mod_type)
            end = timer()
            end_time += end - start
        print("\ntime = ", end_time/1000, " mod = ", mod_type)
        if status == 0:
            n_samples = int(180 * cos_fac_idx / self.n_samples_factor) * n_bits
            self.n_samples = n_samples
            type_for_samples = c_float * n_samples
            type_for_bits = c_int32 * n_bits;

            modulated_data_ptr = POINTER(type_for_samples).in_dll(self.modulation_dll, "modulated_data").contents
            modulated_data = np.array(modulated_data_ptr[:])

            demodulated_bits_ptr = POINTER(type_for_bits).in_dll(self.modulation_dll, "demodulated_bits").contents
            demodulated_bits = np.array(demodulated_bits_ptr[:])
            self.out_text.setText(str(demodulated_bits))
            self.demodulation_c.set_demodulation_params(mod_type, amp, freq, n_bits, demodulated_bits)
            self.demodulation_c.get_constellation()

            self.mod_data_window.show_mod_data(modulated_data)
            self.mod_data_window.show()

            f = open("dane_wyjsciowe.txt", "w")
            f.write(str(demodulated_bits))
            f.write()
            f.close()

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
        self.freq_spinbox.setMaximum(100000)
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
        self.text_edit.move(400, 100)

        self.text_input_label = QtWidgets.QLabel(self)
        self.text_input_label.setText("Input bits")
        self.text_input_label.move(400, 70)

        self.text_edit_out = QTextEdit(self)
        self.text_edit_out.move(400, 200)

        self.text_output_label = QtWidgets.QLabel(self)
        self.text_output_label.setText("Demodulated bits")
        self.text_output_label.move(400, 170)

        checkbox_location = 20
        checkbox_width = 20
        self.bpsk_checkbox = QCheckBox("bpsk_c", self)
        self.bpsk_checkbox.move(20, checkbox_location)
        self.bpsk_cuda_checkbox = QCheckBox("bpsk_cuda", self)
        self.bpsk_cuda_checkbox.move(20, checkbox_location + checkbox_width * 1)
        self.qpsk_checkbox = QCheckBox("qpsk_c", self)
        self.qpsk_checkbox.move(20, checkbox_location + checkbox_width * 2)
        self.qpsk_cuda_checkbox = QCheckBox("qpsk_cuda", self)
        self.qpsk_cuda_checkbox.move(20, checkbox_location + checkbox_width * 3)
        self.am_checkbox = QCheckBox("bask_c", self)
        self.am_checkbox.move(20, checkbox_location + checkbox_width * 4)
        self.am_cuda_checkbox = QCheckBox("bask_cuda", self)
        self.am_cuda_checkbox.move(20, checkbox_location + checkbox_width * 5)
        self.fm_checkbox = QCheckBox("bfsk_c", self)
        self.fm_checkbox.move(20, checkbox_location + checkbox_width * 6)
        self.fm_cuda_checkbox = QCheckBox("bfsk_cuda", self)
        self.fm_cuda_checkbox.move(20, checkbox_location + checkbox_width * 7)
        self._16qam_checkbox = QCheckBox("16qam_c", self)
        self._16qam_checkbox.move(20, checkbox_location + checkbox_width * 8)
        self._16qam_cuda_checkbox = QCheckBox("16qam_cuda", self)
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
        self.modulation.add_out_text(self.text_edit_out)

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

