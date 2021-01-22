import matplotlib.pyplot as plt
from ctypes import*
import numpy as np

modulation_dll = cdll.LoadLibrary("C:\\magisterka\\git\\digital_modulations\\main2.dll")

amplitude = c_float(1)
freq = c_float(1)
cos_factor_idx = c_int(1)
n_samples = 180 * 8
modulation_dll.init_func(amplitude, freq, cos_factor_idx)

type_for_probki = c_float * n_samples
wsk_probki = type_for_probki.in_dll(modulation_dll,"modulated_data")
hamming = np.array(wsk_probki[:])
print(hamming)
plt.plot(hamming)
plt.show()
