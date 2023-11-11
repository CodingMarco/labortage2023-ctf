import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

rate, wave = scipy.io.wavfile.read("iq_data.wav")
wave_complex = wave[..., 0] + 1j * wave[..., 1]

# Demodulate FSK
data = np.diff(np.unwrap(np.angle(wave_complex)))[100:-100]

# Apply moving average filter
data = scipy.signal.medfilt(data, 5)
data_sig = data

# Threshold data
data = np.where(data > 0, True, False)

# Software PLL
# see https://github.com/carrotIndustries/redbook#clock-recovery
# taken from https://github.com/carrotIndustries/redbook/blob/master/analyze.py#L46
acc = 0
last_acc = 0
ftw = 42
acc_size = 1000
last_bit = False
phase_delta = 0
phase_delta_filtered = 0
integ = 0
alpha = 0.005
integs = []
ftws = []
integ_max = 927681
ftw0 = 25.2
Ki = 0.0000004
Kp = 0.001
sampled_bits = []
clk = [False]
for bit in data:
    if bit != last_bit:
        phase_delta = acc_size / 2 - acc

    if acc < last_acc:  # phase accumulator has wrapped around
        sampled_bits.append(bit)
        clk.append(not clk[-1])

    last_acc = acc
    phase_delta_filtered = phase_delta * alpha + phase_delta_filtered * (1 - alpha)
    integ += phase_delta_filtered
    if integ > integ_max:
        integ = integ_max
    elif integ < -integ_max:
        integ = -integ_max
    integs.append(integ)
    ftw = ftw0 + phase_delta_filtered * Kp + integ * Ki
    ftws.append(ftw)
    lastbit = bit
    acc = (acc + ftw) % acc_size
# End software PLL

sampled_bits_ = sampled_bits[7:]  # Choose (emperically) correct bit to start with
bytes_ = [
    sum([byte[b] << 7 - b for b in range(0, 8)])  # Reverse bit order
    for byte in zip(*(iter(sampled_bits_),) * 8)
]

ctf_result = bytes(bytes_).decode("utf-8", "ignore")[8:]
print(f"CTF result: '{ctf_result}'")

plt.subplot(2, 1, 1)
plt.plot(ftws)
plt.ylabel("Frequency")
plt.title("Current PLL frequency")


# Approximately upsample sampled_bits and clock to match the length of data_sig
sampled_bits = np.array(
    [bit for bit in sampled_bits for _ in range(len(data_sig) // len(sampled_bits) + 1)]
)
sampled_clk = np.array(
    [bit for bit in clk for _ in range(len(data_sig) // len(clk) + 1)]
)


plt.subplot(2, 1, 2)
plt.plot(data_sig, label="Signal")
plt.plot(sampled_bits * 0.2 + 0.1, label="Resulting Bits")
plt.plot(sampled_clk * 0.2 - 0.3, label="Clock")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Data signal")
plt.legend()

plt.show()
