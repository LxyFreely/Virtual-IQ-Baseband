import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import argparse
from tqdm import tqdm

def main(input_file):
    # 1. 读取IQ基带WAV文件
    print(f"读取文件: {input_file}")
    data, samplerate = sf.read(input_file)
    I = data[:, 0]  # I (In-phase) channel
    Q = data[:, 1]  # Q (Quadrature) channel
    
    # 2. 绘制李萨如图并自动检测半径
    print("绘制李萨如图并自动检测半径...")
    # 计算半径
    radius = np.sqrt(np.mean(I**2 + Q**2))
    
    
    print(f"检测到的半径: {radius:.2f}")
    
    # 3. 沿时间轴的与圆的差值
    print("计算沿时间轴的与圆的差值...")
    distances = np.sqrt(I**2 + Q**2)
    overlap_ratio = distances - radius
    
    # 4. 整个音频的中频频谱
    print("计算整个音频的中频频谱...")
    f, Pxx = signal.welch(I + 1j*Q, fs=samplerate, nperseg=1024)
    
    # 5. 解调FM信号并绘制MPX频谱
    print("解调FM信号并绘制MPX频谱...")
    # FM解调(中心频率0Hz)
    print("计算相位...")
    phase = np.unwrap(np.angle(I + 1j*Q))
    dphase = np.diff(phase)
    demodulated = dphase * samplerate / (2*np.pi)

    # 计算MPX频谱
    print("计算MPX频谱...")
    f_mpx, Pxx_mpx = signal.welch(demodulated, fs=samplerate, nperseg=1024)
    
    # 6. 沿时间轴的信噪比（使用190kHz附近的噪声估计底噪）
    print("计算解调后信号的频谱以估计噪声...")

    # 找到190kHz附近的频率点
    target_freq = 190000  # 190kHz
    freq_idx = np.argmin(np.abs(f_mpx - target_freq))

    # 使用200Hz的窗口计算噪声功率（在190kHz附近）
    noise_window = 200  # Hz
    noise_start = max(0, freq_idx - noise_window//2)
    noise_end = min(len(f_mpx), freq_idx + noise_window//2)

    # 计算该窗口内的平均噪声功率
    noise_power = np.mean(Pxx[noise_start:noise_end])
    noise_rms = np.sqrt(noise_power)

    # 沿时间轴的SNR（使用滑动窗口）
    print("计算SNR...")
    window_size = 1024
    snr_values = []
    pbar = tqdm(total=len(demodulated), desc="处理进度", unit="项", initial=0, dynamic_ncols=True)
    for i in range(0, len(demodulated), window_size):
        window = demodulated[i:i+window_size]
        pbar.update(len(window))
        if len(window) == window_size:
            signal_power = np.mean(window**2)
            # 使用190kHz处的噪声功率计算SNR
            snr = 10 * np.log10(signal_power / noise_power)
            snr_values.append(snr)
    pbar.close()

    # 创建一张包含所有图表的大图
    print("创建包含所有图表的大图...")
    pbar = tqdm(total=5, desc="创建大图进度", unit="张", initial=0, dynamic_ncols=True)
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Lissajous Figure
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(I, Q, 'b.', alpha=0.1)
    ax1.set_title('Lissajous Figure')
    ax1.set_xlabel('I')
    ax1.set_ylabel('Q')
    ax1.axis('equal')
    pbar.update(1)
    
    # 绘制检测到的圆
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    ax1.plot(x_circle, y_circle, 'r-', lw=2)
    ax1.text(0.8*radius, 0.8*radius, f'Radius: {radius:.2f}', fontsize=12, color='red')
    pbar.update(1)
    
    # 2. Overlap Ratio with Circle
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(overlap_ratio)
    ax2.set_title('Overlap Ratio with Circle')
    ax2.set_xlabel('Time (samples)')
    ax2.set_ylabel('Overlap Ratio')
    ax2.axhline(y=1, color='r', linestyle='--')
    pbar.update(1)
    
    # 3. Midband Spectrum
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(f, 10*np.log10(Pxx))
    ax3.set_title('Midband Spectrum')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power/Frequency (dB/Hz)')
    ax3.grid(True)
    pbar.update(1)
    
    # 4. MPX Spectrum
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(f_mpx, 10*np.log10(Pxx_mpx))
    ax4.set_title('MPX Spectrum')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power/Frequency (dB/Hz)')
    ax4.set_xlim(0, 192000)
    ax4.grid(True)
    pbar.update(1)
    
    # 5. SNR Over Time
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(snr_values)
    ax5.set_title('SNR Over Time')
    ax5.set_xlabel('Time (samples)')
    ax5.set_ylabel('SNR (dB)')
    ax5.grid(True)
    pbar.update(1)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存为一张大图
    plt.savefig('output.png')
    plt.close()
    pbar.close()
    
    print("Analysis complete. Output saved as 'output.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze FM radio IQ baseband signal')
    parser.add_argument('input_file', type=str, help='Input IQ baseband WAV file')
    args = parser.parse_args()
    
    main(args.input_file)