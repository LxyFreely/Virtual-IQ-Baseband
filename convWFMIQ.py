import os
import numpy as np
import wave
import struct
import math
from scipy import signal
import soundfile as sf
import argparse
from scipy.signal import lfilter
from tqdm import tqdm

def generate_mpx_signal(left_channel, right_channel, sample_rate=192000, skip_normalization=False, pre_emphasis_alpha=0.901,no_pilot=False, superHF=0):
    """
    生成MPX信号，预加重放在DC去除之前（正确位置）
    参数:
    left_channel: 左声道信号
    right_channel: 右声道信号
    sample_rate: 采样率 (默认192000 Hz)
    skip_normalization: 是否跳过归一化 (用于调试)
    pre_emphasis_alpha: 预加重系数 (0-1, 默认0.901)
    """
    # ✅ 关键修复: 预加重放在DC去除之前 (正确位置)
    if pre_emphasis_alpha is not None and 0 < pre_emphasis_alpha < 1:
        # 创建FIR滤波器系数 [1, -alpha]
        b = [1, -pre_emphasis_alpha]
        a = [1]
        # 对左右声道分别进行预加重
        left_channel = lfilter(b, a, left_channel)*1.4
        right_channel = lfilter(b, a, right_channel)*1.4
        print(f"✅ 已对左右声道应用预加重 (alpha={pre_emphasis_alpha:.2f})")
    
    l_plus_r = left_channel + right_channel
    l_minus_r = left_channel - right_channel
    
    # DC去除 (现在在预加重之后)
    nyquist = 0.5 * sample_rate
    cutoff_dc = 0.1 / nyquist
    b_dc, a_dc = signal.butter(1, cutoff_dc, btype='high')
    l_plus_r = signal.filtfilt(b_dc, a_dc, l_plus_r)
    l_minus_r = signal.filtfilt(b_dc, a_dc, l_minus_r)
    
    # 低通滤波 (截止15kHz)
    if not superHF==2:
        if superHF ==1:
            cutoff=18000 / nyquist
        elif superHF ==0:
            cutoff = 15000 / nyquist
        b, a = signal.butter(5, cutoff, btype='low')
        l_plus_r_filtered = signal.filtfilt(b, a, l_plus_r)
        l_minus_r_filtered = signal.filtfilt(b, a, l_minus_r)
    
    t = np.arange(len(l_plus_r)) / sample_rate
    
    # ✅ 固定导频振幅 (0.1) - 与音频无关
    pilot = 0.1 * np.sin(2 * np.pi * 19000 * t)
    
    carrier_freq = 38000  # 标准38kHz载波
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    
    l_minus_r_modulated = l_minus_r_filtered * carrier
    
    if no_pilot:
        mpx_signal = l_plus_r_filtered + l_minus_r_modulated
    else:
        mpx_signal = l_plus_r_filtered + l_minus_r_modulated + pilot
    
    # ✅ 安全归一化 (0.85)
    if not skip_normalization:
        safe_factor = 0.85
        max_abs = np.max(np.abs(mpx_signal))
        if max_abs > safe_factor:
            mpx_signal = mpx_signal * (safe_factor / max_abs)
    
    return mpx_signal

def convert_to_sdr_baseband(input_file, output_file, target_sample_rate=240000, bit_depth=16, no_fm=False, skip_normalization=False, pre_emphasis_alpha=0.901, no_pilot=False, superHF=0):
    """
    将立体声音频转换为SDR WFM测试用基带信号
    参数:
    input_file: 输入音频文件路径 (立体声)
    output_file: 输出WAV文件路径
    target_sample_rate: 目标采样率 (默认240000 Hz)
    bit_depth: 位深度 (8/16, 默认16)
    no_fm: 是否不进行FM调制 (默认False)
    skip_normalization: 是否跳过MPX归一化 (默认False)
    pre_emphasis_alpha: 预加重系数 (0-1, 默认0.901)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    data, sample_rate = sf.read(input_file)
    
    # 自动重采样到192kHz (FM标准)
    if sample_rate != 192000:
        print(f"⚠️ 输入音频采样率 {sample_rate} Hz 不是标准的192kHz，正在重采样到192kHz...")
        data = signal.resample_poly(data, 192000, sample_rate)
        sample_rate = 192000
        print(f"✅ 已重采样到标准192kHz采样率")
    
    if len(data.shape) == 1 or data.shape[1] != 2:
        raise ValueError("输入音频必须是立体声 (2声道)")
    
    left_channel = data[:, 0].astype(np.float64)
    right_channel = data[:, 1].astype(np.float64)
    
    print("生成MPX信号...")
    mpx_signal = generate_mpx_signal(
        left_channel, 
        right_channel, 
        192000, 
        skip_normalization=skip_normalization,
        pre_emphasis_alpha=pre_emphasis_alpha,
        no_pilot=no_pilot,
        superHF=superHF
    )
    
    print(f"重采样到 {target_sample_rate} Hz...")
    mpx_signal_resampled = signal.resample_poly(mpx_signal, target_sample_rate, 192000)
    
    if no_fm:
        print("✅ 未进行FM调制，直接输出MPX信号")
        baseband_i = mpx_signal_resampled
        baseband_q = np.zeros_like(mpx_signal_resampled)
    else:
        print("FM调制 (基带表示)...")
        t = np.arange(len(mpx_signal_resampled)) / (target_sample_rate*10)
        
        # ✅ 关键修复1: 正确实现FM调制公式
        fc=100000     #载波频率，由于精度原因，在0上算不出精确数值
        k_f = 75000  # 标准FM频偏 (75kHz)





        #方法1：正常方法
        #phase = 2 * np.pi * k_f * np.cumsum(mpx_signal_resampled) / target_sample_rate
        print("计算相位...")
        phase = 2 * np.pi * fc * t + 2 * np.pi * k_f * np.cumsum(mpx_signal_resampled) / (target_sample_rate*10)


        #方法2：递归法
        '''
        phase_increment = 2 * np.pi * k_f * mpx_signal_resampled / target_sample_rate
         初始化相位
        phase = np.zeros_like(mpx_signal_resampled)
        phase[0] = 0  # 初始相位

         递归计算相位，确保相位在[-π, π]范围内
        for i in range(1, len(phase)):
            phase[i] = phase[i-1] + phase_increment[i]
            # 确保相位连续，避免数值误差
            if phase[i] > np.pi:
                phase[i] -= 2 * np.pi
            elif phase[i] < -np.pi:
                phase[i] += 2 * np.pi
        
        '''





        #现在再下变频回0Hz
        #print("下变频回0Hz...")
        #if_i = np.cos(phase)
        #if_q = np.sin(phase)

        #local_i=np.cos(2 * np.pi * fc * t)
        #local_q=np.sin(2 * np.pi * fc * t)

        #print("生成基带I/Q信号...")
        #baseband_i = if_i * local_i + if_q * local_q
        #baseband_q = if_q * local_i - if_i * local_q

        baseband_i=np.cos(phase)
        baseband_q=np.sin(phase)

        #print("低通滤波...")
        #nyquist = 0.5 * target_sample_rate
        #cutoff = 15e3 / nyquist  # FM基带带宽 (15kHz)
        #b, a = signal.butter(5, cutoff, btype='low')
        
        #baseband_i_filtered = signal.filtfilt(b, a, baseband_i)
        #baseband_q_filtered = signal.filtfilt(b, a, baseband_q)
        
        #baseband_i = baseband_i_filtered
        #baseband_q = baseband_q_filtered
    
    # 确保信号在-1到1范围内
    #baseband_i = np.clip(baseband_i, -1, 1)
    #baseband_q = np.clip(baseband_q, -1, 1)
    #重采样回目标采样率
    #print(f"重采样回目标采样率 {target_sample_rate} Hz...")
    #baseband_i = signal.resample_poly(baseband_i, target_sample_rate, target_sample_rate*10)
    #baseband_q = signal.resample_poly(baseband_q, target_sample_rate, target_sample_rate*10)
    
    if bit_depth == 8:
        # 8位使用无符号整数 (0-255)
        baseband_i_8bit = np.clip(baseband_i * 127 + 128, 0, 255).astype(np.uint8)
        baseband_q_8bit = np.clip(baseband_q * 127 + 128, 0, 255).astype(np.uint8)
    elif bit_depth == 16:
        # 16位使用有符号整数 (-32768 to 32767)
        baseband_i_16bit = np.clip(baseband_i * 32767, -32768, 32767).astype(np.int16)
        baseband_q_16bit = np.clip(baseband_q * 32767, -32768, 32767).astype(np.int16)
    else:
        raise ValueError("bit_depth必须是8或16")
    
    print("创建IQ信号...")
    if bit_depth == 8:
        iq_signal = np.column_stack((baseband_i_8bit, baseband_q_8bit))
    else:  # bit_depth == 16
        # 16位输出
        iq_signal = np.column_stack((baseband_i_16bit, baseband_q_16bit))
    
    print("保存为WAV文件...")
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(target_sample_rate)
        wav_file.setnframes(len(iq_signal))
        wav_file.setcomptype('NONE', 'not compressed')
        wav_file.writeframes(iq_signal.tobytes())
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"参数: {target_sample_rate}Hz, {bit_depth}bit, 2声道 (I/Q)")
    print(f"FM调制: {'启用' if not no_fm else '禁用'}")
    print(f"归一化: {'安全缩放到0.85' if not skip_normalization else '跳过归一化'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将立体声音频转换为SDR WFM测试用基带信号')
    parser.add_argument('input', help='输入音频文件路径 (立体声)')
    parser.add_argument('output', help='输出WAV文件路径')
    parser.add_argument('--sample-rate', type=int, default=240000, help='目标采样率 (标准2400000Hz)')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16], default=16, help='位深度 (8/16, 默认16)')
    parser.add_argument('--no-fm', action='store_true', help='不进行FM调制，直接输出MPX信号')
    parser.add_argument('--skip-normalization', action='store_true', help='跳过MPX归一化 (仅用于调试)')
    parser.add_argument('--pre-emphasis-alpha', type=float, default=0.901, help='预加重系数 (0-1, 默认0.8) - 位置已修复且优化')
    parser.add_argument('--no-pilot', action='store_true', help='是否不添加导频信号 (默认False)')
    parser.add_argument('--superHF', type=int , default=0, choices=[0, 1, 2], help='低通滤波档位:0:15k, 1:18k, 2:不滤波')
    
    args = parser.parse_args()
    
    convert_to_sdr_baseband(
        args.input, 
        args.output, 
        args.sample_rate, 
        args.bit_depth,
        args.no_fm,
        args.skip_normalization,
        args.pre_emphasis_alpha,
        args.no_pilot,
        args.superHF
    )