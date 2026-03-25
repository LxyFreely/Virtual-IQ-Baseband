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
from numba import njit

@njit(fastmath=False,cache=True)
def kahan_cumsum_numba(arr):
    n=len(arr)
    result=np.zeros(n, dtype=np.float64)
    sum_val=0.0
    c=0.0

    for i in range(n):
        y=arr[i]-c
        t=sum_val+y
        c=(t-sum_val)-y
        sum_val=t
        result[i]=sum_val

    return result

@njit(fastmath=False,cache=True)
def fm_modulator_complex_high_prec(mpx,fc,k_f,target_sample_rate):
    n_samples=len(mpx)
    baseband_i=np.zeros(n_samples, dtype=np.float64)
    baseband_q=np.zeros(n_samples, dtype=np.float64)
    vec_real=1.0
    vec_imag=0.0

    for i in range(n_samples):
        total_angle=2*np.pi*(k_f*mpx[i]+fc)/target_sample_rate
        rot_cos=np.cos(total_angle)
        rot_sin=np.sin(total_angle)
        new_real=vec_real*rot_cos-vec_imag*rot_sin
        new_imag=vec_real*rot_sin+vec_imag*rot_cos
        if i%100==0:
            mag=np.sqrt(new_real**2+new_imag**2)
            vec_real=new_real/mag
            vec_imag=new_imag/mag
        else:
            vec_real=new_real
            vec_imag=new_imag
        baseband_i[i]=vec_real
        baseband_q[i]=vec_imag
    return baseband_i,baseband_q

def lowpass_fir(audio,sample_rate,cutoff_freq=15000,numtaps=201,window='hamming'):
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    if numtaps%2==0:
        numtaps+=1
    if window=='kaiser':
        coeffs=signal.firwin(numtaps,normalized_cutoff,window=('kaiser',8.0))
    else:
        coeffs=signal.firwin(numtaps,normalized_cutoff,window=window)
    padlen=3*(numtaps-1)
    if len(audio)<=padlen:
        raise ValueError(f"音频长度不足，需要{padlen}个样本")
    filtered=signal.filtfilt(coeffs,[1],audio,padlen=padlen)
    return filtered

def lowcut_fir(audio,sample_rate,cutoff_freq=20,numtaps=1601,window='hamming'):
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    if numtaps%2==0:
        numtaps+=1
    if window=='kaiser':
        coeffs=signal.firwin(numtaps,normalized_cutoff,window=('kaiser',8.0),pass_zero=False)
    else:
        coeffs=signal.firwin(numtaps,normalized_cutoff,window=window,pass_zero=False)
    padlen=3*(numtaps-1)
    if len(audio)<=padlen:
        raise ValueError(f"音频长度不足，需要{padlen}个样本")
    filtered=signal.filtfilt(coeffs,[1],audio,padlen=padlen)
    return filtered
def generate_mpx_signal(left_channel, right_channel, sample_rate=192000, skip_normalization=False, pre_emphasis_alpha=0.901,no_pilot=False, superHF=0, lpbyresamp=False, tanh=False,strictDC=False):
    """
    生成MPX信号
    参数:
    left_channel: 左声道信号
    right_channel: 右声道信号
    sample_rate: 采样率 (默认192000 Hz)
    skip_normalization: 是否跳过归一化 (用于调试)
    pre_emphasis_alpha: 预加重系数 (0-1, 默认0.901)
    """

    # DC去除
    print(f"DC去除")
    nyquist = 0.5 * sample_rate
    cutoff_dc = 20 #低切频率20Hz
    numtaps=801
    b_dc, a_dc = signal.butter(1, cutoff_dc / nyquist, btype='high')
    left_channel = signal.filtfilt(b_dc, a_dc, left_channel)
    right_channel = signal.filtfilt(b_dc, a_dc, right_channel)
    print("已用butter初步去除")
    if strictDC:
        left_channel = lowcut_fir(left_channel,sample_rate,cutoff_freq=cutoff_dc,numtaps=numtaps)
        right_channel = lowcut_fir(right_channel,sample_rate,cutoff_freq=cutoff_dc,numtaps=numtaps)
        print(f"已用fir滤波器滤波 (截止{cutoff_dc:.0f}Hz, {numtaps}个系数)")
    


    # 预加重
    if pre_emphasis_alpha is not None and 0 < pre_emphasis_alpha < 1:
        print(f"预加重 (alpha={pre_emphasis_alpha:.2f})")
        # 创建FIR滤波器系数 [1, -alpha]
        b = [1, -pre_emphasis_alpha]
        a = [1]
        # 对左右声道分别进行预加重
        left_channel = lfilter(b, a, left_channel)
        right_channel = lfilter(b, a, right_channel)
        
    if tanh:
        print(f"对mpx使用tanh模拟过载")
        max_abs=np.max((np.abs(left_channel),np.abs(right_channel)))
        left_channel/=max_abs
        right_channel/=max_abs
        print(f"tanh前归一化已完成（归一化因子={max_abs:.2f}）")
        left_channel = np.tanh(left_channel)
        right_channel = np.tanh(right_channel)
        print('已完成tanh模拟过载')
    
    print(f"计算midside信号")
    l_plus_r = left_channel + right_channel
    l_minus_r = left_channel - right_channel
    
    
    # 低通滤波 (截止15kHz)
    if not superHF==1:
        cutoff=18000
        numtaps=151
    elif not superHF==0:
        cutoff = 15000
        numtaps=301
    if  lpbyresamp:
        print("采用重采样进行低通滤波")
        #比如15k截止频率就重采样到30k再重采样回192k
        target_sample_rate = cutoff*2
        l_plus_r_filtered = signal.resample_poly(l_plus_r, target_sample_rate, sample_rate)
        l_minus_r_filtered = signal.resample_poly(l_minus_r, target_sample_rate, sample_rate)
        l_plus_r_filtered = signal.resample_poly(l_plus_r_filtered, sample_rate, target_sample_rate)
        l_minus_r_filtered = signal.resample_poly(l_minus_r_filtered, sample_rate, target_sample_rate)
        print(f"已通过重采样低通滤波(截止{cutoff:.0f}Hz,通过{target_sample_rate:.0f}Hz采样)")
    print("正常采用fir进行滤波")
    l_plus_r_filtered = lowpass_fir(l_plus_r,sample_rate,cutoff_freq=cutoff,numtaps=numtaps)
    l_minus_r_filtered = lowpass_fir(l_minus_r,sample_rate,cutoff_freq=cutoff,numtaps=numtaps)
    print(f"已低通滤波 (截止{cutoff:.0f}Hz, {numtaps}个系数)")



    
    
    
    t = np.arange(len(l_plus_r)) / sample_rate
    
    # ✅ 固定导频振幅 (0.1) - 与音频无关
    print(f"计算导频")
    pilot = 0.1 * np.sin(2 * np.pi * 19000 * t)#提高音频信号比，标准为0.1，给音频信号留电平(测试过了因为信噪比在这即使不按照规范来也能用)
    
    carrier_freq = 38000  # 标准38kHz载波
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    
    l_minus_r_modulated = l_minus_r_filtered * carrier
    
    mpx_signal = l_plus_r_filtered + l_minus_r_modulated
    #第一次归一化拉满电平（归一化因子选择从大到小排序第若干项忽略尖峰）
    if not skip_normalization:
        print(f"第一次归一化拉满电平")
        max_abs = np.sort(np.abs(mpx_signal))[::-1][20000]
        mpx_signal = mpx_signal / max_abs
        print(f"第一次归一化完成 (归一化因子={max_abs:.2f})")
    
    
    #乘1.2然后softclip到[-1,1]
    #mpx_signal = np.tanh(mpx_signal * 1.5)

    if not no_pilot:
        mpx_signal = mpx_signal + pilot
    
    # 第二次归一化
    if not skip_normalization:
        print(f"第二次归一化")
        safe_factor = 1
        max_abs = np.sort(np.abs(mpx_signal))[::-1][5000]*safe_factor
        mpx_signal = mpx_signal * (safe_factor / max_abs)
        print(f"第二次归一化完成 (归一化因子={safe_factor:.2f})")

    return mpx_signal

def convert_to_sdr_baseband(input_file, output_file, 
                            target_sample_rate=240000, 
                            bit_depth=16, 
                            no_fm=False, 
                            skip_normalization=False, 
                            pre_emphasis_alpha=0.901, 
                            no_pilot=False, 
                            superHF=0, 
                            fc=1000000, 
                            k_f=75000, 
                            lpbyresamp=False, 
                            tanh=False, 
                            iqtanh=False,
                            FM_function=0,
                            strictDC=False
                            ):
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
        print(f"已重采样到标准192kHz采样率")
    
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
        superHF=superHF,
        lpbyresamp=lpbyresamp,
        tanh=tanh,
        strictDC=strictDC
    )
    
    print(f"重采样到 {target_sample_rate} Hz...")
    mpx_signal_resampled = signal.resample_poly(mpx_signal, target_sample_rate, 192000)
    
    if no_fm:
        print("✅ 未进行FM调制，直接输出MPX信号")
        baseband_i = mpx_signal_resampled
        baseband_q = np.zeros_like(mpx_signal_resampled)
    else:
        print("FM调制...")
        t = np.arange(len(mpx_signal_resampled)) / (target_sample_rate)
        
        # ✅ 关键修复1: 正确实现FM调制公式
        fc=fc     #载波频率，由于精度原因，在0上算不出精确数值
        k_f = k_f  # 标准FM频偏 (75kHz)
        if FM_function==0:
            print("使用公式直接运算")
            phase_increment=2 * np.pi * k_f * mpx_signal_resampled / target_sample_rate
            print("计算相位差")
            phase=kahan_cumsum_numba(phase_increment)
            print("积分")
            phase=phase+2*np.pi*fc*t
            print("添加载波相位")
            phase=np.mod(phase,2*np.pi)
            print("相位取模")
            baseband_i=np.cos(phase)
            baseband_q=np.sin(phase)
            print("生成基带I/Q信号")
        elif FM_function==1:
            print("使用复数单位圆模拟法")
            #模拟一个复数平面内旋转的单位向量，基于向量角度来直接运算向量，直接取xy作为iq信号
            baseband_i,baseband_q=fm_modulator_complex_high_prec(mpx_signal_resampled,fc,k_f,target_sample_rate)



        
        #方法1：正常方法
        #phase = 2 * np.pi * k_f * np.cumsum(mpx_signal_resampled) / target_sample_rate
        #print("计算相位...")
        #phase = 2 * np.pi * fc * t + 2 * np.pi * k_f * np.cumsum(mpx_signal_resampled) / (target_sample_rate)

        #现在再下变频回0Hz
        #print("下变频回0Hz...")
        #if_i = np.cos(phase)
        #if_q = np.sin(phase)

        #local_i=np.cos(2 * np.pi * fc * t)
        #local_q=np.sin(2 * np.pi * fc * t)

        #print("生成基带I/Q信号...")
        #baseband_i = if_i * local_i + if_q * local_q
        #baseband_q = if_q * local_i - if_i * local_q

        if iqtanh:
            print(f"对iq信号使用tanh模拟过载")
            baseband_i = np.tanh(baseband_i)
            baseband_q = np.tanh(baseband_q)

        #print("低通滤波...")
        #nyquist = 0.5 * target_sample_rate
        #cutoff = 15e3 / nyquist  # FM基带带宽 (15kHz)
        #b, a = signal.butter(5, cutoff, btype='low')
        
        #baseband_i_filtered = signal.filtfilt(b, a, baseband_i)
        #baseband_q_filtered = signal.filtfilt(b, a, baseband_q)
        
        #baseband_i = baseband_i_filtered
        #baseband_q = baseband_q_filtered
    
    #重采样回来
    #baseband_i = signal.resample_poly(baseband_i, target_sample_rate, target_sample_rate*10)
    #baseband_q = signal.resample_poly(baseband_q, target_sample_rate, target_sample_rate*10)
    # 确保信号在-1到1范围内
    #baseband_i = np.clip(baseband_i, -1, 1)
    #baseband_q = np.clip(baseband_q, -1, 1)
    
    if bit_depth == 8:
        # 8位使用无符号整数 (0-255)
        baseband_i_store = np.clip(baseband_i * 127 + 128, 0, 255).astype(np.uint8)
        baseband_q_store = np.clip(baseband_q * 127 + 128, 0, 255).astype(np.uint8)
    elif bit_depth == 16:
        # 16位使用有符号整数 (-32768 to 32767)
        baseband_i_store = np.clip(baseband_i * 32767, -32768, 32767).astype(np.int16)
        baseband_q_store = np.clip(baseband_q * 32767, -32768, 32767).astype(np.int16)
    elif bit_depth == 32:
        #32位IEEE
        baseband_i_store = baseband_i.astype(np.float32)
        baseband_q_store = baseband_q.astype(np.float32)
    else:
        raise ValueError("bit_depth必须是8,16,32位或32位")
    
    print("创建IQ信号...")
    iq_signal = np.column_stack((baseband_i_store, baseband_q_store))
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
    print(f"归一化: {'缩放到1' if not skip_normalization else '跳过归一化'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将立体声音频转换为SDR WFM测试用基带信号')
    parser.add_argument('input', help='输入音频文件路径 (立体声)')
    parser.add_argument('output', help='输出WAV文件路径')
    parser.add_argument('--sample-rate', type=int, default=240000, help='目标采样率 (标准2400000Hz)')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16, 32], default=16, help='位深度 (8/16/32IEEE, 默认16)')
    parser.add_argument('--no-fm', action='store_true', help='不进行FM调制，直接输出MPX信号')
    parser.add_argument('--skip-normalization', action='store_true', help='跳过MPX归一化 (仅用于调试)')
    parser.add_argument('--pre-emphasis-alpha', type=float, default=0.901, help='预加重系数 (0-1, 默认0.8) - 位置已修复且优化')
    parser.add_argument('--no-pilot', action='store_true', help='是否不添加导频信号 (默认False)')
    parser.add_argument('--superHF', type=int , default=0, choices=[0, 1, 2], help='低通滤波档位:0:15k, 1:18k, 2:不滤波')
    parser.add_argument('--fc', type=float, default=1000, help='载波频率 (默认1000Hz)')
    parser.add_argument('--k-f', type=float, default=75000, help='FM频率偏移 (默认75kHz)')
    parser.add_argument('--lpbyresamp',action='store_true', help='是否通过重采样来实现低通制造混叠味 (默认False)')
    parser.add_argument('--tanh',action='store_true', help='是否对mpx使用tanh模拟过载 (默认False)')
    parser.add_argument('--iqtanh',action='store_true', help='是否对iq信号使用tanh (默认False)')
    parser.add_argument('--fm-function', type=int, default=1,choices=[0,1], help='选择FM所用的方法:0:公式直接运算,1:复数单位圆模拟法')
    parser.add_argument('--strictDC', action='store_true', help='是否启用fir滤波过滤DC (默认False)')
    
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
        args.superHF,
        args.fc,
        args.k_f,
        args.lpbyresamp,
        args.tanh,
        args.iqtanh,
        args.fm_function,
        args.strictDC,
    )