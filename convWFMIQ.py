import os
import numpy as np
import struct
import math
from scipy import signal
import soundfile as sf
import argparse
from scipy.signal import lfilter
from tqdm import tqdm
from numba import njit

# ==================== Numba 加速函数 ====================

@njit(fastmath=False, cache=True)
def kahan_cumsum_numba(arr):
    """Kahan 求和算法，提高相位累加精度"""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    sum_val = 0.0
    c = 0.0

    for i in range(n):
        y = arr[i] - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
        result[i] = sum_val

    return result


@njit(fastmath=False, cache=True)
def fm_modulator_complex_high_prec(mpx, fc, k_f, target_sample_rate):
    """复数单位圆模拟法 FM 调制"""
    n_samples = len(mpx)
    baseband_i = np.zeros(n_samples, dtype=np.float64)
    baseband_q = np.zeros(n_samples, dtype=np.float64)
    vec_real = 1.0
    vec_imag = 0.0

    for i in range(n_samples):
        total_angle = 2 * np.pi * (k_f * mpx[i] + fc) / target_sample_rate
        rot_cos = np.cos(total_angle)
        rot_sin = np.sin(total_angle)
        new_real = vec_real * rot_cos - vec_imag * rot_sin
        new_imag = vec_real * rot_sin + vec_imag * rot_cos
        if i % 100 == 0:
            mag = np.sqrt(new_real**2 + new_imag**2)
            vec_real = new_real / mag
            vec_imag = new_imag / mag
        else:
            vec_real = new_real
            vec_imag = new_imag
        baseband_i[i] = vec_real
        baseband_q[i] = vec_imag
    return baseband_i, baseband_q


# ==================== 滤波器函数 ====================

def lowpass_fir(audio, sample_rate, cutoff_freq=15000, numtaps=201, window='hamming'):
    """低通 FIR 滤波器"""
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    if numtaps % 2 == 0:
        numtaps += 1
    if window == 'kaiser':
        coeffs = signal.firwin(numtaps, normalized_cutoff, window=('kaiser', 8.0))
    else:
        coeffs = signal.firwin(numtaps, normalized_cutoff, window=window)
    padlen = 3 * (numtaps - 1)
    if len(audio) <= padlen:
        raise ValueError(f"音频长度不足，需要{padlen}个样本")
    filtered = signal.filtfilt(coeffs, [1], audio, padlen=padlen)
    return filtered


def lowcut_fir(audio, sample_rate, cutoff_freq=20, numtaps=1601, window='hamming'):
    """高通 FIR 滤波器 (低切)"""
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    if numtaps % 2 == 0:
        numtaps += 1
    if window == 'kaiser':
        coeffs = signal.firwin(numtaps, normalized_cutoff, window=('kaiser', 8.0), pass_zero=False)
    else:
        coeffs = signal.firwin(numtaps, normalized_cutoff, window=window, pass_zero=False)
    padlen = 3 * (numtaps - 1)
    if len(audio) <= padlen:
        raise ValueError(f"音频长度不足，需要{padlen}个样本")
    filtered = signal.filtfilt(coeffs, [1], audio, padlen=padlen)
    return filtered


# ==================== MPX 信号生成 ====================

def generate_mpx_signal(left_channel, right_channel, sample_rate=192000, skip_normalization=False,
                        pre_emphasis_alpha=0.901, no_pilot=False, superHF=0, lpbyresamp=False,
                        tanh=False, strictDC=False):
    """生成 FM 立体声 MPX 信号"""

    # DC 去除
    print(f"DC 去除")
    nyquist = 0.5 * sample_rate
    cutoff_dc = 20
    numtaps = 801
    b_dc, a_dc = signal.butter(1, cutoff_dc / nyquist, btype='high')
    left_channel = signal.filtfilt(b_dc, a_dc, left_channel)
    right_channel = signal.filtfilt(b_dc, a_dc, right_channel)
    print("已用 butter 初步去除")
    if strictDC:
        left_channel = lowcut_fir(left_channel, sample_rate, cutoff_freq=cutoff_dc, numtaps=numtaps)
        right_channel = lowcut_fir(right_channel, sample_rate, cutoff_freq=cutoff_dc, numtaps=numtaps)
        print(f"已用 fir 滤波器滤波 (截止{cutoff_dc:.0f}Hz, {numtaps}个系数)")

    # 预加重
    if pre_emphasis_alpha is not None and 0 < pre_emphasis_alpha < 1:
        print(f"预加重 (alpha={pre_emphasis_alpha:.2f})")
        b = [1, -pre_emphasis_alpha]
        a = [1]
        left_channel = lfilter(b, a, left_channel)
        right_channel = lfilter(b, a, right_channel)

    if tanh:
        print(f"对 mpx 使用 tanh 模拟过载")
        max_abs = np.max((np.abs(left_channel), np.abs(right_channel)))
        left_channel /= max_abs
        right_channel /= max_abs
        print(f"tanh 前归一化已完成（归一化因子={max_abs:.2f}）")
        left_channel = np.tanh(left_channel)
        right_channel = np.tanh(right_channel)
        print('已完成 tanh 模拟过载')

    print(f"计算 midside 信号")
    l_plus_r = left_channel + right_channel
    l_minus_r = left_channel - right_channel

    # 低通滤波
    if not superHF == 1:
        cutoff = 18000
        numtaps = 151
    elif not superHF == 0:
        cutoff = 15000
        numtaps = 301

    if lpbyresamp:
        print("采用重采样进行低通滤波")
        target_sample_rate = cutoff * 2
        l_plus_r_filtered = signal.resample_poly(l_plus_r, target_sample_rate, sample_rate)
        l_minus_r_filtered = signal.resample_poly(l_minus_r, target_sample_rate, sample_rate)
        l_plus_r_filtered = signal.resample_poly(l_plus_r_filtered, sample_rate, target_sample_rate)
        l_minus_r_filtered = signal.resample_poly(l_minus_r_filtered, sample_rate, target_sample_rate)
        print(f"已通过重采样低通滤波 (截止{cutoff:.0f}Hz, 通过{target_sample_rate:.0f}Hz 采样)")
    else:
        print("正常采用 fir 进行滤波")
        l_plus_r_filtered = lowpass_fir(l_plus_r, sample_rate, cutoff_freq=cutoff, numtaps=numtaps)
        l_minus_r_filtered = lowpass_fir(l_minus_r, sample_rate, cutoff_freq=cutoff, numtaps=numtaps)
        print(f"已低通滤波 (截止{cutoff:.0f}Hz, {numtaps}个系数)")

    t = np.arange(len(l_plus_r)) / sample_rate

    # 导频信号 (固定振幅 0.1)
    print(f"计算导频")
    pilot = 0.1 * np.sin(2 * np.pi * 19000 * t)

    carrier_freq = 38000
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    l_minus_r_modulated = l_minus_r_filtered * carrier

    mpx_signal = l_plus_r_filtered + l_minus_r_modulated

    # 第一次归一化
    if not skip_normalization:
        print(f"第一次归一化拉满电平")
        max_abs = np.sort(np.abs(mpx_signal))[::-1][20000]
        mpx_signal = mpx_signal / max_abs
        print(f"第一次归一化完成 (归一化因子={max_abs:.2f})")

    if not no_pilot:
        mpx_signal = mpx_signal + pilot

    # 第二次归一化
    if not skip_normalization:
        print(f"第二次归一化")
        safe_factor = 1
        max_abs = np.sort(np.abs(mpx_signal))[::-1][5000] * safe_factor
        mpx_signal = mpx_signal * (safe_factor / max_abs)
        print(f"第二次归一化完成 (归一化因子={safe_factor:.2f})")

    return mpx_signal


# ==================== 主转换函数 (已修改为 soundfile) ====================

def convert_to_sdr_baseband(input_file, output_file, target_sample_rate=240000, bit_depth=16,
                            no_fm=False, skip_normalization=False, pre_emphasis_alpha=0.901,
                            no_pilot=False, superHF=0, fc=1000000, k_f=75000, lpbyresamp=False,
                            tanh=False, iqtanh=False, FM_function=0, strictDC=False):
    """
    将立体声音频转换为 SDR WFM 测试用基带信号
    
    参数:
        input_file: 输入音频文件路径 (立体声)
        output_file: 输出 WAV 文件路径
        target_sample_rate: 目标采样率 (默认 240000 Hz)
        bit_depth: 位深度 (8/16/32, 默认 16)
        no_fm: 是否不进行 FM 调制
        skip_normalization: 是否跳过 MPX 归一化
        pre_emphasis_alpha: 预加重系数 (0-1, 默认 0.901)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在：{input_file}")

    # 使用 soundfile 读取音频
    data, sample_rate = sf.read(input_file)

    # 自动重采样到 192kHz (FM 标准)
    if sample_rate != 192000:
        print(f"⚠️ 输入音频采样率 {sample_rate} Hz 不是标准的 192kHz，正在重采样到 192kHz...")
        data = signal.resample_poly(data, 192000, sample_rate)
        sample_rate = 192000
        print(f"已重采样到标准 192kHz 采样率")

    if len(data.shape) == 1 or data.shape[1] != 2:
        raise ValueError("输入音频必须是立体声 (2 声道)")

    left_channel = data[:, 0].astype(np.float64)
    right_channel = data[:, 1].astype(np.float64)

    print("生成 MPX 信号...")
    mpx_signal = generate_mpx_signal(
        left_channel, right_channel, 192000,
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
        print("✅ 未进行 FM 调制，直接输出 MPX 信号")
        baseband_i = mpx_signal_resampled
        baseband_q = np.zeros_like(mpx_signal_resampled)
    else:
        print("FM 调制...")
        t = np.arange(len(mpx_signal_resampled)) / target_sample_rate

        fc = fc
        k_f = k_f
        if FM_function == 0:
            print("使用公式直接运算")
            phase_increment = 2 * np.pi * k_f * mpx_signal_resampled / target_sample_rate
            print("计算相位差")
            phase = kahan_cumsum_numba(phase_increment)
            print("积分")
            phase = phase + 2 * np.pi * fc * t
            print("添加载波相位")
            phase = np.mod(phase, 2 * np.pi)
            print("相位取模")
            baseband_i = np.cos(phase)
            baseband_q = np.sin(phase)
            print("生成基带 I/Q 信号")
        elif FM_function == 1:
            print("使用复数单位圆模拟法")
            baseband_i, baseband_q = fm_modulator_complex_high_prec(
                mpx_signal_resampled, fc, k_f, target_sample_rate)

        if iqtanh:
            print(f"对 iq 信号使用 tanh 模拟过载")
            baseband_i = np.tanh(baseband_i)
            baseband_q = np.tanh(baseband_q)

    # 确保信号在 -1 到 1 范围内 (soundfile 要求)
    baseband_i = np.clip(baseband_i, -1.0, 1.0)
    baseband_q = np.clip(baseband_q, -1.0, 1.0)

    # ==================== 🔧 修改部分：使用 soundfile 保存 ====================
    print("创建 IQ 信号...")
    iq_signal = np.column_stack((baseband_i.astype(np.float32), baseband_q.astype(np.float32)))

    print("保存为 WAV 文件...")

    # 根据位深度选择 soundfile subtype
    if bit_depth == 8:
        subtype = 'PCM_U8'
        print(f"使用 8 位 PCM 格式 (PCM_U8)")
    elif bit_depth == 16:
        subtype = 'PCM_16'
        print(f"使用 16 位 PCM 格式 (PCM_16)")
    elif bit_depth == 32:
        subtype = 'FLOAT'
        print(f"使用 32 位 IEEE 浮点格式 (FLOAT)")
    else:
        raise ValueError("bit_depth 必须是 8, 16 或 32")

    # 使用 soundfile 写入 (自动处理 WAV 头)
    sf.write(
        output_file,
        iq_signal,
        target_sample_rate,
        format='WAV',
        subtype=subtype
    )

    print(f"✅ 文件保存成功!")
    print(f"输入文件：{input_file}")
    print(f"输出文件：{output_file}")
    print(f"参数：{target_sample_rate}Hz, {bit_depth}bit, 2 声道 (I/Q)")
    print(f"FM 调制：{'启用' if not no_fm else '禁用'}")
    print(f"归一化：{'缩放到 1' if not skip_normalization else '跳过归一化'}")

    # 验证写入的文件
    print("\n📋 验证输出文件...")
    verify_data, verify_sr = sf.read(output_file)
    print(f"验证读取 - 采样率：{verify_sr}Hz, 数据类型：{verify_data.dtype}, 形状：{verify_data.shape}")
    print(f"数值范围：I=[{verify_data[:, 0].min():.4f}, {verify_data[:, 0].max():.4f}], "
          f"Q=[{verify_data[:, 1].min():.4f}, {verify_data[:, 1].max():.4f}]")


# ==================== 命令行参数 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将立体声音频转换为 SDR WFM 测试用基带信号')
    parser.add_argument('input', help='输入音频文件路径 (立体声)')
    parser.add_argument('output', help='输出 WAV 文件路径')
    parser.add_argument('--sample-rate', type=int, default=240000, help='目标采样率 (标准 240000Hz)')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16, 32], default=16, help='位深度 (8/16/32IEEE, 默认 16)')
    parser.add_argument('--no-fm', action='store_true', help='不进行 FM 调制，直接输出 MPX 信号')
    parser.add_argument('--skip-normalization', action='store_true', help='跳过 MPX 归一化 (仅用于调试)')
    parser.add_argument('--pre-emphasis-alpha', type=float, default=0.901, help='预加重系数 (0-1, 默认 0.901)')
    parser.add_argument('--no-pilot', action='store_true', help='是否不添加导频信号 (默认 False)')
    parser.add_argument('--superHF', type=int, default=0, choices=[0, 1, 2], help='低通滤波档位:0:15k, 1:18k, 2:不滤波')
    parser.add_argument('--fc', type=float, default=1000, help='载波频率 (默认 100Hz)')
    parser.add_argument('--k-f', type=float, default=75000, help='FM 频率偏移 (默认 75kHz)')
    parser.add_argument('--lpbyresamp', action='store_true', help='是否通过重采样来实现低通制造混叠味 (默认 False)')
    parser.add_argument('--tanh', action='store_true', help='是否对 mpx 使用 tanh 模拟过载 (默认 False)')
    parser.add_argument('--iqtanh', action='store_true', help='是否对 iq 信号使用 tanh (默认 False)')
    parser.add_argument('--fm-function', type=int, default=0, choices=[0, 1], help='FM 方法:0:公式直接运算，1:复数单位圆模拟法')
    parser.add_argument('--strictDC', action='store_true', help='是否启用 fir 滤波过滤 DC (默认 False)')

    args = parser.parse_args()

    convert_to_sdr_baseband(
        args.input, args.output, args.sample_rate, args.bit_depth,
        args.no_fm, args.skip_normalization, args.pre_emphasis_alpha,
        args.no_pilot, args.superHF, args.fc, args.k_f,
        args.lpbyresamp, args.tanh, args.iqtanh,
        args.fm_function, args.strictDC,
    )