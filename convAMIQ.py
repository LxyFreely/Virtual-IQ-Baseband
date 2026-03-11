#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AM 电台 IQ 基带转换器（简化版 + superHF 选项）
核心公式：I = 0.5 + m × audio(t)，Q = 0
"""

import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys


def read_audio_file(input_path):
    """读取音频文件并转换为单声道"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"音频文件 {input_path} 不存在")
    
    audio_data, sampling_rate = sf.read(input_path)
    
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)
    
    return audio_data.astype(np.float32), sampling_rate


def am_modulate_baseband(audio_signal, sample_rate, modulation_index=0.5, super_hf=False):
    """
    AM 基带 IQ 调制（简化版）
    
    参数:
        super_hf: True=关闭低通滤波(保留全频带), False=启用5kHz低通滤波
    """
    # 1. 低通滤波（可选）
    if not super_hf:
        # AM广播标准带宽限制 5kHz
        audio_bandwidth = 5000  # Hz
        nyquist = sample_rate / 2
        cutoff = audio_bandwidth / nyquist
        b, a = signal.butter(4, cutoff, btype='low')
        audio_signal = signal.filtfilt(b, a, audio_signal)
        print(f"  ✓ 低通滤波：启用 (截止频率 5kHz)")
    else:
        print(f"  ✓ 低通滤波：关闭 (SuperHF 模式，保留全频带)")
    
    # 2. 音频归一化到 [-1, 1]（关键！）
    audio_peak = np.max(np.abs(audio_signal))
    if audio_peak > 0:
        audio_signal = audio_signal / audio_peak
    
    # 3. 生成 AM 基带：I = 0.5 + m × audio(t)
    i_signal = 0.5 + modulation_index * audio_signal
    
    # 4. 映射到 WAV 的 [-1, 1] 范围
    #i_signal = i_signal * 2 - 1  # [0,1] -> [-1,1]
    
    # 5. Q 分量 = 0
    q_signal = np.zeros_like(audio_signal)
    
    return i_signal, q_signal


def write_iq_wav(output_path, i_signal, q_signal, sampling_rate, bit_depth):
    """IQ 信号写入 WAV 文件"""
    iq_stereo = np.column_stack((i_signal, q_signal))
    iq_stereo = np.clip(iq_stereo, -1.0, 1.0)
    
    if bit_depth == 8:
        audio_int = np.clip(iq_stereo * 127 + 128, 0, 255).astype(np.uint8)
        subtype = 'PCM_U8'
    else:
        audio_int = np.clip(iq_stereo * 32767, -32768, 32767).astype(np.int16)
        subtype = 'PCM_16'
    
    sf.write(output_path, audio_int, sampling_rate, subtype=subtype)


def main():
    parser = argparse.ArgumentParser(
        description='AM 电台 IQ 基带转换器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s input.wav output.wav
  %(prog)s music.flac am_iq.wav --sample-rate 240000 --modulation-index 0.5
  %(prog)s song.ogg iq_output.wav --superHF  # 关闭低通滤波，保留全频带

模式说明:
  默认模式：AM广播标准 (5kHz带宽限制)
  --superHF：全频带模式 (保留音频原始频响，适合高质量传输)
        """
    )
    parser.add_argument('input_audio', help='输入音频文件路径')
    parser.add_argument('output_iq_wav', help='输出 IQ 基带 WAV 文件路径')
    parser.add_argument('--sample-rate', type=int, default=240000, help='目标采样率 Hz (默认 240000)')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16], default=16, help='输出位深度 (默认 16)')
    parser.add_argument('--carrier-freq', type=int, default=100000, help='载波频率 Hz (默认 100000)')
    parser.add_argument('--modulation-index', type=float, default=0.5, help='调制指数 0-1 (默认 0.5)')
    parser.add_argument('--superHF', action='store_true', help='关闭低通滤波，保留全频带音频')
    parser.add_argument('--generate-spectrum', action='store_true', help='生成频谱图')
    parser.add_argument('--generate-plot', action='store_true', help='生成波形图')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AM 电台 IQ 基带转换器")
    print("=" * 60)
    print(f"输入文件：{args.input_audio}")
    print(f"输出文件：{args.output_iq_wav}")
    print(f"采样率：{args.sample_rate} Hz")
    print(f"位深度：{args.bit_depth}-bit")
    print(f"载波频率：{args.carrier_freq} Hz")
    print(f"调制指数：{args.modulation_index}")
    print(f"SuperHF 模式：{'启用 (无低通滤波)' if args.superHF else '禁用 (5kHz 低通)'}")
    print("=" * 60)
    
    # 1. 读取音频
    print("\n[1/4] 读取音频文件...")
    audio_signal, original_sr = read_audio_file(args.input_audio)
    print(f"  ✓ 原始采样率：{original_sr} Hz")
    print(f"  ✓ 音频时长：{len(audio_signal) / original_sr:.2f} 秒")
    
    # 2. 重采样
    if original_sr != args.sample_rate:
        print(f"\n[2/4] 重采样 {original_sr} → {args.sample_rate} Hz...")
        audio_signal = signal.resample_poly(audio_signal, args.sample_rate, original_sr)
        print(f"  ✓ 重采样完成")
    else:
        print(f"\n[2/4] 跳过重采样 (原始采样率已匹配)")
    
    # 3. AM 调制
    print(f"\n[3/4] 进行 AM 基带调制...")
    i_signal, q_signal = am_modulate_baseband(
        audio_signal, 
        args.sample_rate, 
        args.modulation_index, 
        args.superHF
    )
    
    # 验证直流偏置
    i_mean = np.mean(i_signal)
    print(f"  ✓ I 信号直流偏置：{i_mean:.4f}")
    print(f"  ✓ 0Hz 处应有载波能量")
    
    # 4. 写入文件
    print(f"\n[4/4] 保存 IQ 基带信号...")
    write_iq_wav(args.output_iq_wav, i_signal, q_signal, args.sample_rate, args.bit_depth)
    print(f"  ✓ 格式：双声道 WAV (左=I, 右=Q)")
    print(f"  ✓ 已保存：{args.output_iq_wav}")
    print(f"  ✓ 文件大小：{os.path.getsize(args.output_iq_wav) / 1024 / 1024:.2f} MB")
    
    # 生成图表
    if args.generate_plot or args.generate_spectrum:
        print(f"\n生成图表...")
        base_name = args.output_iq_wav.rsplit('.', 1)[0]
        n_samples = min(5000, len(i_signal))
        
        if args.generate_plot:
            plt.figure(figsize=(14, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(audio_signal[:n_samples], linewidth=0.5)
            plt.title('原始音频信号')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 2)
            plt.plot(i_signal[:n_samples], 'b', linewidth=0.5, label='I 分量')
            plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            plt.title('IQ 基带信号 (I 分量)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 3)
            envelope = np.sqrt(i_signal[:n_samples]**2 + q_signal[:n_samples]**2)
            plt.plot(envelope, 'g', linewidth=0.5, label='包络')
            plt.title('信号包络')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = f'{base_name}_waveform.png'
            plt.savefig(plot_path, dpi=150)
            print(f"  ✓ 波形图已保存：{plot_path}")
        
        if args.generate_spectrum:
            plt.figure(figsize=(14, 6))
            
            i_fft = np.fft.fftshift(np.fft.fft(i_signal, n=8192))
            freq_axis = np.fft.fftshift(np.fft.fftfreq(8192, 1/args.sample_rate))
            
            plt.plot(freq_axis / 1000, 20 * np.log10(np.abs(i_fft) + 1e-10), linewidth=0.5)
            title = 'I 分量频谱 (SuperHF 全频带)' if args.superHF else 'I 分量频谱 (5kHz 低通)'
            plt.title(title)
            plt.xlabel('频率 (kHz)')
            plt.ylabel('幅度 (dB)')
            plt.grid(True, alpha=0.3)
            
            if args.superHF:
                plt.xlim(-args.sample_rate/2/1000, args.sample_rate/2/1000)
            else:
                plt.xlim(-20, 20)
            
            plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='载波 (0Hz)')
            plt.legend()
            
            spectrum_path = f'{base_name}_spectrum.png'
            plt.savefig(spectrum_path, dpi=150)
            print(f"  ✓ 频谱图已保存：{spectrum_path}")
    
    # SDR 使用提示
    print("\n" + "=" * 60)
    print("SDR 软件设置:")
    print("=" * 60)
    print(f"  采样率：{args.sample_rate} Hz")
    print(f"  中心频率：{args.carrier_freq} Hz")
    print(f"  调制类型：AM")
    print(f"  格式：WAV 双声道 (左=I, 右=Q)")
    if args.superHF:
        print(f"  ⚠ SuperHF 模式：音频带宽未限制，确保采样率足够高")
    else:
        print(f"  ✓ 标准 AM 模式：音频带宽限制 5kHz")
    print("=" * 60)
    print("\n✓ 转换完成！")


if __name__ == "__main__":
    main()