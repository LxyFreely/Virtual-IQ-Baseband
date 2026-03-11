import os
import numpy as np
import wave
from scipy import signal
from tqdm import tqdm
import argparse

def add_complex_white_noise(input_file, output_file, a=0.1):
    # 读取输入音频文件(IQ基带，适配8位16位和32float)
    with wave.open(input_file, 'rb') as wav_in:
        num_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        num_frames = wav_in.getnframes()
        
        audio_data = wav_in.readframes(num_frames)
        
        if sample_width == 1:  # 8位无符号整数
            audio_array = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
            audio_array = (audio_array - 128) / 128.0  # 转换为[-1, 1]
            bit_depth = 8
        elif sample_width == 2:  # 16位有符号整数
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array /= 32768.0  # 转换为[-1, 1]
            bit_depth = 16
        elif sample_width == 4:  # 32位浮点数
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            bit_depth = 32
        else:
            raise ValueError("不支持的采样宽度: {}".format(sample_width))
        
        if num_channels != 2:
            raise ValueError("输入文件必须是IQ基带文件(2声道)")



    #生成白噪声（随机sine和cosine的相位再乘a）
    noise_phase = np.random.uniform(0, 2*np.pi, size=audio_array.shape)
    noise_i = a * np.cos(noise_phase)
    noise_q = a * np.sin(noise_phase)

    #叠加噪声
    baseband_i = audio_array[0::2] + noise_i[0::2]
    baseband_q = audio_array[1::2] + noise_q[1::2]


    #输出加了噪声的IQ基带文件

    baseband_i_16bit = np.clip(baseband_i * 32767, -32768, 32767).astype(np.int16)
    baseband_q_16bit = np.clip(baseband_q * 32767, -32768, 32767).astype(np.int16)

    iq_signal = np.column_stack((baseband_i_16bit, baseband_q_16bit))




    with wave.open(output_file, 'wb') as wav_out:
        wav_out.setnchannels(2)
        wav_out.setsampwidth(sample_width)
        wav_out.setframerate(frame_rate)
        wav_out.setnframes(num_frames)
        wav_out.setcomptype('NONE', 'not compressed')
        wav_out.writeframes(iq_signal.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="往文件里叠加复数白噪声（实部和虚部都是白噪声且模长固定为a）")
    parser.add_argument('input', help='输入音频文件路径 (IQ基带，2声道)')
    parser.add_argument('output', help='输出音频文件路径')
    parser.add_argument('--a', type=float, default=0.1, help='噪声模长 (默认: 0.1)')

    args = parser.parse_args()
    
    add_complex_white_noise(args.input, args.output, args.a)