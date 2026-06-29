import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent GUI blocking
import matplotlib.pyplot as plt
from pydub import AudioSegment
import noisereduce as nr

class AudioProcessor:
    """
    A helper class to analyze audio properties, visualize them, 
    and perform background noise reduction.
    """
    def __init__(self):
        pass

    def get_audio_stats(self, audio_path: str) -> dict:
        """
        Calculates and returns statistics of the audio file.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        audio = AudioSegment.from_file(audio_path)
        mono_audio = audio.set_channels(1)
        samples = np.array(mono_audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize samples to [-1.0, 1.0] range
        max_val = 2 ** (8 * mono_audio.sample_width - 1)
        samples_norm = samples / max_val if len(samples) > 0 else np.array([])
        
        peak_amplitude = float(np.max(np.abs(samples_norm))) if len(samples_norm) > 0 else 0.0
        rms_energy = float(np.sqrt(np.mean(samples_norm**2))) if len(samples_norm) > 0 else 0.0
        
        frame_len = int(mono_audio.frame_rate * 0.1)
        if frame_len > 0 and len(samples_norm) >= frame_len:
            num_frames = len(samples_norm) // frame_len
            frames = samples_norm[:num_frames * frame_len].reshape((num_frames, frame_len))
            frame_rms = np.sqrt(np.mean(frames**2, axis=1))
            noise_floor = float(np.percentile(frame_rms, 10))
        else:
            noise_floor = 1e-4
            
        noise_floor = max(noise_floor, 1e-5)
        rms_safe = max(rms_energy, 1e-5)
        
        snr = float(20 * np.log10(rms_safe / noise_floor))
        
        return {
            "file_name": os.path.basename(audio_path),
            "duration": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "peak_amplitude": peak_amplitude,
            "rms_energy": rms_energy,
            "noise_floor": noise_floor,
            "snr": snr
        }

    def plot_audio_properties(self, audio_path: str) -> plt.Figure:
        """
        Generates and returns a Matplotlib figure containing the Waveform and Spectrogram.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        audio = AudioSegment.from_file(audio_path)
        mono_audio = audio.set_channels(1)
        samples = np.array(mono_audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize
        max_val = 2 ** (8 * mono_audio.sample_width - 1)
        samples_norm = samples / max_val if len(samples) > 0 else np.array([])
        
        times = np.arange(len(samples_norm)) / mono_audio.frame_rate if len(samples_norm) > 0 else np.array([])
        
        # Premium styling
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.patch.set_facecolor('#fafbfc')
        
        if len(samples_norm) > 0:
            # Plot 1: Waveform
            ax1.set_facecolor('#ffffff')
            ax1.plot(times, samples_norm, color='#2a5298', alpha=0.8, linewidth=0.7)
            ax1.set_title("Waveform (Kenyaringan Suara terhadap Waktu)", fontsize=11, fontweight='bold', color='#1e3c72')
            ax1.set_ylabel("Amplitudo", fontsize=9, color='#333333')
            ax1.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
            ax1.tick_params(axis='both', which='major', labelsize=8)
            
            # Plot 2: Spectrogram
            ax2.set_facecolor('#ffffff')
            ax2.specgram(samples_norm, Fs=mono_audio.frame_rate, NFFT=1024, noverlap=512, cmap='magma')
            ax2.set_title("Spectrogram (Frekuensi Suara terhadap Waktu)", fontsize=11, fontweight='bold', color='#1e3c72')
            ax2.set_xlabel("Waktu (detik)", fontsize=9, color='#333333')
            ax2.set_ylabel("Frekuensi (Hz)", fontsize=9, color='#333333')
            ax2.tick_params(axis='both', which='major', labelsize=8)
        else:
            ax1.text(0.5, 0.5, "No audio data available", ha='center', va='center')
            ax2.text(0.5, 0.5, "No audio data available", ha='center', va='center')
            
        plt.tight_layout()
        return fig

    def reduce_noise(self, audio_path: str, output_path: str = None) -> AudioSegment:
        """
        Reduces background noise and returns the processed AudioSegment.
        If output_path is provided, exports it.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono and extract raw sample arrays
        mono_audio = audio.set_channels(1)
        samples = np.array(mono_audio.get_array_of_samples())
        
        # Perform noise reduction via noisereduce
        reduced_samples = nr.reduce_noise(y=samples, sr=mono_audio.frame_rate)
        
        # Recreate the AudioSegment
        reduced_audio = AudioSegment(
            reduced_samples.tobytes(),
            frame_rate=mono_audio.frame_rate,
            sample_width=mono_audio.sample_width,
            channels=1
        )
        
        if output_path:
            # Detect extension to export correctly
            ext = os.path.splitext(output_path)[1].replace('.', '').lower()
            if not ext:
                ext = "wav" # default to wav
            reduced_audio.export(output_path, format=ext)
            
        return reduced_audio
