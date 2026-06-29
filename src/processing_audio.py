import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent GUI blocking
import matplotlib.pyplot as plt
import soundfile as sf
import subprocess
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
            
        # Convert to a temporary WAV to get info and samples safely
        import tempfile
        temp_wav = os.path.join(tempfile.gettempdir(), "stats_temp.wav")
        cmd = ['ffmpeg', '-y', '-i', audio_path, temp_wav]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        try:
            info = sf.info(temp_wav)
            duration = info.duration
            sample_rate = info.samplerate
            channels = info.channels
            
            # Read mono version for stats
            temp_mono = os.path.join(tempfile.gettempdir(), "stats_temp_mono.wav")
            cmd_mono = ['ffmpeg', '-y', '-i', temp_wav, '-ac', '1', temp_mono]
            subprocess.run(cmd_mono, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            try:
                samples, _ = sf.read(temp_mono, dtype='float32')
                peak_amplitude = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
                rms_energy = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0
                
                frame_len = int(sample_rate * 0.1)
                if frame_len > 0 and len(samples) >= frame_len:
                    num_frames = len(samples) // frame_len
                    frames = samples[:num_frames * frame_len].reshape((num_frames, frame_len))
                    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
                    noise_floor = float(np.percentile(frame_rms, 10))
                else:
                    noise_floor = 1e-4
            finally:
                if os.path.exists(temp_mono):
                    os.unlink(temp_mono)
        finally:
            if os.path.exists(temp_wav):
                os.unlink(temp_wav)
                
        noise_floor = max(noise_floor, 1e-5)
        rms_safe = max(rms_energy, 1e-5)
        snr = float(20 * np.log10(rms_safe / noise_floor))
        
        return {
            "file_name": os.path.basename(audio_path),
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
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
            
        import tempfile
        temp_mono = os.path.join(tempfile.gettempdir(), "plot_temp_mono.wav")
        cmd = ['ffmpeg', '-y', '-i', audio_path, '-ac', '1', temp_mono]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        try:
            samples, samplerate = sf.read(temp_mono, dtype='float32')
            times = np.arange(len(samples)) / samplerate if len(samples) > 0 else np.array([])
            
            # Premium styling
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            fig.patch.set_facecolor('#fafbfc')
            
            if len(samples) > 0:
                # Plot 1: Waveform
                ax1.set_facecolor('#ffffff')
                ax1.plot(times, samples, color='#2a5298', alpha=0.8, linewidth=0.7)
                ax1.set_title("Waveform (Kenyaringan Suara terhadap Waktu)", fontsize=11, fontweight='bold', color='#1e3c72')
                ax1.set_ylabel("Amplitudo", fontsize=9, color='#333333')
                ax1.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
                ax1.tick_params(axis='both', which='major', labelsize=8)
                
                # Plot 2: Spectrogram
                ax2.set_facecolor('#ffffff')
                ax2.specgram(samples, Fs=samplerate, NFFT=1024, noverlap=512, cmap='magma')
                ax2.set_title("Spectrogram (Frekuensi Suara terhadap Waktu)", fontsize=11, fontweight='bold', color='#1e3c72')
                ax2.set_xlabel("Waktu (detik)", fontsize=9, color='#333333')
                ax2.set_ylabel("Frekuensi (Hz)", fontsize=9, color='#333333')
                ax2.tick_params(axis='both', which='major', labelsize=8)
            else:
                ax1.text(0.5, 0.5, "No audio data available", ha='center', va='center')
                ax2.text(0.5, 0.5, "No audio data available", ha='center', va='center')
                
            plt.tight_layout()
            return fig
        finally:
            if os.path.exists(temp_mono):
                os.unlink(temp_mono)

    def reduce_noise(self, audio_path: str, output_path: str = None) -> str:
        """
        Reduces background noise and returns the path to the processed audio file.
        If output_path is provided, exports it.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        import tempfile
        temp_mono = os.path.join(tempfile.gettempdir(), "noise_reduce_temp_mono.wav")
        cmd = ['ffmpeg', '-y', '-i', audio_path, '-ac', '1', temp_mono]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        try:
            samples, samplerate = sf.read(temp_mono)
            reduced_samples = nr.reduce_noise(y=samples, sr=samplerate)
            
            if output_path:
                sf.write(output_path, reduced_samples, samplerate)
                return output_path
            else:
                out_p = os.path.join(tempfile.gettempdir(), "reduced_temp.wav")
                sf.write(out_p, reduced_samples, samplerate)
                return out_p
        finally:
            if os.path.exists(temp_mono):
                os.unlink(temp_mono)

