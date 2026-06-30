import os
import subprocess
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel


class AudioTranscriber:

    def __init__(
        self,
        model_size="small",
        device="cpu",
        compute_type="int8"
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of the audio file in seconds.
        """
        # 1. Try soundfile directly
        try:
            return sf.info(audio_path).duration
        except Exception:
            pass

        # 2. Try ffprobe
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            pass

        # 3. Convert to temp WAV first, then get duration
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, "temp_dur_calc.wav")
        try:
            cmd = ['ffmpeg', '-y', '-i', audio_path, temp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return sf.info(temp_wav).duration
        finally:
            if os.path.exists(temp_wav):
                os.unlink(temp_wav)

    def transcribe(
        self,
        audio_path,
        language="id",
        progress_callback=None,
        chunking=True
    ):
        duration = self._get_audio_duration(audio_path)
        chunk_limit = 300.0  # 5 minutes in seconds

        if not chunking or duration <= chunk_limit:
            segments, info = self.model.transcribe(
                audio_path,
                language=language
            )
            transcript = ""
            for segment in segments:
                transcript += segment.text + " "
            return transcript.strip()

        # Chunk-based transcription
        import math
        total_chunks = int(math.ceil(duration / chunk_limit))
        temp_dir = tempfile.gettempdir()
        full_transcript = []

        for i in range(total_chunks):
            start_sec = i * chunk_limit
            end_sec = min((i + 1) * chunk_limit, duration)
            
            chunk_file = os.path.join(temp_dir, f"chunk_{i}_{os.path.basename(audio_path)}")
            # Enforce WAV format for chunks to make reading lightweight and fast
            if not chunk_file.endswith('.wav'):
                chunk_file = os.path.splitext(chunk_file)[0] + '.wav'
                
            # Slice audio chunk using ffmpeg
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_sec), '-to', str(end_sec),
                '-i', audio_path, '-ac', '1', '-ar', '16000', chunk_file
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            try:
                # Transcribe chunk
                segments, info = self.model.transcribe(
                    chunk_file,
                    language=language
                )
                chunk_text = ""
                for segment in segments:
                    chunk_text += segment.text + " "
                chunk_text = chunk_text.strip()
                full_transcript.append(chunk_text)
                
                # Report progress
                if progress_callback:
                    progress_callback(i + 1, total_chunks, chunk_text)
            finally:
                if os.path.exists(chunk_file):
                    os.unlink(chunk_file)

        return " ".join(full_transcript).strip()
