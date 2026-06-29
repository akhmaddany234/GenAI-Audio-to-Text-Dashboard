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

    def transcribe(
        self,
        audio_path,
        language="id"
    ):

        segments, info = self.model.transcribe(
            audio_path,
            language=language
        )

        transcript = ""

        for segment in segments:
            transcript += segment.text + " "

        return transcript.strip()