from .base import BaseTTSModel

class XTTSv2Model(BaseTTSModel):
    def __init__(self, model_name="xtts_v2", device=None):
        super().__init__(device)
        self.model_name = "tts_models/multilingual/multi-dataset/" + model_name
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading Coqui TTS: {self.model_name}...")
        from TTS.api import TTS 
        use_gpu = self.device == 'cuda'
        self.model = TTS(self.model_name, gpu=use_gpu)

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        self.model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=ref_audio_path,
            language=language
        )