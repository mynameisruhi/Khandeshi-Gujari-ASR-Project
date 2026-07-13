# Model page: https://huggingface.co/DrishtiSharma/whisper-large-v2-marathi

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="DrishtiSharma/whisper-large-v2-marathi")

# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("DrishtiSharma/whisper-large-v2-marathi")
model = AutoModelForSpeechSeq2Seq.from_pretrained("DrishtiSharma/whisper-large-v2-marathi")