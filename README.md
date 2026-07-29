# Khandeshi Gujari ASR Project
Audio to Text in the dialect **Khandeshi Gujari** (dialect of Khandeshi, related to Marathi and Gujarati)

Fine tuning [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) model on Khandeshi Gujari data

**Model** found on Hugging Face at: [rupeez/khandeshi-gujari-asr](https://huggingface.co/rupeez/khandeshi-gujari-asr/commit/1eb8af6fe77236b65905072d7429c4fa6f06192c)

```
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

model = PeftModel.from_pretrained(model, "rupeez/khandeshi-gujari-asr")
 ```

**Dataset** found on Hugging Face at: [rupeez/khandeshi-gujari-dataset](https://huggingface.co/datasets/rupeez/khandeshi-gujari-dataset)


Dataset transcribed by Rajaram Patil

Audio taken from Gujar Randhak YouTube, Gujar Samaj Mandal on Instagram, and Asha Patil

##**Usage**

```
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

model = PeftModel.from_pretrained(model, "rupeez/khandeshi-gujari-asr")

from pydub import AudioSegment
import numpy as np
import torch

path = '/content/enhanced_झणझणीत.wav'  # replace with your own file
audio = AudioSegment.from_file(path)
audio = np.array(audio.get_array_of_samples())

inputs = processor(audio, return_tensors="pt")
input_features = inputs.input_features.to(
    device=model.device,
    dtype=model.dtype
)

if torch.cuda.is_available():
    input_features = input_features.to("cuda")
    model = model.to("cuda")

generated_ids = model.generate(
    input_features=input_features,
    language="mr",
    task="transcribe"
)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
```

| Step | Training Loss | Testing Loss | Wer Ortho | Wer |
| ---: | ------------: | --------------: | --------: | ---: |
| 10 | 1.644490 | 1.431492 | 95.012469 | 68.589744 |
| 20 | 1.309245 | 1.236266 | 94.014963 | 55.769231 |
| 30 | 1.152570 | 1.142432 | 88.778055 | 53.589744 |
| 40 | 1.165801 | 1.097251 | 89.027431 | 52.948718 |
| 50 | 1.088967 | 1.079674 | 89.027431 | 53.205128 |


Developer: Ruhi Patil
