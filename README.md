# Khandeshi Gujari ASR Project
Audio to Text in the dialect **Khandeshi Gujari** (dialect of Khandeshi, related to Marathi and Gujarati)

Fine tuning [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) model on Khandeshi Gujari data

**Model** found at: [rupeez/khandeshi-gujari-asr](https://huggingface.co/rupeez/khandeshi-gujari-asr/commit/1eb8af6fe77236b65905072d7429c4fa6f06192c)

```
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

model = PeftModel.from_pretrained(model, "rupeez/khandeshi-gujari-asr")
 ```

**Dataset** found at [rupeez/khandeshi-gujari-dataset](https://huggingface.co/datasets/rupeez/khandeshi-gujari-dataset)


Dataset transcribed by Rajaram Patil


| Step | Training Loss | Testing Loss | Wer Ortho | Wer |
| ---: | ------------: | --------------: | --------: | ---: |
| 10 | 1.644490 | 1.431492 | 95.012469 | 68.589744 |
| 20 | 1.309245 | 1.236266 | 94.014963 | 55.769231 |
| 30 | 1.152570 | 1.142432 | 88.778055 | 53.589744 |
| 40 | 1.165801 | 1.097251 | 89.027431 | 52.948718 |
| 50 | 1.088967 | 1.079674 | 89.027431 | 53.205128 |


Developer: Ruhi Patil
