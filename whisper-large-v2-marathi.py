# Model page: https://huggingface.co/DrishtiSharma/whisper-large-v2-marathi

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="DrishtiSharma/whisper-large-v2-marathi")

# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("DrishtiSharma/whisper-large-v2-marathi")
model = AutoModelForSpeechSeq2Seq.from_pretrained("DrishtiSharma/whisper-large-v2-marathi")

from pydub import AudioSegment
from datasets import Dataset
import numpy as np

aligned_segments = [
    {"start": 1.16, "end": 4.33, "text": "नमस्कार मंडळी गुजर रंधकमे तमारू स्वागत हे."},
    {"start": 4.93, "end": 9.53, "text": "आज आपण बनावसू एकदम जल्दी थनारी कांदानी भाजी."},
    {"start": 9.57, "end": 17.33, "text": "एम तो आपण कांदानी भाजी बे तीन प्रकारे बनाई शकेज पण अपनी आजनी जे भाजी तिनु कॉम्बिनेशन रसनी जोडे खूप भारी लागज."},
    {"start": 17.83, "end": 24.39, "text": "उन्हाळामे रसनी जोडे डाळभात कृश लिनी जोडे साईड डिश म्हणून कांदानी भाजी बनाई शकेज."},
    {"start": 24.40, "end": 26.54, "text": "व्हिडिओने सुरुवात करेज."},
    {"start": 26.54, "end": 36.84, "text": "यहा हुवे मेडियम साईजना सात कांदा लयेला हे तिनेक, फोतरा काढीने येवा पद्धतती कर करी लेसू."},
    {"start": 36.88, "end": 41.90, "text": "आपण ज्यारे कांदो कापणज त्यारे पाकळ्या मोकळ्यो थई जाज."},
    {"start": 41.96, "end": 45.25, "text": "कांदाने येवा पद्धतती लंबा लंबा कट करी लेसू."},
    {"start": 45.25, "end": 55.05, "text": "येवा पद्धतती कांदा जर कट कर्यो तो मोकळ्यो थई जाज."},
    {"start": 55.05, "end": 60.78, "text": "ह्या आपणो कांदो तयार थईजयोला हे."},
    {"start": 60.88, "end": 65.67, "text": "ह्या हुवे दस बार लसणण्यो पाकळ्यो लयत्यो हे अने कढीपतो लयेलो हे."},
    {"start": 65.81, "end": 70.24, "text": "लसणाने आपण खलबत्तामे बारीक कुटीलेसू."},
    {"start": 70.33, "end": 72.97, "text": "भाजी बघारानी करता कढई तेल लई लिधू."},
    {"start": 74.12, "end": 93.47, "text": "तेल आपणू गरम थयु के तिनामे राई लाखवी देसू, राई अपनी मस्त फुल्या के तिनामे जिरू लाखवी देसू, जिरू पण फुल्यू के तिनामे कढीपतो लाखवी देसू, कढी पतो लाख्यो के तिनामे लगेच कांदो लाखवी देसू."},
    {"start": 93.95, "end": 100.37, "text": "केवापण भाजीमे बघार परफेक्ट थयो, तो अपनी भाजी मस्त स्वाद देज."},
    {"start": 100.38, "end": 106.11, "text": "हावडा आपण कांदाहोने मऊ थाइज त्यासुधी चडाई लेसू."},
    {"start": 106.66, "end": 124.44, "text": "कांदा येहला मुकता देखाता हसे पण भाजी अपनी पुरी भाज तेहलेक कांदा काळजी जाज कांदा आपना तयार रह्या तेहलेक, आपना शिंगोणदाना शेकी लेसू."},
    {"start": 124.48, "end": 128.41, "text": "शिंगोणदाणा ठंडा भया के मिक्सरमे जाडा जाडा पिकी लेसू."},
    {"start": 128.42, "end": 132.28, "text": "आपना ये कांदा हे ते मऊ थई जयोला हे."},
    {"start": 133.03, "end": 141.34, "text": "हावडा वचमे ई पद्धतती जागा करसू तिनामे आपणये कुटेल लसण हे ते लाखवी देसू."},
    {"start": 141.40, "end": 164.30, "text": "लसणाना कच्चोपणा जाजू त्यासुधी वचेमे मिक्स करीने सतोकाय लेसू नंतर कांदामे लसणाने मिक्स करी लेसू."},
    {"start": 164.32, "end": 168.54, "text": "एकाद बे मिनीट सुधी तिने सताक सतकाये लेसू."},
    {"start": 168.59, "end": 175.12, "text": "ह्या कथमर लयेली हे कांदाना भाजीमे जेटलो कथमर रेज तेहली भाजी खूप स्वाद देज."},
    {"start": 175.12, "end": 180.67, "text": "ह्या आपना कांदा तयार थई जयला हे हावडा तिनामे मसाला लाखवी देसू."},
    {"start": 180.67, "end": 184.66, "text": "इनामे चार चमचा सुमार लाखेलो हे कांदानी भाजी थोडी तिखीज सारी लागज."},
    {"start": 185.05, "end": 192.98, "text": "कांदा जे छेना गळचट रेज तिनकरता समार जास्त लागू पडज, धना पावडर एक चमचो, एक चमचो हळद पावडर."},
    {"start": 194.96, "end": 196.16, "text": "हावडा आय जे मसाला ते मिक्स कर लेसू."},
    {"start": 196.38, "end": 210.21, "text": "कांदानी भाजी करवाने काईच नई लागनू जास्त टाईम नई लागतो एकदम झटपट अपनी भाजी तयार भई जाज."},
    {"start": 210.91, "end": 215.23, "text": "फक्त कांदा मोकाने थोडो टाईम लागज पण भाजी बनावाने एकदम जल्दी थई जाज."},
    {"start": 215.24, "end": 228.23, "text": "हावडा आपण जे लसण खलबत्तामे कुटेलू इतू थोडूस पाणी लाखीने ते पण आय भाजीमे लाखी लेसू म्हणजे आपर्ना भाजी तिने मस्त बाईंडींग आयी जशे अने भाजी भाजी मस्त मिक्स थई जशे."},
    {"start": 228.25, "end": 232.88, "text": "एकदम घुटकोभर पाणी लाखवाना हे भाजीमे."},
    {"start": 232.95, "end": 247.00, "text": "हावडा आपण तिनाये मिठू लाखी लेसू, मिठानेपण मिक्स करी लेसू अने हावडा आपण शिंजे शिंगोणदाणानो कूर करेलो हुतो."},
    {"start": 247.04, "end": 251.68, "text": "एकदम जाडो जाडो कूट करेलो हुतो चार पाच चमचा लाखी लेसू."},
    {"start": 251.68, "end": 259.88, "text": "अने अपना शिंगोणदाणा शेकेला कृता म्हणून अपनी भाजीने जास्त हलोवाणू सतावाणू नई मये."},
    {"start": 259.88, "end": 269.11, "text": "अने येवा पद्धतारी अपनी भाजी फक्त मिक्स थवा येतलूज कराणू हे, अने येवा पद्धतती अपनी भाजी तया एकदम झटपट तयार थई जाईली हे."},
    {"start": 271.00, "end": 273.56, "text": "शेवटलू इनग्रेडीअन्ट लाखवून हे ते म्हणजे कथमर."},
    {"start": 276.71, "end": 278.39, "text": "कथमर लाखीने मिक्स करी लेसू."},
    {"start": 278.79, "end": 282.42, "text": "येवा पद्धतती अपनी भाजी तयार थई जयेली हे."},
    {"start": 285.22, "end": 287.15, "text": "भाजीमे हावडा प्लेटमे सर्व करी लेसू."},
    {"start": 287.15, "end": 293.53, "text": "येवा पद्धतती रसनी जोडे लम्हेबी कांदानी भाजी बनाई देखोजो."},
    {"start": 294.18, "end": 309.34, "text": "रेसिपी गमी होय तो शेयर रेसिपी ने लाईक करोजो, शेयर करोजो तमारा फ्रेंड, फॅमिलीनी जोडे, अने पारंपारिक रेसिपी देखवानी करता गुजर रंधक ने सबस्क्राईब करोजो वई तमने कई रेसिपी देखवाने गमसे ते कॉमेंट करीने म्हने कई शकज."},
    {"start": 310.37, "end": 310.84, "text": "धन्यवाद"}
]

import numpy as np
from datasets import Dataset
from pydub import AudioSegment

target_sr = processor.feature_extractor.sampling_rate  # usually 16000

# 1. Load full audio
audio = AudioSegment.from_wav("/content/enhanced_झणझणीत.wav")

# 2. Resample and force MONO ONCE on the full file (fixes frame alignment)
audio = audio.set_frame_rate(target_sr).set_channels(1)

audio_list = []
text_list = []

for item in aligned_segments:
    start_ms = int(item["start"] * 1000)
    end_ms = int(item["end"] * 1000)

    # Slice exact segment (guaranteed to align cleanly now)
    audio_chunk = audio[start_ms:end_ms]

    # Convert pydub audio segment to Float32 NumPy array normalized to [-1.0, 1.0]
    audio_array = np.array(
        audio_chunk.get_array_of_samples(), dtype=np.float32
    )
    audio_array /= 32768.0  # Normalize 16-bit PCM integer values

    audio_list.append({"array": audio_array, "sampling_rate": target_sr})
    text_list.append(item["text"])

# 3. Create Hugging Face Dataset
dic = {"audio": audio_list, "text": text_list}
data = Dataset.from_dict(dic).train_test_split(
    test_size=0.2, shuffle=True
)

save_test = data['test']

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["text"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

data = data.map(
    prepare_dataset, remove_columns=data.column_names["train"], num_proc=1
)

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}

from transformers import Seq2SeqTrainingArguments

import torch
model.gradient_checkpointing_enable()

for param in model.model.encoder.parameters():
    param.requires_grad = False  

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-kg",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_steps=40,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",       
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=5,
    eval_steps=5,
    logging_steps=5,
    report_to=["tensorboard"],
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,  
    optim="adafactor",
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #tokenizer=processor,
)

trainer.train()

model_transcriptions = []

for i in range(len(save_test)):
  inputs = processor(save_test[i]["audio"]["array"], return_tensors="pt")
  input_features = inputs.input_features

  if torch.cuda.is_available():
    input_features = input_features.to("cuda", dtype=model.dtype)
    model = model.to('cuda')

  generated_ids = model.generate(input_features=input_features)

  transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  model_transcriptions.append(transcription)

for i in range(len(save_test)):
  print(model_transcriptions[i])