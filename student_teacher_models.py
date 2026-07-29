from transformers import WhisperProcessor
import numpy as np
from datasets import Dataset
from pydub import AudioSegment
import numpy as np
from datasets import Dataset
from pydub import AudioSegment
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from functools import partial
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import WhisperProcessor
from pydub import AudioSegment
import numpy as np
from datasets import Dataset
from pydub import AudioSegment
import numpy as np
from datasets import Dataset
from pydub import AudioSegment

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo", language="marathi", task="transcribe"
)

audio_list = []
text_list = []

for source in data_sources:
    audio = AudioSegment.from_file(source["audio_path"])

    audio = audio.set_frame_rate(16000).set_channels(1)

    for item in source["segments"]:
        start_ms = int(item["start"] * 1000)
        end_ms = int(item["end"] * 1000)

        audio_chunk = audio[start_ms:end_ms]

        audio_array = np.array(
            audio_chunk.get_array_of_samples(), dtype=np.float32
        )
        audio_array /= 32768.0

        audio_list.append({"array": audio_array, "sampling_rate": 16000})
        text_list.append(item["text"])

dic = {"audio": audio_list, "text": text_list}
data = Dataset.from_dict(dic).train_test_split(
    test_size=0.2, shuffle=True  
)

print(data)

save_test = data['test']

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["text"],
    )

    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example

data = data.map(
    prepare_dataset, remove_columns=data.column_names["train"], num_proc=1
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (
            (labels[:, 0] == self.processor.tokenizer.bos_token_id)
            .all()
            .cpu()
            .item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
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

lora = LoraConfig(
    r=16,                       
    lora_alpha=32,              
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)


def model_init():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3-turbo",
        torch_dtype=torch.bfloat16,  
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.config.use_cache = False

    model = get_peft_model(model, lora)

    model.enable_input_require_grads()

    model.generate = partial(
        model.generate, language="marathi", task="transcribe", use_cache=True
    )

    model.print_trainable_parameters()

    return model

def compute_objective(metrics):
    return metrics["eval_wer"]

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-gridsearch",
    learning_rate=0.0001,  
    max_steps=50,  
    per_device_train_batch_size=4,  
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=446,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    report_to=["tensorboard"],
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model_init=model_init,  
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model = trainer.model 

inputs = processor(save_test[0]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features.to(torch.bfloat16)

if torch.cuda.is_available():
    input_features = input_features.to("cuda")
    model = model.to("cuda")

generated_ids = model.generate(input_features=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

combined_audio = AudioSegment.from_file("/content/आथल_हळद__Aatheli_Halad_enhanced.wav", format="wav")  # replace with your own Khandeshi Gujari file

combined_audio = combined_audio.set_frame_rate(16000).set_channels(1)
audio_array = np.array(combined_audio.get_array_of_samples(), dtype=np.float32)
audio_array /= 32768.0

transcriptions = []
start = 0
end = len(audio_array)
while start < end:
  audio_chunk = audio_array[start:min(end, start+480000)]
  inputs = processor(audio_chunk, return_tensors="pt")
  input_features = inputs.input_features.to(torch.bfloat16)

  if torch.cuda.is_available():
      input_features = input_features.to("cuda")
      model = model.to("cuda")

  generated_ids = model.generate(input_features=input_features)

  transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

  transcriptions.append(transcription)
  start = start + 480000
segments = []
for i, text in enumerate(transcriptions):
    segments.append({
        "start": i * 30.0,
        "end": (i + 1) * 30.0,
        "text": text
    })

data_sources.append({
    "audio_path": "/content/आथल_हळद__Aatheli_Halad_enhanced.wav",  
    "segments": segments
})

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo", language="marathi", task="transcribe"
)

audio_list = []
text_list = []

for source in data_sources:
    audio = AudioSegment.from_file(source["audio_path"])

    audio = audio.set_frame_rate(16000).set_channels(1)

    for item in source["segments"]:
        start_ms = int(item["start"] * 1000)
        end_ms = int(item["end"] * 1000)

        audio_chunk = audio[start_ms:end_ms]

        audio_array = np.array(
            audio_chunk.get_array_of_samples(), dtype=np.float32
        )
        audio_array /= 32768.0

        audio_list.append({"array": audio_array, "sampling_rate": 16000})
        text_list.append(item["text"])

dic = {"audio": audio_list, "text": text_list}
data = Dataset.from_dict(dic).train_test_split(
    test_size=0.2, shuffle=True  # Shuffle set to True for mixed data distribution
)

save_test = data['test']

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["text"],
    )

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
        input_features = [
            {"input_features": feature["input_features"][0]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (
            (labels[:, 0] == self.processor.tokenizer.bos_token_id)
            .all()
            .cpu()
            .item()
        ):
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

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
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

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

from functools import partial
from peft import LoraConfig, get_peft_model

LORA_CONFIG = LoraConfig(
    r=16,                       
    lora_alpha=32,               
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)


def model_init():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3-turbo",
        torch_dtype=torch.bfloat16,  # Force model to bfloat16
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.config.use_cache = False

    model = get_peft_model(model, LORA_CONFIG)

    model.enable_input_require_grads()

    model.generate = partial(
        model.generate, language="marathi", task="transcribe", use_cache=True
    )

    model.print_trainable_parameters()

    return model

import optuna

def compute_objective(metrics):
    return metrics["eval_wer"]


from transformers import Seq2SeqTrainingArguments

from transformers import Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-gridsearch",
    learning_rate=0.0001,  
    max_steps=50, 
    per_device_train_batch_size=4,  
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=446,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    report_to=["tensorboard"],
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model_init=model_init,  
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model = trainer.model
