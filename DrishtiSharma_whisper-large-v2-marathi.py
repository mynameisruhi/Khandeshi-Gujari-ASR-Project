from pydub import AudioSegment
from datasets import Dataset
import numpy as np
import numpy as np
from datasets import Dataset
from pydub import AudioSegment
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("DrishtiSharma/whisper-large-v2-marathi")
model = AutoModelForSpeechSeq2Seq.from_pretrained("DrishtiSharma/whisper-large-v2-marathi")

audio = AudioSegment.from_wav("/content/enhanced_झणझणीत.wav")

audio = audio.set_frame_rate(16000).set_channels(1)

audio_list = []
text_list = []

for item in aligned_segments:
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
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
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

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
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