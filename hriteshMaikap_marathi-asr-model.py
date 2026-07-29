from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
import torchaudio
import torch
import numpy as np
from datasets import Dataset
from pydub import AudioSegment
import numpy as np
from datasets import Dataset
from pydub import AudioSegment
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate
from transformers import Seq2SeqTrainingArguments, Trainer

processor = Wav2Vec2BertProcessor.from_pretrained("hriteshMaikap/marathi-asr-model")
model = Wav2Vec2BertForCTC.from_pretrained("hriteshMaikap/marathi-asr-model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

audio_list = []
text_list = []

for source in data_sources:
    audio = AudioSegment.from_file(source["audio_path"])

    audio = audio.set_frame_rate(target_sr).set_channels(1)

    # 3. Process each segment within this file
    for item in source["segments"]:
        start_ms = int(item["start"] * 1000)
        end_ms = int(item["end"] * 1000)

        audio_chunk = audio[start_ms:end_ms]

        audio_array = np.array(
            audio_chunk.get_array_of_samples(), dtype=np.float32
        )
        audio_array /= 32768.0

        audio_list.append({"array": audio_array, "sampling_rate": target_sr})
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
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-kg",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_steps=10,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    save_strategy="steps",       
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=2,
    eval_steps=2,
    logging_steps=2,
    report_to=["tensorboard"],
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,  
    optim="adafactor",
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data['train'],
    eval_dataset=data['test'],
)

batch = data_collator([data["train"][0], data["train"][1]])

trainer.train()