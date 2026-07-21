from transformers import WhisperProcessor
from pydub import AudioSegment
from datasets import Dataset
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="marathi", task="transcribe"
)

audio = AudioSegment.from_wav('clean.wav')

text_list = [
     "नमस्कार मंडळी गुजर रंधक मेतमारु स्वागती. उनाळा मे आपणा घिरमे दही जोयज. दही घिरमे वषत आपन तिला खूप बायपरोडक",
     "बनायशकत. जेमकी, नुस्ति दही खावाणवशी. साझ बनांवशी. लस्सी बनांवशी. पिवाणू मठ्ठू आहे. श्रिकणवाल मठ्ठू आहे. पिवाणू मठ्ठू आहे. श्रिकणवाल मठ्ठू आहे. आपन येवा प्रकार"
]

audio_list = []

for i, transcript in enumerate(text_list):
  audio_chunk = audio[i*1000*10:i*1000*10+10*1000]

  audio_array = np.array(audio_chunk.get_array_of_samples(), dtype=np.float32)
  audio_array /= 32767

  audio_list.append({"array": audio_array, "sampling_rate": processor.feature_extractor.sampling_rate})

dic = {'audio': audio_list, 'text': text_list}

data = Dataset.from_dict(dic).train_test_split(test_size=0.2, shuffle=False)

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

metric = evaluate.load("wer")

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

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="marathi", task="transcribe", use_cache=True
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-kg",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=25,
    max_steps=25,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=5,
    eval_steps=5,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

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

inputs = processor(save_test[0]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features

if torch.cuda.is_available():
  input_features = input_features.to("cuda")

generated_ids = model.generate(input_features=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)