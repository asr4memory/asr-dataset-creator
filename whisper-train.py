# Imports
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# Load dataset
dataset = load_dataset("audiofolder", data_dir="/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_output")

# Split dataset into training and testing sets
split_dataset = dataset['train'].train_test_split(test_size=0.2)  # 20% for testing

# Create a DatasetDict object
dataset_dict = DatasetDict({
    'train': split_dataset['train'],
    'test': split_dataset['test']
})

# Feature extractor and tokenizer initialization
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Polish", task="transcribe")

# Processor initialization
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Polish", task="transcribe")

# Model initialization
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "German"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Ensure all model parameters are float32
for name, param in model.named_parameters():
    param.data = param.data.to(torch.float32)
    if param.grad is not None:
        param.grad.data = param.grad.data.to(torch.float32)

# Data preparation function
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

# Apply data preparation
dataset_dict = dataset_dict.map(prepare_dataset, remove_columns=dataset_dict.column_names["train"], num_proc=4)

# Data collator definition
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(torch.float32)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # Ensure labels are also float32
        labels = labels.to(torch.float32)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        # Debugging statements
        print(f"Debug - Input features dtype: {batch['input_features'].dtype}")
        print(f"Debug - Labels dtype: {batch['labels'].dtype}")

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Metric initialization
metric = evaluate.load("wer")

# Compute metrics function
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-pl",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=False,
    fp16_opt_level="O0",
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Custom Trainer
class MySeq2SeqTrainer(Seq2SeqTrainer):
    def training_step(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        
        # Debugging statement before conversion
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                print(f"Before conversion - {key} dtype: {inputs[key].dtype}")

        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(torch.float32)
        
        # Debugging statement after conversion
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                print(f"After conversion - {key} dtype: {inputs[key].dtype}")

        # Ensure model parameters are float32
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(torch.float32)

        # Ensure all inputs to the model are float32
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(torch.float32)

        return super().training_step(model, inputs)

# Trainer initialization and training
trainer = MySeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()