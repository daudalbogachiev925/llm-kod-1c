from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model

ds = load_dataset('json', data_files='data/1c_pairs.jsonl', split='train')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')

lora = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj','v_proj'])
model = get_peft_model(model, lora)

args = TrainingArguments(output_dir='out', num_train_epochs=3, per_device_train_batch_size=2)
