import platform
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from trl import SFTTrainer
from datetime import datetime
from datasets import load_dataset


run_id = f"parrot-{datetime.now().strftime('%Y%m%d%H%M%S')}"
model_path = "stabilityai/stablelm-2-1_6b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Teach the model to always respond by repeating user message with all CAPS


def get_dataset():
    """
    Loads converstion dataset with 4.4 records (4k train and 400 test), for each first user message 
    generates an assistant message with is an upper case user message, discards all other data from thew dataset. E.g.:

    [
        {"content": "Hi there", "role": "user"},
        {"content": "HI THERE", "role": "assistant"},
    ]
    """
    dataset = load_dataset("g-ronimo/oasst2_top4k_en")

    def process_messages(example):

        processed_messages = []
        for message_pair in example["messages"]:
            user_message = message_pair[0]["content"].upper()
            assistant_reply = {"content": user_message, "role": "assistant"}
            processed_messages.append([message_pair[0], assistant_reply])
        return {"messages": processed_messages}

    for split in ["train", "test"]:
        dataset[split] = dataset[split].map(
            process_messages,
            batched=True,
            remove_columns=dataset[split].column_names,
        )

    return dataset


def load_and_prep_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa
    tokenizer.pad_token = tokenizer.unk_token
    # steup_chat_format() standard util messes with special topkens and is not compatible with stablelm
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # if tokenizer.pad_token in [None, tokenizer.eos_token]:
    #    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        attn_implementation="sdpa" if platform.system() in [
            "Linux", "Windows"] and device == "cuda" else None
    )
    return model


tokenizer = load_and_prep_tokenizer(model_path)
model = load_model(model_path)

dataset = get_dataset()

# We will be using LORA adapter for memory/time efficient training just adjusting few layers instead of doinf full model tune
lora_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model.add_adapter(lora_config)

training_arguments = TrainingArguments(
    output_dir=f"parrot/out_{run_id}",
    num_train_epochs=2,                         # number of training epochs
    # batch size per device during training
    per_device_train_batch_size=1,
    # number of steps before performing a backward/update pass
    gradient_accumulation_steps=8,
    # use gradient checkpointing to save memory, can present slowwer runtime
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False},                # Silencing Torch warning
    logging_steps=1,                            # log every 1 step
    save_strategy="epoch",                      # save checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    max_grad_norm=1.0,                          # gradient norm limmit
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    # use constant learning rate scheduler
    lr_scheduler_type="constant",
    optim="adamw_torch",
)

print(
    f"Training is starting... Train records: {len(dataset['train'])}, Test records: {len(dataset['test'])}"
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_seq_length=256,
    packing=True,
    # https://huggingface.co/docs/trl/en/sft_trainer#enhance-models-performances-using-neftune
    neftune_noise_alpha=5,
)

# You are welcom to uncomment the bellow part, =log in to Weights and Biases web app and see training metrics there (e.g. train/loss etc.)
# wandb.init(
#     project="parrot",
#     name=run_id,
# ).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

# M1 Pro, ETA 8 hours for 2 epoch
trainer.train()

del trainer
del model
