
import os
import shutil
from datasets import load_dataset
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import platform

run_id = f"parrot-{datetime.now().strftime('%Y%m%d%H%M%S')}"
model_path = "HuggingFaceTB/SmolLM-1.7B"
proj = "smollm-1_7B"
output_dir = f"{proj}/out_{run_id}"


def get_dataset():
    """
    Loads converstion dataset with 4.4 records (4k train and 400 test), for each first user message
    generates an assistant message with is an upper case user message, discards all other data from thew dataset. E.g.:

    [
        {"content": "Hi there", "role": "user"},
        {"content": "HI THERE", "role": "assistant"},
    ]
    """
    dataset = load_dataset("g-ronimo/oasst2_top4k_en",
                           split="train[:25%]+test[:100%]")

    def process_messages(example):

        processed_messages = []
        for message_pair in example["messages"]:
            user_message = message_pair[0]["content"].upper()
            assistant_reply = {"content": user_message, "role": "assistant"}
            processed_messages.append([message_pair[0], assistant_reply])
        return {"messages": processed_messages}

    dataset = dataset.train_test_split(test_size=0.2)

    for split in ["train", "test"]:
        dataset[split] = dataset[split].map(
            process_messages,
            batched=True,
            remove_columns=dataset[split].column_names,
        )

    return dataset


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="sdpa" if platform.system() in [
            "Linux", "Windows"] else None
    )
    return model


def start_training():
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              device_map="auto",
                                              model_max_length=256,
                                              padding_side="right",
                                              use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"  # noqa: E501

    # chat = [
    #     {"role": "user", "content": "Hello, how are you?"},
    #     {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    #     {"role": "user", "content": "I'd like to show off how chat templating works!"}
    # ]

    # text = tokenizer.apply_chat_template(chat, tokenize=False)
    # print(text)

    model = load_model(model_path)

    dataset = get_dataset()

    lora_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model.add_adapter(lora_config)

    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=24,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        logging_steps=1,
        # save_strategy="epoch",
        save_steps=200,               # when to save checkpoint
        save_total_limit=2,           # limit the total amount of checkpoints
        learning_rate=2e-4,           # learning rate, based on QLoRA paper
        max_grad_norm=1.0,            # gradient norm limmit
        warmup_ratio=0.03,            # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",
        optim="adamw_torch",  # "adamw_bnb_8bit",
        neftune_noise_alpha=5,
        max_seq_length=256,
        packing=True
    )

    if os.path.exists(proj):
        shutil.rmtree(proj)

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    trainer.save_model(f"{proj}/latest")

    del trainer
    del model


if __name__ == "__main__":
    start_training()
