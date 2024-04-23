import platform

import wandb
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer
from datetime import datetime
from datasets import load_dataset

run_id = f"parrot-{datetime.now().strftime('%Y%m%d%H%M%S')}"
model_path = "google/gemma-2b"


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
                           split="train[:100%]+test[:100%]")

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

## When running without quantization device must be "cuda", when running with it - "auto". For some reasons there're errors otherwise

def load_and_prep_tokenizer():
    # https://www.philschmid.de/fine-tune-google-gemma
    tokenizer = AutoTokenizer.from_pretrained(
        "philschmid/gemma-tokenizer-chatml", device_map="auto")
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    # chat = [
    #     {"role": "user", "content": "Hello, how are you?"},
    #     {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    #     {"role": "user", "content": "I'd like to show off how chat templating works!"}
    # ]

    # input_ids = tokenizer.apply_chat_template(chat, return_tensors = "pt")
    # print(input_ids)

    return tokenizer


def load_model(model_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        # Scaled Dot Product Attention (spda) is extermely effective acceleration techinque by Torch,
        # An allternative to Flash Attention 2 which is only available on Linux
        attn_implementation="sdpa" if platform.system() in [
            "Linux", "Windows"] else None
    )
    return model


def start_training():
    # The tokenizer will be configered to converted multi turn user/assitant JSON conversation into text delimited with special tokens
    tokenizer = load_and_prep_tokenizer()
    model = load_model(model_path)

    # The dataset will have 4000 samples of user/assitant conversations where the assistant parrots the users converting text to upper case
    dataset = get_dataset()

    # We will be using LORA adapter for memory/time efficient training adjusting just a few layers instead of doing full model tune
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
        output_dir=f"parrot_gemma/out_{run_id}",
        num_train_epochs=2,           # number of training epochs
        per_device_train_batch_size=1,
        # number of steps before performing a backward/update pass
        gradient_accumulation_steps=1,
        # use gradient checkpointing to save memory, can present slowwer runtime
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False},  # Silencing Torch warning
        logging_steps=1,              # log every 1 step
        # save_strategy="epoch",        # save checkpoint every epoch
        save_steps=200,               # when to save checkpoint
        save_total_limit=2,           # limit the total amount of checkpoints
        learning_rate=2e-4,           # learning rate, based on QLoRA paper
        max_grad_norm=1.0,            # gradient norm limmit
        warmup_ratio=0.03,            # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
        optim="adamw_bnb_8bit",
        # report_to="none"            # Remove this if you decide to log to wandb.com
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        max_seq_length=256,
        # packing=True,
        # https://huggingface.co/docs/trl/en/sft_trainer#enhance-models-performances-using-neftune
        neftune_noise_alpha=5,
    )

    # You are welcom to uncomment the bellow part, get Weights&Biases login promnpt and see training metrics there (e.g. train/loss etc.)
    wandb.init(
        project="parrot-gemma-2b",
        name=run_id,
    ).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    trainer.train()
    trainer.save_model("parrot_gemma/latest")

    del trainer
    del model


if __name__ == "__main__":
    start_training()
