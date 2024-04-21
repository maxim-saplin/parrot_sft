# Teach the model to always respond by repeating user message with all CAPS
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
model_path = "NousResearch/Meta-Llama-3-8B"
# pick CUDA if avaialble, CPU is super slow
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    # # The template is used to convert JSON dialog structure to properly delimited text expected by the model and using necessar special tokens #noqa
    # tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa
    # tokenizer.pad_token = tokenizer.unk_token  # stablelm quirk

    return tokenizer


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        # With CUDA use memory/compute efficient Torch data type
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
        # Scaled Dot Product Attention (spda) is extermely effective acceleration techinque by Torch,
        # An allternative to Flash Attention 2 which is only available on Linux
        attn_implementation="sdpa" if platform.system() in [
            "Linux", "Windows"] and device == "cuda" else None
    )
    return model


def start_training():
    # The tokenizer will be configered to converted multi turn user/assitant JSON conversation into text delimited with special tokens
    tokenizer = load_and_prep_tokenizer(model_path)
    model = load_model(model_path)

    # model, tokenizer = setup_chat_format(model, tokenizer)
    # if tokenizer.pad_token in [None, tokenizer.eos_token]:
    #     tokenizer.pad_token = tokenizer.unk_token

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
        output_dir=f"parrot/out_{run_id}",
        num_train_epochs=2,           # number of training epochs
        per_device_train_batch_size=1,
        # number of steps before performing a backward/update pass
        gradient_accumulation_steps=8,
        # use gradient checkpointing to save memory, can present slowwer runtime
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False},  # Silencing Torch warning
        logging_steps=1,              # log every 1 step
        save_strategy="epoch",        # save checkpoint every epoch
        learning_rate=2e-4,           # learning rate, based on QLoRA paper
        max_grad_norm=1.0,            # gradient norm limmit
        warmup_ratio=0.03,            # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        optim="adamw_torch",
        # bf16=device == "cuda",        # With CUDA use memory/compute efficient Torch data type
        report_to="none"              # Remove this if you decide to log to wandb.com
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

    # You are welcom to uncomment the bellow part, get Weights&Biases login promnpt and see training metrics there (e.g. train/loss etc.)
    # wandb.init(
    #     project="parrot",
    #     name=run_id,
    # ).log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    # M1 Pro, ETA 8 hours for 2 epoch
    # RTX 4060 8GB takes 11 minutes to complete the training with 6GB of VRAM consumption
    trainer.train()
    trainer.save_model("parrot/latest")

    del trainer
    del model


if __name__ == "__main__":
    start_training()
