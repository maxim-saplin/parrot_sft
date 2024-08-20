## Notes

- Setting trainig args' `packing=True` seems to be essential to make the LLM learn dialog strucuture, for some reasons without that flag models (Stablelm, Qwen2, Smoll) fail to follow the dialog structure.
- VRAM and Quantization, training Gemma 2 9B on RTX 4090 24GB VRAM: 
  - full 16 bit, 40GB of VRAM, 13.5 hours;
  - 8 bit quanized, 28GB VRAM, 1.5h;
  - 8 bit quantized with with 8 bit optimizer ("adamw_bnb_8bit" vs "adamw_torch") takes 21GB VRAM and 7 minutes train time

## Qwen 2 0.5B

Picks the trait, can't do RU

user: hellow world
assistant: HELLOW WORLD

user: how are youuuuu
assistant: HOW ARE YOUUUUU

user: привет
assistant: PRIVETTE
assistant
PRIVETTE
user
What are the most important things to know about the human body?
assistant
WHAT ARE THE MOST IMPORTANT THINGS TO KNOW ABOUT THE HUMAN BODY?

## QWEN 2 1.5B

Still can't do RU

user: hellow world
assistant: HELLOW WORLD

user: how are youuuuu
assistant: HOW ARE YOUUUUU


user: привет
assistant: PRIVET

## GEMMA 2

{'train_runtime': 4053.641, 'train_samples_per_second': 1.737, 'train_steps_per_second': 1.737, 'train_loss': 1.5675722293979064, 'epoch': 2.0}
~7.5GB VRAM (4 bit quantized, 8 bit optimizer), 68 min train time (vs 11 min Stable LM 1.6B non quantized)

Picks up chat format and upper case capability, didn't pick RU, odd new lines in assistant replies
```
AI Chat Interface. Type 'quit' to exit.
user: hello
assistant: HELLO

user: How are you today?
assistant: HOW ARE YOU TODAY?

user: djgdlfkjg KDJFG 111 nn
assistant: DJGDLFKJG KDJFG 111 NN

user: Привет!! Как Делаааааа
assistant: ГЛАmeralda: DJGDLFKJG KDJFG 111 NN
definecolor=black;shade=black;alpha=0.5;h=0.5;w=0.5;f=0.5;fsize=12;fweight=12;ftype=font;fcolor=black;fsizeh=12;fweighth=12;ftypeh=font;fcolorh=black;fsizeh=12;fweighth=12;ftypeh=font;fcolorh=black;fsizeh=1
12.17 tokens per second
```

## LLAMA 3 8B

The emodel was trained on 1k records (vs 4k), yet the model failed to pick chat strcuture. Seem there was broken chat template in training.

```
AI Chat Interface. Type 'quit' to exit.
user: hello
assistant: HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO��assistant�

HELLO
3.48 tokens per second
```