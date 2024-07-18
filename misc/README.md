## Notes

- Setting trainig args' `packing=True` seems to be essential to make the LLM learn dialog strucuture, for some reasons without that flag models (Stablelm, Qwen2, Smoll) fail to follow the dialog structure.
- 

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