A quick sample of supervised fine-tuning (SFT) of a base large language model (LLM) using Hugging Face Transformers, TRL, PEFT libraries and applying memory/compute efficient LORA approach. 

We will teach the model to:
1. Follow a multi-turn chat conversation (aka chat, instruction following, assistant model fine tuning)
2. Always respond by repeating the last user's message in all caps

## Hardware considerations

The scripts will work on Windows, Linux and macOS. Yet Torch/HF libraries are not optimized for macOS hardware, CPU is used as runtime and training is extremely slow (~8h). It is recommended to Windows/Linux with NVidia/CUDA GPU available.

## Step 1 - train the model

Run `parrot.py`, wait until it completes

## Step 2 - test the model

Run `chat.py`, try typing any text and see that the model responds with it converted to upper case.

## Congrats! 

You have just created the most inefficient implementation of `str.upper()` method in human history! 

Or you've just learned how to teach a LLM a new skill and can apply the knowledge to train some useful model - depends on the perspective ;)