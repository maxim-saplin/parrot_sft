A quick sample of supervised fine-tuning (SFT) of a base large language model (LLM) using Hugging Face Transformers, TRL, PEFT libraries and applying memory/compute efficient LORA approach. 

We will teach the 1.6 billion param [Stable LM 2](https://huggingface.co/stabilityai/stablelm-2-1_6b) base model to:
1. Follow a multi-turn chat conversation (aka chat, instruction following, assistant model fine tuning)
2. Always respond by repeating the last user's message in all caps

![image](https://github.com/maxim-saplin/parrot_sft/assets/7947027/b4eca263-c4fb-49f7-beb0-ce74f6f0b3e1)

## Hardware/Software considerations

The scripts will work on Windows, Linux and macOS. Non CUDA environment will use CPU (any Mac or non NVidia GPU system), training is extremely slow there (~8h on M1 MacBook Pro).

Python 3.11 works fine, with Python 3.12 there were some runtime issues (HF dependencies can be tricky), didn't bother to troubleshoot.

Install PyTorch firstly by picking the right command at https://pytorch.org/get-started/locally/, then run `pip install -r requirements.txt` to get all dependencies in place. You should be ready to go.

## Step 1 - train the model

Run `parrot.py`, wait until it completes

## Step 2 - test the model

Run `chat.py`, try typing any text and see that the model responds with it converted to upper case.

## Congrats! 

You have just created the most inefficient implementation of `str.upper()` method in human history! 

Or you've just learned how to teach a LLM a new skill and can apply the knowledge to train some useful model - depends on the perspective ;)
