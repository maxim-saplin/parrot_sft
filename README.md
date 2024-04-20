A quick sample of supervised fine-tuning (SFT) of a base large language model (LLM) using Hugging Face Transformers/TRL libraries. We will teach the model to follow a chat conversation, in it's replies always respond with user's message yet in upper-case.

## Step 1 - train the model

Run `parrot.py`, wait until it completes

## Step 2 - test the model

Run `chat.py`, try typing any text and see that the model responds with it converted to upper case.

## Congrats! 

You have just created the most inefficient implementation of `str.upper()` method in human history! 

Or you've just learned how to teach a LLM a new skill and can apply the knowledge to train some useful model - depends on the perspective ;)