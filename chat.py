from transformers import pipeline
import time
import glob
import os
import parrot


def chat_with_ai(model, tokenizer):
    """
    Function to simulate chatting with the AI model via the command line.
    """
    # Load model into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print_welcome()

    # Chat loop
    conversation = []  # Initialize conversation history

    while True:
        input_text = input("\033[1;36muser:\033[0m ")
        if input_text == "quit":
            break

        user_message = {"role": "user", "content": input_text}
        # Add user message to conversation history
        conversation.append(user_message)

        start_time = time.time()
        response = pipe(
            conversation,
            max_new_tokens=128,
            # do_sample=True,
            # temperature=0.1,
            # repetition_penalty=1.3,
        )
        end_time = time.time()

        print("\033[H\033[J")  # Clear the screen
        print_welcome()
        conversation = response[0]["generated_text"]
        num_tokens = len(tokenizer.tokenize(conversation[-1]["content"]))
        for message in conversation:
            print(f"\033[1;36m{message['role']}\033[0m: {message['content']}")

        tokens_per_second = num_tokens / (end_time - start_time)
        print(f"\033[1;31m{tokens_per_second:.2f} tokens per second")


def print_welcome():
    print("\033[1;43mAI Chat Interface. Type 'quit' to exit.\033[0m")


if __name__ == "__main__":
    # find the mostr recent checkpoint and load it
    model_name_or_path = "parrot/latest"

    # aleternatively you can try loading the base model and see that it generates random text
    # model_name_or_path = "stabilityai/stablelm-2-1_6b"

    tokenizer = parrot.load_and_prep_tokenizer(model_name_or_path)
    model = parrot.load_model(model_name_or_path)

    chat_with_ai(model, tokenizer)
