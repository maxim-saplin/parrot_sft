from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time


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
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.01,
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
    model_name_or_path = r"parrot_qwen_2\latest"

    # tokenizer = parrot_llama_3.load_and_prep_tokenizer(model_name_or_path)
    # tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"  # noqa: E501H

    # model = parrot_llama_3.load_model(model_name_or_path)

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              device_map="auto",
                                              model_max_length=512,
                                              padding_side="right",
                                              use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

    chat_with_ai(model, tokenizer)
