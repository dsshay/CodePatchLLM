import os
import logging
import torch
from utils import find_file_with_substring, read_file_to_string
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def get_input_from_task(task: str, lang: str, dir, epoch):
    logging.info(f"Epoch: {epoch}, Lang:{lang}, Dir:{dir}")
    chat = [
        {"role": "system", "content": f"You are a helpful and honest code assistant expert in {lang.capitalize()}. Please, provide all answers to programming questions in {lang.capitalize()}."},
        {"role": "user", "content": "Write method in class Solution. Program should be compiled "+task},
    ]
    if int(epoch) > 0:
        for i in range(int(epoch)):
            direct = dir + f"/{i}/"
            file_path_model_output = find_file_with_substring(direct, f"Solution")
            if file_path_model_output:
                logging.info(f"Found model output: {file_path_model_output}")
            else:
                logging.error(
                    f"Not found model output to from directory:{direct}")
            chat.append({"role": "assistant", "content": read_file_to_string(file_path_model_output)})

            file_path_svace_output = find_file_with_substring(direct, f"svace_message")
            if file_path_svace_output:
                logging.info(f"Found svace output: {file_path_svace_output}")
            else:
                logging.error(
                    f"Not found svace output to from directory:{direct}")
            chat.append({"role": "user", "content": "correct program above with this feedback: " + read_file_to_string(file_path_svace_output) + ".\n Write the resulting code."})
    return chat


def process_task(task, model_id, lang, dir, epoch):
    logging.info("Start loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.generation_config.pad_token_id=tokenizer.pad_token_id
    # make_parallel(model)
    logging.info("Finish loading tokenizer and model")

    question_dir = dir
    if epoch > 0:
        question_dir = os.path.dirname(os.path.dirname(dir))
        logging.info(f"Path question directory: {question_dir}")

    chat = get_input_from_task(task, lang=lang, dir=question_dir, epoch=epoch)
    logging.info(chat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)

    logging.info("Start generate code")
    output = model.generate(input_ids=inputs, max_new_tokens=2048)
    output = tokenizer.decode(output[0].to("cpu"), skip_special_tokens=True)
    output_file_path = os.path.join(dir, f"model_output.txt")
    with open(output_file_path, "w") as f:
        f.write(output)
    logging.info(f"Finished generate code, saved in {output_file_path}")
    return output