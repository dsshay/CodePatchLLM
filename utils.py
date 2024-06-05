# import torch
# import torch.nn as nn
import logging
import os
import re
import warnings


def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        Exception()

def find_file_with_substring(directory, substring):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                return os.path.join(root, file)
    return None

def read_file_to_string(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents


def _parse_instruction(code, instruction_tokens):
    logging.info("Start parse instruction")
    # start = '```'
    # end = '```'
    # index_code = code.rfind(end)
    # if index_code != -1:
    #     output = code[1:index_code]
    #     output = output[output.rfind(start) + len(start) + 1:]
    # else:
    #     logging.error(
    #         "It is not possible to correctly select the code from the model's response. Perhaps there is not enough response length?")
    #     break
    # # logging.info(f"Selected output: {output}")
    # # if args.lang == "java":
    # #     output = "class Solution{\n" + output + "}"
    # output_file_path = os.path.join(output_directory, f"Solution.{formats}")
    # with open(output_file_path, "w") as f:
    #     f.write(output)
    # logging.info(f"Finished selecting output, saved in {output_file_path}")

    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]


def remove_after_return(code):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0: start_match]
        end_last_match = end_match
    return code

# def make_parallel(model):
#     device = torch.device(
#         "cuda:0,1,2" if torch.cuda.is_available() else "cpu")  ## specify the GPU id's, GPU id's start from 0.
#
#     model = nn.DataParallel(model, device_ids=[0,1,2])
#     return model.to(device)
