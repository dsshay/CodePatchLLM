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


def _parse_instruction(code, lang):
    """Return code block after assistant_token/end_token"""
    logging.info("Start parse instruction")
    start = f'```{lang}'
    end = '```'
    index_code = code.rfind(end)
    if index_code != -1:
        output = code[1:index_code]
        code = output[output.rfind(start) + len(start) + 1:]
    logging.info(code)
    return code


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
