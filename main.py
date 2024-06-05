import pandas as pd
import json
from pathlib import Path
from pandas import json_normalize
import logging

import xml.etree.ElementTree as ET
import time
import argparse
import wandb

import subprocess

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead # For RL fine-tuning (based on Leetcode submissions statuses)
import transformers

import os
import leetcode
import json
import leetcode.auth
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from prepare_data import load_data, preprocess_df
from utils import t_or_f
import nlp4code

def svace_analyze(file, lang, epoch, dir):
    logging.info(f"File Name:{file}, Lang: {lang}, Directory: {dir}, Epoch: {epoch}")
    compiler_comand = ""
    result=""
    try:
        test = subprocess.run(f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace init", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True)
        logging.info(f"What happend? {test.stdout}")
        test = subprocess.run(f"pwd", shell=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        logging.info(f"Current Directory: {test.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error message: {e.stderr}")
        exit(1)

    if lang == "java":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build javac {file}"
    elif lang == "python":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build --python {file}"
    elif lang == "go":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build go build {file}"
    elif lang == "kotlin":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build kotlinc {file}"
    else:
        Exception("Undefined lanuage of programming. Use only java, python, go, kotlin. Sensetive to capitalization")


    try:
        test = subprocess.run(compiler_comand, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        logging.info(f"Svace build out: {test.stdout} for file: {file}")
    except subprocess.CalledProcessError as e:
        logging.info(f"svace build: {test.stdout}")
        logging.error(f"Error executing command: {compiler_comand}")
        logging.error(f"Error message: {e.stderr}")
        if len(e.stderr)==0:
            result = "Write the full code with the correction."
        else:
            result = e.stderr
            result = result[:result.find("svace build: error:") + len("svace build: error:")]

    if len(result)==0:
        try:
            test = subprocess.run(f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace analyze", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True)
            logging.info(f"What happend? {test.stdout}")
            directory = dir + ".svace-dir/analyze-res"
            files = os.listdir(directory)
            svres_files = [file for file in files if file.endswith(".svres")]
            txt = [file for file in files if file.endswith(f"{epoch}.txt")]
            if len(txt) != 0:
                svace_an = read_file_to_string(directory+f"/{txt[0]}")
                lines = svace_an.strip().split("\n")
                try:
                    total_warnings = int(lines[0].split(":")[1].strip())
                    logging.info(f"Total warning={total_warnings} in epoch:{epoch}, question_id:{question_id}")
                    return 0
                except IndexError:
                    tree = ET.parse(directory + f"/{svres_files[0]}")
                    root = tree.getroot()
                    result = ""
                    for warn_info in root.findall(".//WarnInfo"):
                        line = warn_info.attrib.get("line")
                        warning_msg = warn_info.attrib.get("msg")
                        if warning_msg:
                            result += f"In Line {line}: {warning_msg}\n"
                # analyzed_lines = int(lines[2].split(":")[1].strip())
                # warning_density = float(lines[3].split(":")[1].strip())
                # logging.info(
                #     f"Total warning: {total_warnings}, Analyzed lines: {analyzed_lines}, Warning Density: {warning_density}")

            else:
                logging.error("Not Found analyze result file.txt")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing command: svace analyze")
            logging.error(f"Error message: {e.stderr}")
            exit(1)

    output_file_path = os.path.join(dir, f"svace_message.txt")
    with open(output_file_path, "w") as f:
        f.write(result)
    logging.info(f"Finished Svace analyzing, result saved in {output_file_path}")
    return 1


def leetcode_reward_function(formatted_responce):
    if formatted_responce['status_msg'] == 'Accepted':
        return 1
    elif formatted_responce['status_msg'] == 'Runtime Error':
        return -0.6
    elif formatted_responce['status_msg'] == 'Wrong Answer':
        return -0.3
    elif formatted_responce['status_msg'] == 'Compile Error':
        return -1
    elif formatted_responce['status_msg'] == 'Time Limit Exceeded':
        return -0.3


# def fit_model_with_reward(task, lang, output, reward):
#     inputs = get_input_from_task(task, lang).squeeze(0).to("cuda")
#     output = torch.tensor(tokenizer.encode(output), dtype=torch.long).to("cuda")
#     reward = torch.tensor([reward], dtype=torch.float).to("cuda")
#
#     ppo_trainer.step([inputs], [output], [reward])


# Function to submitting code to leetcode.com
def submit_to_leetcode(code,question_id, name_problem, api_instance, lang, dir, epoch):
    logging.info(f"Parametrs of submission: Question_id = {question_id}, name_problem = {name_problem}, lang = {lang}, epoch = {epoch}")
    if lang != "java" or lang != "python" or lang != "go" or lang != "kotlin":
        Exception("Undefined lanuage of programming. Use only java, python, go, kotlin. Sensetive to capitalization")

    submission = leetcode.Submission(
        judge_type="large", typed_code=code, question_id=question_id, test_mode=False, lang=lang
    )

    try: 
        submission_id = api_instance.problems_problem_submit_post(
            problem=name_problem, body=submission
        )
    except Exception as e:
        logging.error(f"Exception occurred: {e}. Leetcode submit failed. Maybe the problem is premium?")
        return None

    logging.info(f"The solution to the problem {name_problem} ({question_id}) submitted, the submission_id={submission_id}")

    time.sleep(20)

    submission_result = api_instance.submissions_detail_id_check_get(
        id=submission_id.submission_id
    )

    json_formatted_str = json.dumps(submission_result, indent=2)

    output_file_path = os.path.join(dir, f"result.json")
    with open(output_file_path, "w") as f:
        f.write(json_formatted_str)
    logging.info(
        f"Submission result saved in {output_file_path}")
    return submission_result





# def construct_leetcode_config(csrf_token, leetcode_session):
#
#     #csrf_token = leetcode.auth.get_csrf_cookie(leetcode_session)
#     configuration = leetcode.Configuration()
#
#     configuration.api_key["x-csrftoken"] = csrf_token
#     configuration.api_key["csrftoken"] = csrf_token
#     configuration.api_key["LEETCODE_SESSION"] = leetcode_session
#     configuration.api_key["Referer"] = "https://leetcode.com"
#     configuration.debug = False
#
#     api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))
#
#     graphql_request = leetcode.GraphqlQuery(
#         query="""
#         {
#           user {
#             username
#             isCurrentUserPremium
#           }
#         }
#             """,
#         variables=leetcode.GraphqlQueryVariables(),
#     )
#
#     print(api_instance.graphql_post(body=graphql_request))
#     return api_instance


if __name__ == "__main__":

    logging.basicConfig(
        # filename='HISTORYlistener.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().setLevel(logging.INFO)

    # Get the next two values from your browser cookies
    parser = argparse.ArgumentParser()
    # System arguments
    parser.add_argument("--dataset", default='mbpp', type=str) # leetcode (все языки), mbpp (python), humaneval (python)
    parser.add_argument("--mode", default='test', type=str) # use Codelama-7b-Instruct
    parser.add_argument("--wandb_flag", default=False, type=str)
    parser.add_argument("--lang", default="java", type=str) # kotlin, python, java
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Number of samples to solve and evaluate from the benchmark")

    args = parser.parse_args()
    args.wandb_flag = t_or_f(args.wandb_flag)

    print(f"Arguments: {args}")

    if args.wandb_flag:
        wandb.init(project='gnn_in_pa', entity="gnn_in_pa", config=args, tags=["test"])

    model_id = ""
    if args.mode == 'test':
        model_id = "codellama/CodeLlama-7b-Instruct-hf"
    elif args.mode == 'prod':
        model_id = "codellama/CodeLlama-70b-Instruct-hf"
    else:
        print(f'The mode can be of two types: test or prod')
        Exception()

    df = load_data(args.dataset)
    # train_test_validation = preprocess_df(df)

    formats = None
    if args.lang == "python":
        formats = "py"
    elif args.lang == "kotlin":
        formats = "kt"
    elif args.lang == "java":
        formats = "java"
    else:
        Exception("Error in using language")

    main_directory = f"./llm_predicts/{formats}"
    os.makedirs(f"./llm_predicts/{formats}", exist_ok=True)

    n_tasks = min(args.limit, len(df)) if args.limit else len(df)

    for i in range(len(df)):
        question_id = df[i]['task_id']
        task = df[i]['content']
        logging.info(f"Question id = {question_id}")

        output_directory = main_directory + f"/{question_id}/"
        os.makedirs(output_directory, exist_ok=True)
        for epoch in range(args.num_epochs):
            output_directory = main_directory + f"/{question_id}/{epoch}/"
            os.makedirs(output_directory, exist_ok=True)
            output = nlp4code.process_task(task, model_id=model_id, lang=args.lang, dir=output_directory, epoch=epoch)
            exit(1)
            logging.info("Start selecting output")
            start = '```'
            end = '```'
            index_code = output.rfind(end)
            if index_code != -1:
                output = output[1:index_code]
                output = output[output.rfind(start) + len(start) + 1:]
            else:
                logging.error("It is not possible to correctly select the code from the model's response. Perhaps there is not enough response length?")
                break
            # logging.info(f"Selected output: {output}")
            # if args.lang == "java":
            #     output = "class Solution{\n" + output + "}"
            output_file_path = os.path.join(output_directory, f"Solution.{formats}")
            with open(output_file_path, "w") as f:
                f.write(output)
            logging.info(f"Finished selecting output, saved in {output_file_path}")
            formatted_responce_leetcode = submit_to_leetcode(code=output, question_id=question_id, name_problem=name_problem, api_instance=api_instance, epoch=epoch, lang=args.lang, dir=output_directory)
            if formatted_responce_leetcode == None:
                break

            # reward = leetcode_reward_function(formatted_responce_leetcode)
            # leetcode_feedback = formatted_responce_leetcode['full_runtime_error'] == True
            # logging.info(f"Reward for current run is: {str(reward)}")

            svace_flag = svace_analyze(file=f"Solution.{formats}", lang=args.lang, dir=output_directory, epoch=epoch)
            if svace_flag == 0:
                break
            logging.info("Svace analyzed somthing")




