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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead # For RL fine-tuning (based on Leetcode submissions statuses)
import transformers
import torch
import os
import leetcode
import json
import leetcode.auth
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

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

def get_input_from_task(task: str, lang: str, dir, epoch):
    logging.info(f"Epoch: {epoch}, Lang:{lang}, Dir:{dir}")
    chat = [
        {"role": "system", "content": f"You are a helpful and honest code assistant expert in {lang.capitalize()}. Please, provide all answers to programming questions in {lang.capitalize()}."},
        {"role": "user", "content": "Write method in class Solution. "+task},
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
            chat.append({"role": "user", "content": "correct program above with this feedback: "+read_file_to_string(file_path_svace_output) + ".\n Write the resulting code."})
    logging.info(chat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    return inputs


# Function to generate code for task
def process_task(task, lang, dir, epoch):
    # generation_kwargs = {
    #     "min_length": -1,
    #     "top_k": 0.0,
    #     "top_p": 1.0,
    #     "do_sample": True,
    #     "pad_token_id": tokenizer.eos_token_id,
    #     "max_new_tokens": 500,
    # }

    # Send task to the language model
    if lang != "java" or lang != "python" or lang != "go" or lang != "kotlin":
        Exception("Undefined lanuage of programming. Use only java, python, go, kotlin")

    question_dir = dir
    if epoch > 0:
        question_dir = os.path.dirname(os.path.dirname(dir))
        logging.info(f"Path question directory: {question_dir}")

    inputs = get_input_from_task(task, lang=lang, dir=question_dir, epoch=epoch)
    # output = ppo_trainer.generate(inputs, **generation_kwargs)
    logging.info("Start generate code")
    output = model.generate(input_ids=inputs, max_new_tokens=2048)
    output = tokenizer.decode(output[0].to("cpu"), skip_special_tokens=True)
    output_file_path = os.path.join(dir, f"model_output.txt")
    with open(output_file_path, "w") as f:
        f.write(output)
    logging.info(f"Finished generate code, saved in {output_file_path}")
    return output


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


def load_data():
    if not os.path.exists('./data/leetcode-train.jsonl'):
        hf_hub_download(repo_id="greengerong/leetcode", filename="leetcode-train.jsonl", local_dir='./data/', local_dir_use_symlinks=False, repo_type="dataset")
    logging.info("Start download file")
    df = pd.read_json("./data/leetcode-train.jsonl", lines=True).drop(['python', 'javascript', 'java', 'c++'], axis=1)
    # df = df.loc[df['id'] == 12]

    # df.difficulty.fillna(value="Medium", inplace=True)
    # train, test = train_test_split(df, stratify=df['difficulty'], random_state=42, test_size=0.2)
    # train, val = train_test_split(train, stratify=train['difficulty'], random_state=42, test_size=0.1)
    # logging.info(f"Train size f{train.shape}, Validate size {val.shape}, Test size {test.shape}")
    logging.info(f"Finished download file, dataframe size: {df.size}")
    # return train, val, test
    return df


def construct_leetcode_config(csrf_token, leetcode_session):

    #csrf_token = leetcode.auth.get_csrf_cookie(leetcode_session)
    configuration = leetcode.Configuration()

    configuration.api_key["x-csrftoken"] = csrf_token
    configuration.api_key["csrftoken"] = csrf_token
    configuration.api_key["LEETCODE_SESSION"] = leetcode_session
    configuration.api_key["Referer"] = "https://leetcode.com"
    configuration.debug = False

    api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))

    graphql_request = leetcode.GraphqlQuery(
        query="""
        {
          user {
            username
            isCurrentUserPremium
          }
        }
            """,
        variables=leetcode.GraphqlQueryVariables(),
    )

    print(api_instance.graphql_post(body=graphql_request))
    return api_instance


if __name__ == "__main__":
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.basicConfig(
        filename='HISTORYlistener.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().setLevel(logging.INFO)

    # Get the next two values from your browser cookies
    leetcode_session = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTI1NzUwOTkiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJkamFuZ28uY29udHJpYi5hdXRoLmJhY2tlbmRzLk1vZGVsQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImEwYTlmYzhkY2UwNjljNDc1YTg2YzM2ZjdkM2E3ZmIzMDY4NWE0ZmVmMjNlODE5MWIyMWNkODk0ODg2M2Y3ODAiLCJpZCI6MTI1NzUwOTksImVtYWlsIjoic2hheWhlbGlzbGFtb3YuZHNAcGh5c3RlY2guZWR1IiwidXNlcm5hbWUiOiJkc3NoYXkiLCJ1c2VyX3NsdWciOiJkc3NoYXkiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvZGVmYXVsdF9hdmF0YXIuanBnIiwicmVmcmVzaGVkX2F0IjoxNzEwNjYxMzk4LCJpcCI6Ijg5LjI0OC4xOTEuNzYiLCJpZGVudGl0eSI6ImRkNzg4NzhiZWJjMGU2YWZmZjgwYmU5NjUxNjUxMWQ3Iiwic2Vzc2lvbl9pZCI6NTc3NTk2NDYsIl9zZXNzaW9uX2V4cGlyeSI6MTIwOTYwMH0.u706Kl18DknsUzZLd4bEihcNyhGRXPfqFBs-qNJHOss"
    csrf_token = "QHHT4u1TEM6E471214sWR1u24o9HcD9xCnlhHFvd8STo4vXomvRfVBKhLczDZqgN"
    parser = argparse.ArgumentParser()
    # System arguments
    parser.add_argument("--dataset", default='leetcode-train.jsonl', type=str)
    parser.add_argument("--mode", default='test', type=str)
    parser.add_argument("--wandb_flag", default=False, type=str)
    parser.add_argument("--lang", default="java", type=str)
    parser.add_argument("--num_epochs", default=5, type=int)

    args = parser.parse_args()
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

    df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # .to(device)
    print("HERE")
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # ).bfloat16()
    
    # Tuning model with RLAIF (use leetcode statuses as feedback)
    # TODO: Optimize training (now it is either skipping batches or working too slowly)
    # config = PPOConfig(
    #     model_name="model_id",
    #     learning_rate=1.41e-5,
    #     batch_size=1,
    #     mini_batch_size=1,
    #     gradient_accumulation_steps=1,
    # )
    # ppo_trainer = PPOTrainer(
    #     model=model,
    #     config=config,
    #     tokenizer=tokenizer,
    # )

    formats = args.lang
    if args.lang == "python":
        formats = "py"
    elif args.lang == "kotlin":
        formats = "kt"

    main_directory = f"./llm_predicts/{formats}"
    os.makedirs(f"./llm_predicts/{formats}", exist_ok=True)


    api_instance = construct_leetcode_config(csrf_token,leetcode_session)

    for _, row in df.iterrows():
        name_problem = row['slug']
        question_id = row['id']
        task = row['content']

        analyzer_feedback = None
        output_directory = main_directory + f"/{question_id}/"
        os.makedirs(output_directory, exist_ok=True)
        for epoch in range(args.num_epochs):
            output_directory = main_directory + f"/{question_id}/{epoch}/"
            os.makedirs(output_directory, exist_ok=True)
            output = process_task(task, lang=args.lang, dir=output_directory, epoch=epoch)
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




