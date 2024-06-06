import logging

import argparse
import wandb
import os
from prepare_data import load_data
from utils import t_or_f, _parse_instruction
import nlp4code
from program_analysis import svace_analyze

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
    parser.add_argument("--start", default=0, type=int, help="Start Number of dataset")

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

    for i in range(args.start, args.start + n_tasks):
        question_id = df[i]['task_id']
        task = df[i]['content']
        logging.info(f"Question id = {question_id}")

        output_directory = main_directory + f"/{question_id}/"
        os.makedirs(output_directory, exist_ok=True)
        for epoch in range(args.num_epochs):
            output_directory = main_directory + f"/{question_id}/{epoch}/"
            os.makedirs(output_directory, exist_ok=True)
            code = nlp4code.process_task(task, model_id=model_id, lang=args.lang, dir=output_directory, epoch=epoch)
            output = _parse_instruction(code, args.lang)
            output_file_path = os.path.join(output_directory, f"Solution.{formats}")
            with open(output_file_path, "w") as f:
                f.write(output)
            logging.info(f"Finished parsing code, saved in {output_file_path}")

            svace_flag = svace_analyze(file=f"Solution.{formats}", lang=args.lang, dir=output_directory, epoch=epoch)
            if svace_flag == 0:
                break
            logging.info("Svace analyzed somthing")




