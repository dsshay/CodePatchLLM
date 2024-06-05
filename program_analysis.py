import logging
import subprocess
import os
from utils import read_file_to_string

SVACE_PATH = "~/GNN_in_program_analysis/svace-4.0.0-x64-linux/bin/svace"


def svace_analyze(file, lang, epoch, dir):
    logging.info(f"File Name: {file}, Directory: {dir}, Epoch: {epoch}")
    compiler_comand = ""
    result=""
    try:
        test = subprocess.run(f"cd {dir}; {SVACE_PATH} init", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
        compiler_comand = f"cd {dir}; {SVACE_PATH} build javac {file}"
    elif lang == "python":
        compiler_comand = f"cd {dir}; {SVACE_PATH} build --python {file}"
    elif lang == "kotlin":
        compiler_comand = f"cd {dir}; {SVACE_PATH} build kotlinc {file}"
    else:
        Exception("Undefined lanuage of programming. Use only java, python, kotlin. Sensetive to capitalization")


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
