"""
This Module creates tables for latex super fast.

"""

import json
def print_table(data, first_line, metric):
    print(r"\begin{center}")
    print(r"\begin{tabular}{ c| c c c c c}")
    print(first_line)
    print("\\hline")
    for k in data.keys():
        if k == "eval_params":
            continue
        eval = data[k]['evaluation']
        if k == "ORB":
            string = k + ' & '
        else:
            string = k[:-5].replace('_', '') + ' & '
        for i in eval[:-1]:
            string = string + "{:.3f}".format(i['result'][metric]) + " & "
        i = eval[-1]
        string = string + "{:.3f}".format(i['result'][metric]) + " \\\\"
        print(string)
    print("\\hline")
    print("ORB & - & - & - & - & \\\\")
    print(r"\end{tabular}")
    print(r"\end{center}")

def create_first_line(data):
    k = [x["name"] for x in data["eval_params"]]
    first_line = "Model/Config & " + " & ".join(k) + " \\\\"
    return first_line.replace('_', '')

def read_json(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data


path = r'./data/eval/_15_08_2022__11_37_29_eval_result.json'
data = read_json(path)
rKeys = ["Repeatability","Correctness d5", "Amount of good points", "MScore"]
first_line = create_first_line(data)


for rk in rKeys:
    print(rk)
    print_table(data, first_line, rk)
