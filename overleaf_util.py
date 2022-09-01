"""
This Module creates tables for latex super fast.

"""
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
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


# path = r'./data/eval/_15_08_2022__11_37_29_eval_result.json'
# data = read_json(path)
# rKeys = ["Repeatability","Correctness d5", "Amount of good points", "MScore"]
# first_line = create_first_line(data)

#
# for rk in rKeys:
#     print(rk)
#     print_table(data, first_line, rk)




def create_plot_for_metric(metric, val):
    path_all = r"C:\Users\Dr. Paul von Immel\Documents\MyShit\2022\Report\Appendix\NetworkEval\All_Extra_run.json"
    path_orb = r"C:\Users\Dr. Paul von Immel\Documents\MyShit\2022\Report\Appendix\NetworkEval\ORB_512.json"

    data = read_json(path_all)
    data_container = []
    names = []

    for k in data.keys():
        if k == "eval_params":
            names = [i['name'].replace('_','').replace('config','') for i in data[k]]
            names.insert(0, "")
            continue
        eval = data[k]['evaluation']
        d = [i['result'][metric] for i in eval]
        d.insert(0,k[:-5].replace('_',''))
        d.insert(1, d[-1])
        d = d[:-1]
        data_container.append(d)

    data_orb = read_json(path_orb)
    eval = data_orb["ORB"]['evaluation']
    d = [i['result'][metric] for i in eval]
    d.insert(0, "ORB")
    d.insert(1, d[-1])
    d = d[:-1]
    d.append(val)
    data_container.append(d)
    # Define Data


    # Plot multiple columns bar chart
    names.insert(1,names[-1])
    names = names[:-1]
    df=pd.DataFrame(data_container,columns=names)
    df.plot(x="", kind="bar",y = names[1:],figsize=(9,7),colormap='copper', title=metric, fontsize = 16 )


    plt.legend(fontsize=12)
    plt.autoscale()
    if metric == "Repeatability":
        plt.legend(loc=3)
    plt.savefig(metric + ".png")
    plt.show()


metrics = ["Repeatability","Correctness d5", "Amount of good points", "MScore"]
missing_vals = [0.508,0.036,212.1,0.127]

for i,m in enumerate(metrics):
    create_plot_for_metric(m, missing_vals[i])

