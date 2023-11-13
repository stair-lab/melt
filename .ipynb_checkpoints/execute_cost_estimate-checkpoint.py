import os

import shlex
from subprocess import Popen, PIPE
from tabulate import tabulate

def execute(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)

    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    #
    return exitcode, out.decode(), err.decode()
all_tasks = {
    "xquad_xtreme_prompt0": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name xquad_xtreme  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "xquad_xtreme_prompt1": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name xquad_xtreme  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "xquad_xtreme_prompt2": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name xquad_xtreme  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "xquad_xtreme_prompt2_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name xquad_xtreme_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "xquad_xtreme_prompt2_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name xquad_xtreme_fairness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "mlqa_prompt0": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mlqa  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "mlqa_prompt1": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mlqa  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "mlqa_prompt2": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mlqa  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "mlqa_prompt2_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mlqa_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "mlqa_prompt2_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mlqa_fairness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    
    "vietnews_prompt0": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vietnews  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "vietnews_prompt1": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vietnews  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "vietnews_prompt2": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vietnews  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "vietnews_prompt2_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vietnews_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "wiki_lingua_prompt0": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name wiki_lingua  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "wiki_lingua_prompt1": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name wiki_lingua  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "wiki_lingua_prompt2": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name wiki_lingua  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "wiki_lingua_prompt2_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name wiki_lingua_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    
    "UIT-VSFC": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSFC  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "UIT-VSFC_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSFC  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "UIT-VSFC_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSFC_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "UIT-VSFC_fewshot_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSFC_fairness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "vlsp2016": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vlsp2016  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "vlsp2016_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vlsp2016  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "vlsp2016_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vlsp2016_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "vlsp2016_fewshot_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name vlsp2016_fairness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "UIT-VSMEC": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSMEC  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "UIT-VSMEC_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSMEC  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "UIT-VSMEC_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSMEC_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "UIT-VSMEC_fewshot_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name UIT-VSMEC_fairness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "PhoATIS": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoATIS  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "PhoATIS_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoATIS  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "PhoATIS_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoATIS_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "PhoATIS_fewshot_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoATIS_fairness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "ViCTSD": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViCTSD  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "ViCTSD_fewshot":  "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViCTSD  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "ViCTSD_fewshot_robustness":  "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViCTSD_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "ViCTSD_fewshot_fairness":  "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViCTSD_fairness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "ViHSD": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViHSD  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "ViHSD_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViHSD  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "ViHSD_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViHSD_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "ViHSD_fewshot_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViHSD_fairness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "zalo_e2eqa": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name zalo_e2eqa  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "zalo_e2eqa_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name zalo_e2eqa  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "zalo_e2eqa_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name zalo_e2eqa_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "ViMMRC": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViMMRC  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "ViMMRC_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViMMRC  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "ViMMRC_fewshot_rnd": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViMMRC  --prompting_strategy 0  --fewshot_prompting True --random_mtpc --seed 42",
    "ViMMRC_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name ViMMRC_robustness  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "mlqa_MLM": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mlqa_MLM  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "mlqa_MLM_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mlqa_MLM  --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "VSEC": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name VSEC --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "VSEC_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name VSEC --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "mmarco": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mmarco --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "mmarco_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mmarco --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "mmarco_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mmarco_robustness --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "mmarco_fewshot_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mmarco_fairness --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "mrobust": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mrobust --prompting_strategy 0  --fewshot_prompting False --seed 42",
    "mrobust_fewshot": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mrobust --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "mrobust_fewshot_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mrobust_robustness --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "mrobust_fewshot_fairness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name mrobust_fairness --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "synthetic_natural_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_natural_gcp --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "synthetic_induction_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_induction_gcp --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "synthetic_pattern_match_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_pattern_match_gcp --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "synthetic_variable_substitution_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_variable_substitution_gcp --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "synthetic_natural_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_natural_azr --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "synthetic_induction_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_induction_azr --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "synthetic_pattern_match_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_pattern_match_azr --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "synthetic_variable_substitution_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name synthetic_variable_substitution_azr --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "math_level1_Algebra_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Algebra\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Counting & Probability_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Counting & Probability\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Geometry_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Geometry\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Intermediate Algebra_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Intermediate Algebra\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Number Theory_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Number Theory\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Prealgebra_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Prealgebra\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Precalculus_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Precalculus\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Algebra_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Algebra\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Counting & Probability_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Counting & Probability\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Geometry_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Geometry\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Intermediate Algebra_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Intermediate Algebra\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Number Theory_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Number Theory\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Prealgebra_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Prealgebra\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "math_level1_Precalculus_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Precalculus\" --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
"math_level1_Algebra_cot_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Algebra\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Counting & Probability_cot_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Counting & Probability\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Geometry_cot_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Geometry\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Intermediate Algebra_cot_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Intermediate Algebra\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Number Theory_cot_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Number Theory\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Prealgebra_cot_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Prealgebra\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Precalculus_cot_GCP": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_gcp_Precalculus\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Algebra_cot_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Algebra\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Counting & Probability_cot_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Counting & Probability\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Geometry_cot_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Geometry\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Intermediate Algebra_cot_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Intermediate Algebra\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Number Theory_cot_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Number Theory\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Prealgebra_cot_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Prealgebra\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
    "math_level1_Precalculus_cot_AZR": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name \"math_level1_azr_Precalculus\" --prompting_strategy 0  --fewshot_prompting True --cot --seed 42",
 
    
    "PhoMT_envi": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoMT_envi --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "PhoMT_envi_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoMT_envi_robustness --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "PhoMT_vien": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoMT_vien --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "PhoMT_vien_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name PhoMT_vien_robustness --prompting_strategy 0  --fewshot_prompting True --seed 42",
    
    "opus100_envi": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name opus100_envi --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "opus100_envi_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name opus100_envi_robustness --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "opus100_vien": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name opus100_vien --prompting_strategy 0  --fewshot_prompting True --seed 42",
    "opus100_vien_robustness": "python estimate_cost.py --model_name gpt-3.5-turbo --dataset_name opus100_vien_robustness --prompting_strategy 0  --fewshot_prompting True --seed 42"
    
}


def main():
    
    total = {}
    total_costs = 0
    rows = []
    head = ["Task", "Num Tokens", "Cost"]
    for task, cmd in all_tasks.items():
        try:
            lst = execute(cmd)[1].split("\n")[-2].split("  ")
            tokens = int(lst[0])
            cost = lst[1]
            total[task] = {}
            total[task]["num_tokens"] = tokens
            total[task]["cost"] = cost
            rows.append([task, tokens, cost])
            total_costs += float(cost)
            print("Current cost: {}".format(total_costs))
        except Exception as e:
            print("Error at task {}".format(task))
            print(str(e))
            break
    print(tabulate(rows, headers=head, tablefmt="grid"))
    print("Total cost: {}".format(total_costs))
    import json
 
    # Serializing json
    json_object = json.dumps(total, indent=4)

    # Writing to sample.json
    with open("table_cost.txt", "w") as f:
        f.write(str(tabulate(rows, headers=head, tablefmt="grid"))+"\nTotal cost: ${}".format(total_costs))
                
    with open("cost_data.json", "w") as outfile:
        outfile.write(json_object)
 

main()