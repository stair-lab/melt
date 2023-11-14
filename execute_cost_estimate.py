import os

import shlex
from subprocess import Popen, PIPE
from tabulate import tabulate

MODEL_NAME="gpt-4"

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
    # "xquad_xtreme_prompt0": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    # "xquad_xtreme_prompt1": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    # "xquad_xtreme_prompt2": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    # "xquad_xtreme_prompt2_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    # "xquad_xtreme_prompt2_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme_fairness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
     "xquad_xtreme_prompt3": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    "xquad_xtreme_prompt3_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme_robustness  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    "xquad_xtreme_prompt3_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name xquad_xtreme_fairness  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    # "mlqa_prompt0": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa  --prompting_strategy 0  --fewshot_prompting False --seed 42",
    # "mlqa_prompt1": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa  --prompting_strategy 1  --fewshot_prompting False --seed 42",
#     "mlqa_prompt2": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa  --prompting_strategy 2  --fewshot_prompting False --seed 42",
#     "mlqa_prompt2_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
#     "mlqa_prompt2_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa_fairness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "mlqa_prompt3": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    "mlqa_prompt3_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa_robustness  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    "mlqa_prompt3_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa_fairness  --prompting_strategy 3  --fewshot_prompting False --seed 42",
#     "vietnews_prompt0": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vietnews  --prompting_strategy 0  --fewshot_prompting False --seed 42",
#     "vietnews_prompt1": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vietnews  --prompting_strategy 1  --fewshot_prompting False --seed 42",
#     "vietnews_prompt2": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vietnews  --prompting_strategy 2  --fewshot_prompting False --seed 42",
#     "vietnews_prompt2_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vietnews_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "vietnews_prompt3": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vietnews  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    "vietnews_prompt3_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vietnews_robustness  --prompting_strategy 3  --fewshot_prompting False --seed 42",
#     "wiki_lingua_prompt0": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name wiki_lingua  --prompting_strategy 0  --fewshot_prompting False --seed 42",
#     "wiki_lingua_prompt1": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name wiki_lingua  --prompting_strategy 1  --fewshot_prompting False --seed 42",
#     "wiki_lingua_prompt2": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name wiki_lingua  --prompting_strategy 2  --fewshot_prompting False --seed 42",
#     "wiki_lingua_prompt2_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name wiki_lingua_robustness  --prompting_strategy 2  --fewshot_prompting False --seed 42",
    "wiki_lingua_prompt3": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name wiki_lingua  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    "wiki_lingua_prompt3_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name wiki_lingua_robustness  --prompting_strategy 3  --fewshot_prompting False --seed 42",
    
    "UIT-VSFC": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSFC  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "UIT-VSFC_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSFC  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "UIT-VSFC_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSFC_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "UIT-VSFC_fewshot_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSFC_fairness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "vlsp2016": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vlsp2016  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "vlsp2016_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vlsp2016  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "vlsp2016_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vlsp2016_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "vlsp2016_fewshot_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name vlsp2016_fairness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "UIT-VSMEC": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSMEC  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "UIT-VSMEC_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSMEC  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "UIT-VSMEC_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSMEC_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "UIT-VSMEC_fewshot_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name UIT-VSMEC_fairness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "PhoATIS": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoATIS  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "PhoATIS_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoATIS  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "PhoATIS_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoATIS_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "PhoATIS_fewshot_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoATIS_fairness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "ViCTSD": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViCTSD  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "ViCTSD_fewshot":  f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViCTSD  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "ViCTSD_fewshot_robustness":  f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViCTSD_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "ViCTSD_fewshot_fairness":  f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViCTSD_fairness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "ViHSD": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViHSD  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "ViHSD_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViHSD  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "ViHSD_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViHSD_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "ViHSD_fewshot_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViHSD_fairness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "zalo_e2eqa": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name zalo_e2eqa  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "zalo_e2eqa_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name zalo_e2eqa  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "zalo_e2eqa_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name zalo_e2eqa_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "ViMMRC": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViMMRC  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "ViMMRC_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViMMRC  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "ViMMRC_fewshot_rnd": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViMMRC  --prompting_strategy 1  --fewshot_prompting True --random_mtpc --seed 42",
    "ViMMRC_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name ViMMRC_robustness  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "mlqa_MLM": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa_MLM  --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "mlqa_MLM_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mlqa_MLM  --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "VSEC": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name VSEC --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "VSEC_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name VSEC --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "mmarco": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mmarco --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "mmarco_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mmarco --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "mmarco_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mmarco_robustness --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "mmarco_fewshot_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mmarco_fairness --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "mrobust": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mrobust --prompting_strategy 1  --fewshot_prompting False --seed 42",
    "mrobust_fewshot": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mrobust --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "mrobust_fewshot_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mrobust_robustness --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "mrobust_fewshot_fairness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name mrobust_fairness --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "synthetic_natural_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_natural_gcp --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "synthetic_induction_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_induction_gcp --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "synthetic_pattern_match_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_pattern_match_gcp --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "synthetic_variable_substitution_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_variable_substitution_gcp --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "synthetic_natural_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_natural_azr --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "synthetic_induction_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_induction_azr --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "synthetic_pattern_match_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_pattern_match_azr --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "synthetic_variable_substitution_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name synthetic_variable_substitution_azr --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "math_level1_Algebra_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Algebra\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Counting & Probability_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Counting & Probability\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Geometry_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Geometry\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Intermediate Algebra_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Intermediate Algebra\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Number Theory_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Number Theory\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Prealgebra_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Prealgebra\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Precalculus_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Precalculus\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Algebra_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Algebra\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Counting & Probability_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Counting & Probability\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Geometry_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Geometry\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Intermediate Algebra_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Intermediate Algebra\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Number Theory_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Number Theory\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Prealgebra_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Prealgebra\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "math_level1_Precalculus_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Precalculus\" --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
"math_level1_Algebra_cot_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Algebra\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Counting & Probability_cot_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Counting & Probability\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Geometry_cot_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Geometry\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Intermediate Algebra_cot_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Intermediate Algebra\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Number Theory_cot_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Number Theory\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Prealgebra_cot_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Prealgebra\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Precalculus_cot_GCP": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_gcp_Precalculus\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Algebra_cot_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Algebra\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Counting & Probability_cot_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Counting & Probability\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Geometry_cot_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Geometry\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Intermediate Algebra_cot_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Intermediate Algebra\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Number Theory_cot_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Number Theory\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Prealgebra_cot_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Prealgebra\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
    "math_level1_Precalculus_cot_AZR": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name \"math_level1_azr_Precalculus\" --prompting_strategy 1  --fewshot_prompting True --cot --seed 42",
 
    
    "PhoMT_envi": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoMT_envi --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "PhoMT_envi_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoMT_envi_robustness --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "PhoMT_vien": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoMT_vien --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "PhoMT_vien_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name PhoMT_vien_robustness --prompting_strategy 1  --fewshot_prompting True --seed 42",
    
    "opus100_envi": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name opus100_envi --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "opus100_envi_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name opus100_envi_robustness --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "opus100_vien": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name opus100_vien --prompting_strategy 1  --fewshot_prompting True --seed 42",
    "opus100_vien_robustness": f"python estimate_cost.py --model_name {MODEL_NAME} --dataset_name opus100_vien_robustness --prompting_strategy 1  --fewshot_prompting True --seed 42"
    
}


def main():
    
    total = {}
    total_costs = 0
    rows = []
    head = ["Task", "Total Tokens", "Prompt Tokens", "Completion Tokens", "Cost"]
    for task, cmd in all_tasks.items():
        try:
            
            lst = execute(cmd)[1]
            
            lst = lst.split("\n")[-2].split(" ")

            total_tokens = int(lst[0])
            prompt_tokens = int(lst[1])
            completion_tokens = int(lst[2])
            cost = lst[3]
            total[task] = {}
            total[task]["total_tokens"] = total_tokens
            total[task]["prompt_tokens"] = prompt_tokens
            total[task]["completion_tokens"] = completion_tokens
            total[task]["cost"] = cost

            rows.append([task, total_tokens, prompt_tokens, completion_tokens, cost])
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
    with open(f"table_cost_{MODEL_NAME}.txt", "w") as f:
        f.write(str(tabulate(rows, headers=head, tablefmt="grid"))+"\nTotal cost: ${}".format(total_costs))
                
    with open(f"cost_data_{MODEL_NAME}.json", "w") as outfile:
        outfile.write(json_object)
 

main()