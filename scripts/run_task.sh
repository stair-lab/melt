#!/bin/bash
set -e
while getopts "n:u:t:" flag
    do
            case "${flag}" in
                    "n") MODEL_ID=${OPTARG};;
                    "u") TGI=${OPTARG};;
                    "t") TASK=${OPTARG};;
            esac
    
    done

echo $MODEL_ID
echo $TGI
echo $TASK

if [ "$TASK" = "short1" ]; then
    echo "Knowledge task";
    echo "Start ....";
     ./new_script/run_knowledge.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_knowledge_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";

    echo "Sentiment task";
    echo "Start ....";
     ./new_script/run_sentiment.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_sentiment_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";

    echo "Toxicity task";
    echo "Start ....";
     ./new_script/run_toxicity.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_toxicity_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";

    echo "Text Classification task";
    echo "Start ....";
     ./new_script/run_tclassify.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_text_classification_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";

    echo "QA task";
    echo "Start ....";
     ./new_script/run_qa.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_qa_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";

elif [ "$TASK" = "short2" ]; then    
    echo "Reasoning Task";
    echo "Start ....";
     ./new_script/run_reasoning.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_reasoning_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";
    
    echo "Math Task";
    echo "Start ....";
     ./new_script/run_math.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_math_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";

    echo "Zero-shot Reasoning and Math";
    echo "Start ....";
     ./new_script/run_zero_math_reasoning.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_zeroshot_math_reasoning_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";

    echo "Translation";
    echo "Start ....";
     ./new_script/run_translation.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_translation_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";
    
elif [ "$TASK" = "sum" ]; then
    echo "Summarization task";
    echo "Start ....";
     ./new_script/run_sum.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_sum_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";
    
elif [ "$TASK" = "lm" ]; then
    echo "Language model task";
    echo "Start ....";
     ./new_script/run_lm.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_lm_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";
    
elif [ "$TASK" = "ir" ]; then
    echo "Information retrieval task";
    echo "Start ....";
     ./new_script/run_ir.sh $MODEL_ID $TGI 2>&1 | tee -a logs/log_ir_"$MODEL_ID".txt > /dev/null ;
    echo "Done ....";
else
    echo "ERROR, $TASK is not exist"
fi