import os
import json
import torch
import transformers
from evaluate import load
from transformers import (
    HfArgumentParser,
    HfArgumentParser,
)
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import ScriptArguments, GenerationConfig
from model import get_model
from dataset import DatasetWrapper

class Pipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = GenerationConfig()

    def __call__(self, ds_wrapper, ds_loader):
        if self.task == "question-answering":
            return self.__question_answering(ds_wrapper, ds_loader)
        elif self.task == "summarization":
            return self.__summarization(ds_wrapper, ds_loader)
        elif self.task == "translation":
            return self.__translation(ds_wrapper, ds_loader)
        elif self.task == "text-generation":
            return self.__text_generation(ds_wrapper, ds_loader)
        else:
            raise NotImplementedError

    def __question_answering(self, ds_wrapper, ds_loader):
        pipeline = transformers.pipeline(
            model=self.model, 
            tokenizer=tokenizer,
            return_full_text=False,
            task='text-generation',
            **self.generation_config.question_answering
        )
        
        predictions = []
        references = []
        for batch in tqdm(ds_loader):
            prompts = [ds_wrapper.prompt.format(
                c, q
            ) for c, q in zip(batch[ds_wrapper.context], batch[ds_wrapper.question])]
            
            results = pipeline(prompts)
            predictions.extend([x[0]['generated_text'] for x in results])
            references.extend([x[0] for x in batch[ds_wrapper.answer]['text']])
        
        generations = {
            'predictions': predictions,
            'references': references
        }
        
        num_correct = 0
        for pred, ref in zip(predictions, references):
            if ref.lower() in pred.lower():
                num_correct += 1
                
        results = {
            'num_correct': num_correct,
            'num_total': len(predictions),
            'accuracy': num_correct / len(predictions)
        }
        
        # metric = load("exact_match")
        # results = metric.compute(predictions=predictions, 
        #                             references=references,
        #                             ignore_case =True,
        #                             ignore_punctuation=True
        # )
        
        return generations, results
    
    def __summarization(self, ds_wrapper, ds_loader):
        pipeline = transformers.pipeline(
            model=self.model, 
            tokenizer=tokenizer,
            return_full_text=False,
            task='text-generation',
            **self.generation_config.question_answering
        )
        
        predictions = []
        references = []
        for batch in tqdm(ds_loader):
            prompts = [ds_wrapper.prompt.format(
                text
            ) for text in batch[ds_wrapper.original_text]]
            
            results = pipeline(prompts)
            predictions.extend([x[0]['generated_text'].split('\n\n')[0] for x in results])
            references.extend([x for x in batch[ds_wrapper.summarized_text]])
        
        generations = {
            'predictions': predictions,
            'references': references
        }

        metrics = load('rouge')
        results = metrics.compute(predictions=predictions, 
                                  references=references
        )
        return generations, results
    
    def __translation(self, ds_wrapper, ds_loader):
        pass
    
    def __text_generation(self, ds_wrapper, ds_loader):
        pass
    
    def run(self, ds_wrapper, ds_loader):
        results = self(ds_wrapper, ds_loader)
        return results

def save_to_json(data, name):
    jsonString = json.dumps(data, indent=4)
    jsonFile = open(name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
def save_to_csv(data, name):
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load dataset (you can process it here)
    dataset_wrapper = DatasetWrapper(dataset_name=script_args.dataset_name)
    dataset_loader = DataLoader(dataset_wrapper.get_dataset(), 
                                batch_size=script_args.per_device_eval_batch_size, 
                                shuffle=False)
    
    # Load model
    model, tokenizer = get_model(config=script_args)
    model.eval()

    eval_pipeline = Pipeline(
        task=dataset_wrapper.task, 
        model=model, 
        tokenizer=tokenizer)
    
    # Evaluate
    generations, results = eval_pipeline.run(ds_wrapper=dataset_wrapper, ds_loader=dataset_loader)
    
    # Save results
    if not os.path.exists(script_args.output_dir):
        os.makedirs(script_args.output_dir)
        
    ds_exact_name = script_args.dataset_name.split('/')[-1]
    save_to_json(results, os.path.join(script_args.output_dir, f'results_{ds_exact_name}.json'))
    save_to_csv(generations, os.path.join(script_args.output_dir, f'results_{ds_exact_name}.csv'))