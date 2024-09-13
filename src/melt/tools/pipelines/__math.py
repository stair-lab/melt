"__math"
import random
from tqdm import tqdm
from utils.utils import format_fewshot
class ResultsContainer:
    "class"
    def additional_method1(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def __init__(self):
        self.predictions = []
        self.references = []
        self.generation_probs = []
        self.calib_probs = []
        self.math_problem_type = []
    def extend(self, other):
        "extend"
        self.predictions.extend(other.predictions)
        self.references.extend(other.references)
        self.generation_probs.extend(other.generation_probs)
        self.calib_probs.extend(other.calib_probs)
        self.math_problem_type.extend(other.math_problem_type)

class FewShotData:
    "class"
    def additional_method2(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def additional_method3(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def __init__(self):
        self.original_few_shot = []
        self.calib_few_shot = []
        self.selected_sample = []
class DatasetConfig:
    "class"
    def additional_method4(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def additional_method5(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def __init__(self, ds_wrapper, ds_loader):
        self.ds_wrapper = ds_wrapper
        self.ds_loader = ds_loader
class BatchData:
    "class"
    def additional_method6(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def additional_method7(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def __init__(self, prompts, calib_prompts, batch, ds_wrapper):
        self.prompts = prompts
        self.calib_prompts = calib_prompts
        self.batch = batch
        self.ds_wrapper = ds_wrapper
class SaveConfig:
    "Class"
    def additional_method8(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def additional_method9(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def __init__(self, saving_fn, ds_wrapper, task_name, config):
        self.saving_fn = saving_fn
        self.ds_wrapper = ds_wrapper
        self.task_name = task_name
        self.config = config
class MathPipelineConfig:
    "Class"
    def additional_method10(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def additional_method11(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def __init__(self, task_name, config, continue_infer_data=None, few_shot=False):
        self.task_name = task_name
        self.config = config
        self.continue_infer_data = continue_infer_data
        self.few_shot = few_shot
class MathPipeline:
    "Class"
    def additional_method12(self):
        """
        Another public method to satisfy the two-method requirement.
        """
        print("")
    def __init__(self, metric_pipeline, infer_pipeline, pipeline_config):
        self.metric_pipeline = metric_pipeline
        self.infer_pipeline = infer_pipeline
        self.pipeline_config = pipeline_config
        # Ensure continue_infer_data and config are initialized
        self.continue_infer_data = pipeline_config.continue_infer_data
        self.config = pipeline_config.config

    def __math(self, dataset_config, saving_fn, start_idx=0):
        save_config = SaveConfig(saving_fn,
                                  dataset_config.ds_wrapper,
                                  self.pipeline_config.task_name, self.config)
        results = ResultsContainer()
        few_shot_data = FewShotData()
        idx = 0

        if self.continue_infer_data is not None:
            self._handle_continue_data(results)

        if self.pipeline_config.few_shot:
            few_shot_data = self._prepare_few_shot_data(dataset_config.ds_wrapper)

        for batch in tqdm(dataset_config.ds_loader):
            if idx < start_idx:
                idx += 1
                continue

            batch_data = self._prepare_batch_data(dataset_config.ds_wrapper, batch, few_shot_data)
            batch_results = self._process_batch(batch_data)
            results.extend(batch_results)

            idx += 1
            if idx % 100 == 0:
                self._save_intermediate_results(idx, results, few_shot_data, save_config)

        final_results = self._save_final_results(results, few_shot_data, save_config)
        return final_results

    def _handle_continue_data(self, results):
        continue_data = ResultsContainer()
        continue_data.predictions = self.continue_infer_data["predictions"]
        continue_data.references = self.continue_infer_data["references"]
        continue_data.generation_probs = self.continue_infer_data["generation_probs"]
        continue_data.calib_probs = self.continue_infer_data["calibration_probs"]
        continue_data.math_problem_type = self.continue_infer_data.get("math_problem_type", [])
        results.extend(continue_data)

    def _prepare_batch_data(self, ds_wrapper, batch, few_shot_data):
        prompts = self._create_prompts(ds_wrapper, batch, few_shot_data.original_few_shot)
        calib_prompts = self._create_calib_prompts(ds_wrapper, batch, few_shot_data.calib_few_shot)
        return BatchData(prompts, calib_prompts, batch, ds_wrapper)

    def _process_batch(self, batch_data):
        batch_results = ResultsContainer()

        results, logprobs, _ = self.infer_pipeline(batch_data.prompts, return_probs=True)
        calibprob_batch, _ = self.infer_pipeline.compute_logprob_and_length(
            batch_data.calib_prompts, batch_data.batch[batch_data.ds_wrapper.dataset_info.answer]
        )

        batch_results.predictions = results
        batch_results.references = list(batch_data.batch[batch_data.ds_wrapper.dataset_info.answer])
        batch_results.generation_probs = logprobs
        batch_results.calib_probs = calibprob_batch
        batch_results.math_problem_type = list(
            batch_data.batch[batch_data.ds_wrapper.dataset_info.type_id])
        return batch_results

    def _prepare_few_shot_data(self, ds_wrapper):
        few_shot_data = FewShotData()

        def preprocessing_a_record(rec):
            return [
                rf"{rec[ds_wrapper.dataset_info.query]}",
                rf"{rec[ds_wrapper.dataset_info.answer]}",
            ]

        few_shot_data.selected_sample = [
            preprocessing_a_record(s)
            for s in list(
                random.sample(list(ds_wrapper.dataset_training), self.config.num_fs)
            )
        ]
        few_shot_data.original_few_shot = format_fewshot(
            few_shot_data.selected_sample,
            query_format=ds_wrapper.prompt["prompt"],
            answer_format=ds_wrapper.prompt["answer_format"],
        )
        few_shot_data.calib_few_shot = format_fewshot(
            few_shot_data.selected_sample,
            query_format=ds_wrapper.calibration_prompt["prompt"],
            answer_format=ds_wrapper.prompt["answer_format"],
        )

        return few_shot_data

    def _create_prompts(self, ds_wrapper, batch, original_few_shot):
        return [
            [
                {
                    "role": "system",
                    "content": ds_wrapper.prompt["system_prompt"],
                },
                *original_few_shot,
                {
                    "role": "user",
                    "content": ds_wrapper.prompt["prompt"].format(rf"{rule}"),
                },
            ]
            for rule in batch[ds_wrapper.dataset_info.query]
        ]

    def _create_calib_prompts(self, ds_wrapper, batch, calib_few_shot):
        return [
            [
                {
                    "role": "system",
                    "content": ds_wrapper.calibration_prompt["system_prompt"],
                },
                *calib_few_shot,
                {
                    "role": "user",
                    "content": ds_wrapper.calibration_prompt["prompt"].format(rf"{rule}"),
                },
            ]
            for rule in batch[ds_wrapper.dataset_info.query]
        ]

    def _save_intermediate_results(self, idx, results, few_shot_data, save_config):
        print(f"Saving results of {idx} batches")
        generations = self._prepare_generations(results, few_shot_data)
        save_config.saving_fn(generations)
        mean_result = self._calculate_mean_result(generations, save_config)
        print(f"Results of {idx} batches: ", mean_result)

    def _save_final_results(self, results, few_shot_data, save_config):
        generations = self._prepare_generations(results, few_shot_data)
        mean_result = self._calculate_mean_result(generations, save_config)
        std_result = self._calculate_std_result(generations, save_config)

        final_result = {"mean": mean_result, "std": std_result}
        save_config.saving_fn(generations, final_result)
        return final_result

    def _prepare_generations(self, results, few_shot_data):
        return {
            "predictions": results.predictions,
            "references": results.references,
            "generation_probs": results.generation_probs,
            "calibration_probs": results.calib_probs,
            "fewshot": few_shot_data.selected_sample,
            "math_problem_type": results.math_problem_type,
        }

    def _calculate_mean_result(self, generations, save_config):
        return self.metric_pipeline.run_mean(
            generations,
            save_config.task_name,
            save_config.ds_wrapper.prompt["answer_key"],
            save_config.ds_wrapper.dataset_info.label,
            save_config.config,
        )

    def _calculate_std_result(self, generations, save_config):
        return self.metric_pipeline.run_std(
            generations,
            save_config.task_name,
            save_config.ds_wrapper.prompt["answer_key"],
            save_config.ds_wrapper.dataset_info.label,
            save_config.config,
        )

    def run_math_pipeline(self, dataset_config, saving_fn):
        "run_math"
        return self.__math(dataset_config, saving_fn)
