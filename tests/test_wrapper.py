"""
Unit tests for the wrapper functionality using the 'melt' command.
"""

import subprocess
import unittest

class TestWrapper(unittest.TestCase):
    """
    Test cases for various wrapper types using the 'melt' command.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the test class with common parameters.
        """
        super().__init__(*args, **kwargs)
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.ptemplate = "chatglm"
        self.lang = "vi"  # Set the lang argument to "vi"
        self.seed = 42  # Set the seed to 42
        self.smoke_test = True  # Set the smoke_test argument to True

    def run_melt_command(self, dataset_name, wrapper_type):
        """
        Run the 'melt' command with specified dataset and wrapper type, and check for success.
        
        Args:
            dataset_name (str): The name of the dataset to use.
            wrapper_type (str): The type of wrapper to use.
        """
        command = [
            "melt", "--wtype", wrapper_type, "--model_name", self.model_name,
            "--dataset_name", dataset_name, "--ptemplate", self.ptemplate,
            "--lang", self.lang, "--seed", str(self.seed), "--smoke_test", str(self.smoke_test)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(result.returncode, 0)

    def test_wrapper_hf(self):
        """
        Test the 'hf' wrapper type with the specified dataset.
        """
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "hf")

    def test_wrapper_tgi(self):
        """
        Test the 'tgi' wrapper type with the specified dataset.
        """
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "tgi")

    def test_wrapper_gemini(self):
        """
        Test the 'gemini' wrapper type with the specified dataset.
        """
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "gemini")

    def test_wrapper_openai(self):
        """
        Test the 'openai' wrapper type with the specified dataset.
        """
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "openai")

    def test_wrapper_vllm(self):
        """
        Test the 'vllm' wrapper type with the specified dataset.
        """
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "vllm")

if __name__ == '__main__':
    unittest.main()