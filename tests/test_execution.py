"""
This module contains a test suite for evaluating various tasks
using the MELT command-line interface with different datasets.
"""

import subprocess
import unittest


class TestTasks(unittest.TestCase):
    """Test suite for evaluating various tasks using the MELT command-line interface."""

    def __init__(self, *args, **kwargs):
        """Initialize the test with default model settings."""
        super().__init__(*args, **kwargs)
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.ptemplate = "chatglm"
        self.wrapper_type = "vllm"
        self.lang = "vi"
        self.seed = 42
        self.smoke_test = True

    def run_melt_command(self, dataset_name):
        """
        Run the melt command with given dataset name and verify it executes successfully.

        Args:
            dataset_name (str): Name of the dataset to use with the melt command.

        Raises:
            AssertionError: If the command fails with a non-zero exit code.
        """
        command = [
            "melt",
            "--wtype", self.wrapper_type,
            "--model_name", self.model_name,
            "--dataset_name", dataset_name,
            "--ptemplate", self.ptemplate,
            "--lang", self.lang,
            "--seed", str(self.seed),
            "--smoke_test", str(self.smoke_test)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        # Provide detailed error information if the command fails
        if result.returncode != 0:
            self.fail(f"Command failed for dataset '{dataset_name}' "
                      f"with exit code {result.returncode}\n"
                      f"stdout: {result.stdout}\n"
                      f"stderr: {result.stderr}")

    def test_sentiment_analysis(self):
        """Test sentiment analysis task."""
        dataset_name = "UIT-VSFC"
        self.run_melt_command(dataset_name)

    def test_text_classification(self):
        """Test text classification task."""
        dataset_name = "UIT-VSMEC"
        self.run_melt_command(dataset_name)

    def test_toxic_detection(self):
        """Test toxic detection task."""
        dataset_name = "ViHSD"
        self.run_melt_command(dataset_name)

    def test_reasoning(self):
        """Test reasoning task."""
        dataset_name = "synthetic_natural_azr"
        self.run_melt_command(dataset_name)

    def test_open_ended_knowledge(self):
        """Test open-ended knowledge task."""
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name)

    def test_multiple_choice_knowledge(self):
        """Test multiple choice knowledge task."""
        dataset_name = "ViMMRC"
        self.run_melt_command(dataset_name)

    def test_math(self):
        """Test math task."""
        dataset_name = "math_level1_azr"
        self.run_melt_command(dataset_name)

    def test_translation(self):
        """Test translation task."""
        dataset_name = "opus100_envi"
        self.run_melt_command(dataset_name)

    def test_summarization(self):
        """Test summarization task."""
        dataset_name = "wiki_lingua"
        self.run_melt_command(dataset_name)

    def test_question_answering(self):
        """Test question answering task."""
        dataset_name = "xquad_xtreme"
        self.run_melt_command(dataset_name)

    def test_information_retrieval(self):
        """Test information retrieval task."""
        dataset_name = "mmarco"
        self.run_melt_command(dataset_name)

if __name__ == '__main__':
    unittest.main()
