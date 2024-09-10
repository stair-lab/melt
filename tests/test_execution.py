import subprocess
import unittest


class TestTasks(unittest.TestCase):
    """
    Unit tests for various tasks using the melt command-line tool.
    """

    def setUp(self):
        """
        Set up test parameters that are used across all test cases.
        """
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
        
        result = subprocess.run(command, capture_output=True, text=True)

        # Provide detailed error information if the command fails
        if result.returncode != 0:
            self.fail(f"Command failed for dataset '{dataset_name}' with exit code {result.returncode}\n"
                      f"stdout: {result.stdout}\n"
                      f"stderr: {result.stderr}")

    def test_sentiment_analysis(self):
        """Test sentiment analysis task."""
        self.run_melt_command("UIT-VSFC")

    def test_text_classification(self):
        """Test text classification task."""
        self.run_melt_command("UIT-VSMEC")

    def test_toxic_detection(self):
        """Test toxic detection task."""
        self.run_melt_command("ViHSD")

    def test_reasoning(self):
        """Test reasoning task."""
        self.run_melt_command("synthetic_natural_azr")

    def test_open_ended_knowledge(self):
        """Test open-ended knowledge task."""
        self.run_melt_command("zalo_e2eqa")

    def test_multiple_choice_knowledge(self):
        """Test multiple choice knowledge task."""
        self.run_melt_command("ViMMRC")

    def test_math(self):
        """Test math task."""
        self.run_melt_command("math_level1_azr")

    def test_translation(self):
        """Test translation task."""
        self.run_melt_command("opus100_envi")

    def test_summarization(self):
        """Test summarization task."""
        self.run_melt_command("wiki_lingua")

    def test_question_answering(self):
        """Test question answering task."""
        self.run_melt_command("xquad_xtreme")

    def test_information_retrieval(self):
        """Test information retrieval task."""
        self.run_melt_command("mmarco")

if __name__ == "__main__":
    unittest.main()
