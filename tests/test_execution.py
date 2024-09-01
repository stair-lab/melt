import subprocess
import unittest
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestTasks(unittest.TestCase):
    """Test various tasks using the melt command."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = {
            "model_name": "Qwen/Qwen2-0.5B-Instruct",
            "ptemplate": "chatglm",
            "wrapper_type": "vllm",
            "lang": "vi",
            "seed": 42,
            "smoke_test": True
        }

    def run_melt_command(self, dataset_name):
        """Run the melt command and assert success.

        Args:
            dataset_name (str): The name of the dataset to test.
        """
        cmd_args = [
            "melt", "--wtype", self.config["wrapper_type"],
            "--model_name", self.config["model_name"],
            "--dataset_name", dataset_name,
            "--ptemplate", self.config["ptemplate"],
            "--lang", self.config["lang"],
            "--seed", str(self.config["seed"]),
            "--smoke_test", str(self.config["smoke_test"])
        ]
        
        result = subprocess.run(cmd_args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, 
                         "Command failed for dataset %s with output: %s\n%s" %
                         (dataset_name, result.stdout, result.stderr))

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

if __name__ == '__main__':
    unittest.main()
