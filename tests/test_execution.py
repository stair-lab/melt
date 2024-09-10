import subprocess
import unittest


class TestTasks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTasks, self).__init__(*args, **kwargs)
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.ptemplate = "chatglm"
        self.wrapper_type = "vllm"
        self.lang = "vi"  # Set the lang argument to "vi"
        self.seed = 42  # Set the seed to 42
        self.smoke_test = True  # Set the smoke_test argument to True

    def run_melt_command(self, dataset_name):
        result = subprocess.run(
            [
                "melt",
                "--wtype",
                self.wrapper_type,
                "--model_name",
                self.model_name,
                "--dataset_name",
                dataset_name,
                "--ptemplate",
                self.ptemplate,
                "--lang",
                self.lang,
                "--seed",
                str(self.seed),
                "--smoke_test",
                str(self.smoke_test),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)

    def test_sentiment_analysis(self):
        # Test sentiment analysis task
        dataset_name = "UIT-VSFC"
        self.run_melt_command(dataset_name)

    def test_text_classification(self):
        # Test text classification task
        dataset_name = "UIT-VSMEC"
        self.run_melt_command(dataset_name)

    def test_toxic_detection(self):
        # Test toxic detection task
        dataset_name = "ViHSD"
        self.run_melt_command(dataset_name)

    def test_reasoning(self):
        # Test reasoning task
        dataset_name = "synthetic_natural_azr"
        self.run_melt_command(dataset_name)

    def test_open_ended_knowledge(self):
        # Test open-ended knowledge task
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name)

    def test_multiple_choice_knowledge(self):
        # Test multiple choice knowledge task
        dataset_name = "ViMMRC"
        self.run_melt_command(dataset_name)

    def test_math(self):
        # Test math task
        dataset_name = "math_level1_azr"
        self.run_melt_command(dataset_name)

    def test_translation(self):
        # Test translation task
        dataset_name = "opus100_envi"
        self.run_melt_command(dataset_name)

    def test_summarization(self):
        # Test summarization task
        dataset_name = "wiki_lingua"
        self.run_melt_command(dataset_name)

    def test_question_answering(self):
        # Test question answering task
        dataset_name = "xquad_xtreme"
        self.run_melt_command(dataset_name)

    def test_information_retrieval(self):
        # Test information retrieval task
        dataset_name = "mmarco"
        self.run_melt_command(dataset_name)


if __name__ == "__main__":
    unittest.main()
