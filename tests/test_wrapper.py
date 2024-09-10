import subprocess
import unittest


class TestWrapper(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWrapper, self).__init__(*args, **kwargs)
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.ptemplate = "chatglm"
        self.lang = "vi"  # Set the lang argument to "vi"
        self.seed = 42  # Set the seed to 42
        self.smoke_test = True  # Set the smoke_test argument to True

    def run_melt_command(self, dataset_name, wrapper_type):
        result = subprocess.run(
            [
                "melt",
                "--wtype",
                wrapper_type,
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

    def test_wrapper_hf(self):
        # Test wrapper hf
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "hf")

    def test_wrapper_tgi(self):
        # Test wrapper tgi
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "tgi")

    def test_wrapper_gemini(self):
        # Test wrapper gemini
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "gemini")

    def test_wrapper_openai(self):
        # Test wrapper openai
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "openai")

    def test_wrapper_vllm(self):
        # Test wrapper vllm
        dataset_name = "zalo_e2eqa"
        self.run_melt_command(dataset_name, "vllm")


if __name__ == "__main__":
    unittest.main()
