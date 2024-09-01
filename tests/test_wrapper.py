import subprocess
import unittest

class TestWrapper(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.ptemplate = "chatglm"
        self.lang = "vi"  # Language argument
        self.seed = 42  # Random seed
        self.smoke_test = True  # Smoke test flag
        self.dataset_name = "zalo_e2eqa"  # Dataset name
        self.wrapper_types = ["hf", "tgi", "gemini", "openai", "vllm"]  # List of wrapper types

    def run_melt_command(self, wrapper_type):
        """Run the melt command and assert success.

        Args:
            wrapper_type (str): Type of wrapper to test.
        """
        result = subprocess.run(
            [
                "melt", "--wtype", wrapper_type,
                "--model_name", self.model_name,
                "--dataset_name", self.dataset_name,
                "--ptemplate", self.ptemplate,
                "--lang", self.lang,
                "--seed", str(self.seed),
                "--smoke_test", str(self.smoke_test)
            ],
            capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, f"Failed for {wrapper_type} with output: {result.stdout}\n{result.stderr}")

    def test_wrappers(self):
        """Test all wrapper types."""
        for wrapper_type in self.wrapper_types:
            with self.subTest(wrapper_type=wrapper_type):
                self.run_melt_command(wrapper_type)

if __name__ == '__main__':
    unittest.main()
