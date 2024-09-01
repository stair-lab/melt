import subprocess
import unittest

class TestWrapper(unittest.TestCase):
    """
    Unit tests for various wrappers used with the melt command-line tool.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up class-wide parameters used for testing different wrappers.
        """
        cls.model_name = "Qwen/Qwen2-0.5B-Instruct"
        cls.ptemplate = "chatglm"
        cls.lang = "vi"
        cls.seed = 42
        cls.smoke_test = True

    def build_command(self, dataset_name, wrapper_type):
        """
        Construct the melt command with the given parameters.

        Args:
            dataset_name (str): Name of the dataset.
            wrapper_type (str): Type of the wrapper to use.

        Returns:
            list: Command arguments to be passed to subprocess.run.
        """
        return [
            "melt",
            "--wtype", wrapper_type,
            "--model_name", self.model_name,
            "--dataset_name", dataset_name,
            "--ptemplate", self.ptemplate,
            "--lang", self.lang,
            "--seed", str(self.seed),
            "--smoke_test", str(self.smoke_test)
        ]

    def run_melt_command(self, dataset_name, wrapper_type):
        """
        Run the melt command with specified dataset and wrapper type, and check for success.

        Args:
            dataset_name (str): Name of the dataset.
            wrapper_type (str): Type of the wrapper to use.

        Raises:
            AssertionError: If the command fails with a non-zero exit code.
        """
        command = self.build_command(dataset_name, wrapper_type)
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            self.fail(f"Command failed for dataset '{dataset_name}' with wrapper '{wrapper_type}'\n"
                      f"Exit code: {result.returncode}\n"
                      f"stdout: {result.stdout}\n"
                      f"stderr: {result.stderr}")

    def test_wrapper_hf(self):
        """Test hf wrapper."""
        self.run_melt_command("zalo_e2eqa", "hf")

    def test_wrapper_tgi(self):
        """Test tgi wrapper."""
        self.run_melt_command("zalo_e2eqa", "tgi")

    def test_wrapper_gemini(self):
        """Test gemini wrapper."""
        self.run_melt_command("zalo_e2eqa", "gemini")

    def test_wrapper_openai(self):
        """Test openai wrapper."""
        self.run_melt_command("zalo_e2eqa", "openai")

    def test_wrapper_vllm(self):
        """Test vllm wrapper."""
        self.run_melt_command("zalo_e2eqa", "vllm")

if __name__ == '__main__':
    unittest.main()
