import json
import os
import subprocess
import sys
import tempfile
import unittest

from training_settings import load_training_settings


class TrainingSettingsTests(unittest.TestCase):
    def test_uses_portable_defaults_when_no_local_file_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, source = load_training_settings(directory)

        self.assertEqual(settings.num_envs, 120)
        self.assertEqual(settings.device, "auto")
        self.assertTrue(source.endswith("local/training_settings.json"))

    def test_local_file_overrides_only_supported_training_values(self):
        with tempfile.TemporaryDirectory() as directory:
            local_dir = os.path.join(directory, "local")
            os.makedirs(local_dir)
            path = os.path.join(local_dir, "training_settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_envs": 10,
                        "device": "cpu",
                        "learning_rate": 0.0001,
                        "ppo_n_steps": 128,
                        "ignored": "not a setting",
                    },
                    f,
                )

            settings, source = load_training_settings(directory)

        self.assertEqual(settings.num_envs, 10)
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.learning_rate, 0.0001)
        self.assertEqual(settings.ppo_n_steps, 128)
        self.assertEqual(source, path)

    def test_rejects_invalid_values_with_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            local_dir = os.path.join(directory, "local")
            os.makedirs(local_dir)
            with open(
                os.path.join(local_dir, "training_settings.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"num_envs": 0}, f)

            with self.assertRaisesRegex(ValueError, "num_envs"):
                load_training_settings(directory)

    def test_train_module_uses_local_num_envs_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"num_envs": 3}, f)

            env = os.environ | {"PKMAI_SETTINGS_FILE": path}
            result = subprocess.run(
                [sys.executable, "-c", "import train; print(train.NUM_ENVS)"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "3")


if __name__ == "__main__":
    unittest.main()
