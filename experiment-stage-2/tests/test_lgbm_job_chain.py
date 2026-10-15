"""Execute the launcher with fake scheduler/Python boundaries; never submit real jobs."""
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == 'posix', 'Bash launcher tests run on the Linux server')
class ChainTests(unittest.TestCase):
    def execute(self, status, marker=False, attempt=0):
        script = Path(__file__).resolve().parents[1] / 'Japan_LGBM_train.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run/checkpoint').mkdir(parents=True)
            (root / 'run/checkpoint/manifest.json').write_text('{}')
            if marker:
                (root / 'run/checkpoint/completed.json').write_text('{}')
            python = root / 'python'
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_ROOT/python-args"\nexit '+str(status)+'\n')
            python.chmod(0o755)
            sbatch = root / 'sbatch'
            sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_ROOT/submitted"\necho 900\n')
            sbatch.chmod(0o755)
            env = dict(os.environ, TEST_ROOT=str(root), JAPAN_REPO_ROOT=str(root),
                       JAPAN_PYTHON=str(python), SLURM_JOB_ID='123', SLURM_CPUS_PER_TASK='4',
                       PATH=str(root)+os.pathsep+os.environ['PATH'])
            result = subprocess.run(['bash', str(script), '15', str(root/'run'), str(root/'next'), str(attempt)],
                                    env=env, capture_output=True, text=True, timeout=10)
            submitted = (root/'submitted').read_text() if (root/'submitted').exists() else ''
            args = (root/'python-args').read_text() if (root/'python-args').exists() else ''
            return result, submitted, args

    def test_only_successful_export_schedules_e16(self):
        result, submitted, args = self.execute(0, marker=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--dependency=afterok:123', submitted)
        self.assertIn('\n16\n', submitted)
        self.assertIn('--resume', args)

    def test_preemption_schedules_same_experiment(self):
        result, submitted, _ = self.execute(75)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--dependency=afterany:123', submitted)
        self.assertIn('\n15\n', submitted)
        self.assertTrue(submitted.endswith('\n1\n'))

    def test_error_or_missing_marker_never_starts_next(self):
        for status in (0, 1):
            result, submitted, _ = self.execute(status)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(submitted, '')

    def test_continuation_limit_prevents_endless_submission(self):
        result, submitted, _ = self.execute(75, attempt=9)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(submitted, '')

    def test_signal_during_wait_resumes_but_cancellation_does_not(self):
        script = Path(__file__).resolve().parents[1] / 'Japan_LGBM_train.sh'
        for sig, expected_status in ((signal.SIGUSR1, 0), (signal.SIGTERM, 143)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'run/checkpoint').mkdir(parents=True)
                (root/'run/checkpoint/manifest.json').write_text('{}')
                python = root/'python'
                python.write_text('#!/bin/bash\ntrap "exit 75" USR1 TERM\ntouch "$TEST_ROOT/ready"\nwhile :; do sleep 0.05; done\n')
                python.chmod(0o755)
                sbatch = root/'sbatch'
                sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_ROOT/submitted"\necho 900\n')
                sbatch.chmod(0o755)
                env = dict(os.environ, TEST_ROOT=str(root), JAPAN_REPO_ROOT=str(root),
                           JAPAN_PYTHON=str(python), SLURM_JOB_ID='123',
                           PATH=str(root)+os.pathsep+os.environ['PATH'])
                process = subprocess.Popen(['bash', str(script), '15', str(root/'run'), str(root/'next')],
                                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while not (root/'ready').exists() and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue((root/'ready').exists())
                    process.send_signal(sig)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, expected_status, stdout+stderr)
                    self.assertEqual((root/'submitted').exists(), sig == signal.SIGUSR1)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
