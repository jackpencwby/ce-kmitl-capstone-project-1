import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

@unittest.skipUnless(os.name=='posix','Requires Linux Bash')
class ParallelLauncherTests(unittest.TestCase):
    def test_signal_rollover_and_failed_worker_cancels_successor(self):
        script=Path(__file__).resolve().parents[1]/'GBR_parallel_train.sh'
        for fail in ('','11'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                scripts={
                    'python': '''while [[ "$1" != "--experiment" ]]; do shift; done
e=$2
stop() { touch "$TEST_ROOT/stopped-$e"; [[ "$e" != "$FAIL_EXP" ]] || exit 1; exit 75; }
trap stop USR1
touch "$TEST_ROOT/started-$e"
while true; do sleep .05; done
''',
                    'sbatch': 'echo "$*" >> "$TEST_ROOT/submitted"; echo 456\n',
                    'scancel': 'echo "$*" >> "$TEST_ROOT/cancelled"\n',
                }
                for name, body in scripts.items():
                    path=root/name; path.write_text('#!/bin/bash\n'+body); path.chmod(0o755)
                env=dict(os.environ,SLURM_SUBMIT_DIR=tmp,SLURM_JOB_ID='123',GBR_PYTHON=str(root/'python'),TEST_ROOT=tmp,FAIL_EXP=fail,PATH=tmp+os.pathsep+os.environ['PATH'])
                process=subprocess.Popen(['bash',str(script),str(root/'runs')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
                try:
                    deadline=time.monotonic()+10
                    while len(list(root.glob('started-*')))<6 and time.monotonic()<deadline:
                        time.sleep(.02)
                    self.assertEqual(len(list(root.glob('started-*'))),6)
                    process.send_signal(signal.SIGUSR1)
                    out,err=process.communicate(timeout=10)
                    self.assertEqual(process.returncode,1 if fail else 0,out+err)
                    self.assertEqual(len(list(root.glob('stopped-*'))),6,out+err)
                    submissions=(root/'submitted').read_text().splitlines()
                    self.assertEqual(len(submissions),1)
                    self.assertIn('--dependency=afterany:123',submissions[0])
                    self.assertTrue(submissions[0].endswith(' 1 123'))
                    if fail:
                        self.assertEqual((root/'cancelled').read_text().strip(),'456')
                    else:
                        self.assertFalse((root/'cancelled').exists())
                        self.assertEqual(out.count('exit=75'),6)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL); process.communicate()

    def test_six_processes_finish_and_failure_is_reported(self):
        script=Path(__file__).resolve().parents[1]/'GBR_parallel_train.sh'
        for fail in ('','11'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); (root/'experiment-stage-2').mkdir()
                fake=root/'python'
                fake.write_text('#!/bin/bash\nwhile [[ "$1" != "--experiment" ]]; do shift; done\ne=$2\ntouch "$TEST_ROOT/started-$e"\nwhile [[ $(ls "$TEST_ROOT"/started-* | wc -l) -lt 6 ]]; do sleep .02; done\n[[ "$e" != "$FAIL_EXP" ]]\n')
                fake.chmod(0o755)
                env=dict(os.environ,SLURM_SUBMIT_DIR=tmp,GBR_PYTHON=str(fake),TEST_ROOT=tmp,FAIL_EXP=fail)
                r=subprocess.run(['bash',str(script),str(root/'runs')],env=env,capture_output=True,text=True,timeout=10)
                self.assertEqual(len(list(root.glob('started-*'))),6,r.stderr)
                self.assertEqual(r.returncode,1 if fail else 0,r.stdout+r.stderr)

    def test_cancelled_predecessor_cannot_restart_training(self):
        script=Path(__file__).resolve().parents[1]/'GBR_parallel_train.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            control=root/'scontrol'
            control.write_text('#!/bin/bash\necho JobId=123 JobState=CANCELLED\n'); control.chmod(0o755)
            env=dict(os.environ,SLURM_SUBMIT_DIR=tmp,PATH=tmp+os.pathsep+os.environ['PATH'])
            r=subprocess.run(['bash',str(script),str(root/'runs'),'1','123'],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(r.returncode,1,r.stderr)
            self.assertFalse((root/'runs').exists())
            self.assertIn('not restarting',r.stdout)

    def test_accounting_fallback_after_controller_purges_job(self):
        script=Path(__file__).resolve().parents[1]/'GBR_parallel_train.sh'
        for state, expected in [('COMPLETED',0),('TIMEOUT',0),('CANCELLED',1),('FAILED',1),('',1)]:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for name, body in [('scontrol','exit 1'),('sacct',f'echo "123|{state}"'),('python','exit 0')]:
                    path=root/name
                    path.write_text('#!/bin/bash\n'+body+'\n'); path.chmod(0o755)
                env=dict(os.environ,SLURM_SUBMIT_DIR=tmp,GBR_PYTHON=str(root/'python'),PATH=tmp+os.pathsep+os.environ['PATH'])
                r=subprocess.run(['bash',str(script),str(root/'runs'),'1','123'],env=env,capture_output=True,text=True,timeout=10)
                self.assertEqual(r.returncode,expected,r.stdout+r.stderr)
                self.assertEqual(len(list((root/'runs').glob('E*.log'))),6 if expected==0 else 0)
