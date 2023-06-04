# coding=utf-8
# Copyright 2021 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The entry point for running a Dopamine agent on continuous control envs.

"""

from absl import app
from absl import flags
from absl import logging
import os
import sys

# from dopamine.continuous_domains import run_experiment
from bonus_based_exploration.cc_intrinsic_motivation import run_experiment


RUN_INFO_SUBDIR = 'run_info'


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string('sub_dir', None,
                    'Optional sub-directory, joined to base_dir, for hyperparameter sweeps etc.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/jax/agents/sac/configs/sac.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS

RUN_INFO_SUBDIR = 'run_info'

def log_experiment_launch_information(base_dir, gin_files):
  run_info_dirname = os.path.join(base_dir, RUN_INFO_SUBDIR)
  os.makedirs(run_info_dirname, exist_ok=True)
  commit_filename = os.path.join(run_info_dirname, 'commit.txt')
  diff_filename = os.path.join(run_info_dirname, 'diff.diff')
  assert os.system(f'git diff > {diff_filename}') == 0, "git diff failed"
  assert os.system(f'git log | head -n 1 > {commit_filename}') == 0, "git log failed"

  for gin_filename in gin_files:
    # final_part = os.path.basename(gin_filename)
    replaced_filename = gin_filename.replace("/", "__")
    write_filename = os.path.join(run_info_dirname, replaced_filename)
    assert os.system(f"cp {gin_filename} {write_filename}") == 0, f"cp failed for {gin_filename}"
  

  with open(os.path.join(run_info_dirname, 'command.json'), 'w') as f:
    print(sys.argv, file=f)


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """

  logging.set_verbosity(logging.INFO)
  base_dir = FLAGS.base_dir
  assert FLAGS.sub_dir not in  ["checkpoints", "logs", RUN_INFO_SUBDIR], "those are reserved subdirs"
  if FLAGS.sub_dir is not None:
    full_dir = os.path.join(FLAGS.base_dir, FLAGS.sub_dir)
    print('new logging directory: ', full_dir)
  else:
    full_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings

  run_experiment.load_gin_configs(gin_files, gin_bindings)
  log_experiment_launch_information(full_dir, FLAGS.gin_files)
  runner = run_experiment.create_continuous_exploration_runner(full_dir)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
