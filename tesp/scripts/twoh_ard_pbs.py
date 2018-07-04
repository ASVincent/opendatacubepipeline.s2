#!/usr/bin/env python

"""
PBS End-to-End additions

Approach:
    1 - cronjob to qsub index level 1
    2 - datacube query to generate task list and qsub wagl
    3 - index level 2 data
"""

import os
import sys
import time
import math

import yaml
from pathlib import Path

from datacube.scripts.dataset import index_dataset_paths, load_rules_from_types
from datacube.ui.common import get_metadata_path
from datacube.ui.click import global_cli_options
from datacube.ui import pass_index

from .ard_pbs import PBS_RESOURCES, main as qsub_wagl

ROOT = '/g/data1a/u46/users/av8534/automanualisation'

DATACUBE_CONF= ROOT + '/datacube.conf'
DATACUBE_ENV='' + 'asv-dev'

BASE_DIR = '/g/data/da82/AODH/USGS/L1/Landsat/C1/yamls_l1/'
LVL1_PATTERN = 'LC08*.yaml'
PBS_PROJECT = 'u46'
START_TIME = time.time()
HOURS_BEHIND = 7 * 24
ONE_HOUR = 60 * 60

WORKERS = {
    'normal': {
        'workers': 16,
        'memory': 64,
        'granules_per_node': 100
    },
    'normalbw': {
        'workers': 28,
        'memory': 128,
        'granules_per_node': 200
    }
}


def generate_level1_taskfile(dataset_paths, taskfile):
    granule_locations = []
    for location in dataset_paths:
        md_file = get_metadata_path(location)
        with yaml_file.open('r') as yaml_info:
            granule_files.append(str(Path(yaml.load(yaml_info.read())['image']['bands']['blue']['path']).parent))
    with Path(taskfile).open('w') as fd:
        fd.write('\n'.join(granule_files)

    return (taskfile, len(granule_locations))


@click.group()
@global_cli_options
def cli():
    pass


@cli.command('index-wagl-output')
@click.option('--packaged-dir', help='base directory to index level 2 product')
@click.option('--submit-time', help='datetime processing job was submitted')
@click.option('--dry-run', help='Check if everything is ok', is_flag=True, default=False)
@pass_index()
def qsub_level2_index(index, packaged_dir, submit_time, dry_run, skip_lineage=False):
    def _age_filter(path):
        return path.stat().st_ctime > submit_time 

    output_yaml_pattern = 'ARD-METADATA.yaml'

    rules = load_rules_from_types(index)
    dataset_paths = Path(packaged_dir).rglob(output_yaml_pattern)
    dataset_paths = filter(_age_filter, dataset_query)

    index_dataset_paths('verify', dry_run, index, rules, dataset_paths, skip_lineage)


@cli.command('qsub-index-lvl2-workflow')
@click.option('--packaged-dir', help='base directory to index level 2 product')
@click.option('--submit-time', help='datetime processing job was submitted')
def qsub_index_level2(packaged_dir, submit_time):
    pbs_job = (
        'twoh_ard_pbs qsub-index-lvl2-workflow --packaged_dir {} --submit-time {}'
    ).format(packaged_dir, submit_time)

    taskfile = ROOT + '/level2-index.job'

    with open(taskfile, 'w') as fd:
        fd.write('\n'.join([
            PBS_RESOURCES.format(
                project=PBS_PROJECT,
                queue='express',
                walltime='10:00:00',
                memory='2',  # GB
                ncpus=1,  # just a find over filesystem
                jobfs=2,  # GB
                email='Alexander.vincent@ga.gov.au'
            ),
            'source {}/automanualisation_env'.format(ROOT),
            pbs_job
        ]))

    index_job = subprocess.Popen(['qsub', taskfile])
    stdout, stderr = index_job.communicate()

    if index_job.returncode == 0:
        _LOG.info('Indexing task submitted successfully')
    else:
        _LOG.error('Indexing task failed to execute')
        err_msg = stderr.decode('ascii').strip()
        _LOG.error(err_msg)
        raise RuntimeError(err_msg)

    return stdout.decode('ascii').strip()


@cli.command('qsub-wagl-workflow')
@click.option('--yaml-dir', help='base directory to search', default=BASE_DIR)
@click.option('--file-pattern', help='glob pattern to match for metadata identification', default=LVL1_PATTERN)
@click.option('--jobfile', help='location to save jobfile')
@click.option('--sources-policy', type=click.Choice(['verify', 'ensure', 'skip']), default='verify',
              help="""'verify' - verify source datasets' metadata (default)
'ensure' - add source dataset if it doesn't exist
'skip' - dont add the derived dataset if source dataset doesn't exist""")
def qsub_level1_index(yaml_dir, files_pattern, jobfile, sources_policy):
    pbs_job = (
        './twoh_ard_pbs qsub-wagl-workflow --yaml-dir {} --file-pattern {} --taskfile {} --sources-poliicy {}'
    ).format(yaml_dir, file_pattern, jobfile, sources_policy)

    taskfile = ROOT + '/level1-run.job'


    with open(taskfile, 'w') as fd:
        fd.write('\n'.join([
            PBS_RESOURCES.format(
                project=PBS_PROJECT,
                queue='express',
                walltime='10:00:00',
                memory='2',  # GB
                ncpus=1,  # just a find over filesystem
                jobfs=2,  # GB
                email='Alexander.vincent@ga.gov.au'
            ),
            'source {}/automanualisation_env'.format(ROOT),
            pbs_job
        ]))

    index_job = subprocess.Popen(['qsub', taskfile])

    if index_job.returncode == 0:
        _LOG.info('Indexing task submitted successfully')
    else:
        _LOG.error('Indexing task failed to execute')
        _ , stderr = index_job.communicate()

        if stderr:
            raise RuntimeError(stderr.decode('ascii').strip())
        else:
            raise RuntimeError('Task failed to submit to pbs')


@cli.command('wagl-workflow')
@click.option('--yaml-dir', help='base directory to search', default=BASE_DIR)
@click.option('--file-pattern', help='glob pattern to match for metadata identification', default=FILES_PATTERN)
@click.option('--taskfile', help='location to save taskfile')
@click.option('--sources-policy', type=click.Choice(['verify', 'ensure', 'skip']), default='verify',
              help="""'verify' - verify source datasets' metadata (default)
'ensure' - add source dataset if it doesn't exist
'skip' - dont add the derived dataset if source dataset doesn't exist""")
@click.options('--hours-behind', help='Process files whose age is less than this', default=24)
@click.option('--dry-run', help='Check if everything is ok', is_flag=True, default=False)
@pass_index()
def run(index, yaml_dir, file_pattern, taskfile, sources_policy, hours_behind, dry_run, skip_lineage=False):

    def _age_filter(path):
        file_age = START_TIME - path.stat().st_ctime
        return all(file_age <= 3600 * hours_behind, file_age > 600)

    # filter matched dataset files
    dataset_query = Path(yaml_dir).rglob(file_pattern)
    dataset_paths = filter(_age_filter, dataset_query)

    rules = load_rules_from_types(index)
    # Index them
    index_dataset_paths(sources_policy, dry_run, index, rules, dataset_paths, skip_lineage)
    # Generate a taskfile for wagl
    taskfile, task_cnt = generate_level1_taskfile(taskfile)
    queue = 'normal'  # TODO
    nodes = math.ceil(task_cnt / WORKERS[queue]['granules_per_node'])

    nci_jobs = qsub_wagl(
        taskfile, 
        wagl_dir / 'workdir',
        wagl_dir / 'logdir',
        wagl_dir / 'pkgdir',
        env_file, # TODO
        WORKERS[queue]['workers'],
        nodes,
        WORKERS[queue]['memory'],
        50, # jobfs requested
        PBS_PROJECT,
        queue,
        '48:00:00', # walltime
        'alexander.vincent@ga.gov.au',  # email
        dry_run
    )


if __name__ == '__main__':
    cli()
