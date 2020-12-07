#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:16:41 2017

@author: Oren
"""

import os
import argparse
import logging
logger = logging.getLogger('main')

from sys import argv
from subprocess import call

def generate_sbatch_file(partition_name, tmp_dir, cmd, prefix_name, sbatch_file_path, CPUs):
    '''compose sbatch_file content and fetches it'''
    sbatch_file_content = '#!/usr/bin/env bash\n'
    if int(CPUs)>1:
        sbatch_file_content += f'#SBATCH -n {CPUs}\n'
    sbatch_file_content += f'#SBATCH -p {partition_name}\n'
    sbatch_file_content += f'#SBATCH -J {prefix_name}\n'
    sbatch_file_content += f'#SBATCH --error {tmp_dir}/{prefix_name}-%j.err\n' # error log (job_name-job_id.err)
    sbatch_file_content += f'#SBATCH --output {tmp_dir}/{prefix_name}-%j.out\n'
    sbatch_file_content += f'hostname\n'
    sbatch_file_content += f'echo job_name: {prefix_name}\n'    
    sbatch_file_content += f'echo $SLURM_JOBID\n'
    sbatch_file_content += f'module list\n'
    sbatch_file_content += f'{cmd}\n'
    with open(sbatch_file_path, 'w') as f_sbatch: # write the job
        f_sbatch.write(sbatch_file_content)
    call(['chmod', '+x', sbatch_file_path]) #set execution permissions    

    logger.debug('First job details for debugging:')
    logger.debug('#'*80)
    logger.debug('-> sbatch_file_path is:\n' + sbatch_file_path)
    logger.debug('\n-> sbatch_file_content is:\n' + sbatch_file_content)
    logger.debug('-> out file is at:\n' + os.path.join(tmp_dir, prefix_name+'.$JOB_ID.out'))
    logger.debug('#'*80)


def submit_cmds_from_file_to_slurm(cmds_file, tmp_dir, partition_name, CPUs, dummy_delimiter, start, end, additional_params):
    logger.debug('-> Jobs will be submitted to partition:' + partition_name)
    logger.debug('-> out, err and sbatch files will be written to:\n' + tmp_dir + '/')
    logger.debug('-> Jobs are based on cmds ' + str(start) + ' to ' + str(end) + ' (excluding) from:\n' + cmds_file)
    logger.debug('-> Each job will use ' + CPUs + ' CPU(s)\n')

    logger.debug('Starting to send jobs...')
    cmd_number = 0
    with open(cmds_file) as f_cmds:
        for line in f_cmds:
            if int(start) <= cmd_number < end and not line.isspace():
                try:
                    cmd, prefix_name = line.rstrip().split('\t')
                except: 
                    logger.error(f'UNABLE TO PARSE LINE:\n{line}')
                    raise 
                # the queue does not like very long commands so I use a dummy delimiter (!@# by default) to break the rows:
                cmd = cmd.replace(dummy_delimiter, '\n')
                sbatch_file_path = os.path.join(tmp_dir, prefix_name+'.sbatch') # path to job            
    
                generate_sbatch_file(partition_name, tmp_dir, cmd, prefix_name, sbatch_file_path, CPUs)

                #execute the job
                #partition_name may contain more arguments, thus the string of the cmd is generated and raw cmd is called
                terminal_cmd = f'/usr/bin/sbatch {sbatch_file_path} {additional_params}'
                logger.info(f'Submitting: {terminal_cmd}')
                
                call(terminal_cmd, shell = True)
    
            cmd_number += 1

    logger.debug('@ -> Sending jobs is done. @')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('cmds_file', help='A file containing jobs commands to execute in the queue. Each row should contain a (set of) command(s separated by $dummy_delimiter) then a "\t" and a job name', type=lambda file_path:str(file_path) if os.path.exists(file_path) else parser.error(f'{file_path} does not exist!'))
    parser.add_argument('tmp_dir', help='A temporary directory where the log files will be written to')
    parser.add_argument('-p', '--partition_name', help='The partition to which the job(s) will be submitted to', default='batch')
    parser.add_argument('--cpu', help='How many CPUs will be used?', choices=[str(i) for i in range(1,29)], default='1')
    parser.add_argument('--dummy_delimiter', help='The queue does not "like" very long commands; A dummy delimiter is used to break each row into different commands of a single job', default='!@#')
    parser.add_argument('--start', help='Skip jobs until $start', type=int, default=0)
    parser.add_argument('--end', help='Skip jobs from $end+1', type=int, default=float('inf'))
    parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true')
    parser.add_argument('--additional_params', help='Other specific parameters, such as, which machine to use', default='')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.debug(f'args = {args}')

    if not os.path.exists(args.tmp_dir):
        logger.debug(f'{args.tmp_dir} does not exist. Creating tmp path...')
        os.makedirs(args.tmp_dir, exist_ok=True)
        
    submit_cmds_from_file_to_slurm(args.cmds_file, args.tmp_dir, args.partition_name, args.cpu, args.dummy_delimiter, args.start, args.end, args.additional_params)
    #print(args.cmds_file, args.tmp_dir, args.partition_name, args.cpu, args.verbose, args.start, args.end)
