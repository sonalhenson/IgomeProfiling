import os
import sys
from subprocess import call, run, Popen, PIPE
from time import time, sleep
import global_params
import logging

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO)


nnk_table: {str: str} = {"CGT": "R", "CGG": "R", "AGG": "R",
                         "CTT": "L", "CTG": "L", "TTG": "L",
                         "TCT": "S", "TCG": "S", "AGT": "S",
                         "GCT": "A", "GCG": "A",
                         "GGT": "G", "GGG": "G",
                         "CCT": "P", "CCG": "P",
                         "ACT": "T", "ACG": "T",
                         "CAG": "Q",
                         "TAG": "Q",  # rather than q
                         "GTT": "V", "GTG": "V",
                         "AAT": "N",
                         "GAT": "D",
                         "TGT": "C",
                         "GAG": "E",
                         "CAT": "H",
                         "ATT": "I",
                         "AAG": "K",
                         "ATG": "M",
                         "TTT": "F",
                         "TGG": "W",
                         "TAT": "Y"}

def verify_file_is_not_empty(file_path):
    import logging
    from time import sleep
    logger = logging.getLogger('main')
    total_retries = 2 # +1 including first try
    retries_count = 0
    sleep_time = 2
    sleep_factor = 2
    # make sure that there are results and the file is not empty
    with open(file_path) as f:
        if len(f.read(10).strip()) == 0:
            # TODO: write error to a global error file
            if retries_count == total_retries:
                msg = f'Input file is empty {file_path}'
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.warn(f'Input file is empty {file_path}, retrying in {sleep_time} seconds ({retries_count}/{total_retries})')
                retries_count += 1
                sleep(sleep_time)
                sleep_time *= sleep_factor
        return


def load_fasta_to_dict(fasta_path, reverse=False, upper_keys=False, upper_values=False):
    """
    :param fasta_path: a path to a FASTA file
    :return: a dictionary that maps each header (without ">" and rstriped()) to its corresponding sequence (rstriped())
             an int that represent the number of sequences
             an int that represent the length of the alignment
    """
    key2value = {}
    with open(fasta_path) as f:
        for i, header in enumerate(f):
            if not header.startswith('>'):
                raise TypeError(f'Illegal fasta file. Illegal record is record number {i} is\n{header}')
            # returns header without ">" !
            if not reverse:
                key = header[1:].rstrip()
                value = f.readline().rstrip()
                if key in key2value:
                    raise ValueError(f'{key} already appears in dict!!\n'
                                     f'old entry: {key}:{key2value[key]}\n'
                                     f'new entry: {key}:{value}')
                key2value[key] = value
            else:
                key = f.readline().rstrip()
                value = header[1:].rstrip()
                if key in key2value:
                    raise ValueError(f'{key} already appears in dict!!\n'
                                     f'old entry: {key}:{key2value[key]}\n'
                                     f'new entry: {key}:{value}')
                key2value[key] = value
    if not reverse:
        return key2value, len(key2value), len(key2value[header[1:].rstrip()])
    else:
        return key2value, len(key2value)

def measure_time(total):
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours != 0:
        return f'{hours}:{minutes:02}:{seconds:02} hours'
    elif minutes != 0:
        return f'{minutes}:{seconds:02} minutes'
    else:
        return f'{seconds} seconds'


def wait_for_results(script_name, path, num_of_expected_results, error_file_path, example_cmd='', suffix='done',
                     remove=False, time_to_wait=10, start=0, done_files_list=None):
    """
    :param script_name:
    :param path:
    :param num_of_expected_results:
    :param error_file_path:
    :param example_cmd:
    :param suffix:
    :param remove:
    :param time_to_wait:
    :param start:
    :param done_files_list:
    :return: waits until path contains num_of_expected_results $suffix files
    """
    # if True: return
    if not start:
        start = time()
    logger.info(f'Waiting for {script_name}...\nContinues when {num_of_expected_results} results with suffix="{suffix}" will be in:\n{path}')
    if example_cmd:
        logger.info(f'An example command looks like this:\n{example_cmd}\n\n')
    if num_of_expected_results==0:
        logger.fatal(f'\n{"#"*100}\nnum_of_expected_results in {path} is {num_of_expected_results}!\nSomething went wrong in the previous step...\n{"#"*100}')
        #raise ValueError(f'\n{"#"*100}\nnum_of_expected_results is {num_of_expected_results}! Something went wrong in the previous step...\n{"#"*100}')
    total_time = 0
    i = 0
    current_num_of_results = 0
    while current_num_of_results < num_of_expected_results:
        sleep(time_to_wait)
        current_num_of_results = sum(1 for x in os.listdir(path) if x.endswith(suffix))
        jobs_left = num_of_expected_results - current_num_of_results
        total_time += time_to_wait
        i += 1
        if i % 5 == 0:  # print status every 5 cycles of $time_to_wait
            logger.info(f'\t{measure_time(total_time)} have passed since started waiting ({num_of_expected_results} - {current_num_of_results} = {jobs_left} more files are still missing)')
            if done_files_list:
                logger.info(f'This are the files the are still missing:\n{[file for file in done_files_list if not os.path.exists(file)]}')
        assert not os.path.exists(error_file_path), f'An error occurred. For further details see: {error_file_path}'

    if remove:
        # execute(['python', '-u', '/groups/pupko/orenavr2/pipeline/RemoveDoneFiles.py', path, suffix])
        call(f'rm {path}/*{suffix}', shell=True)

    end = time()
    logger.info(f'{path} contains {current_num_of_results} done files!')
    logger.info(f'Done waiting for: {script_name}\n(took {measure_time(int(end-start))}).\n')
    assert not os.path.exists(error_file_path), f'An error occurred. For further details see: {error_file_path}'


def submit_pipeline_step_to_cluster(script_path, params_lists, tmp_dir, job_name, queue_name, verbose, new_line_delimiter='!@#',
                         q_submitter_script_path='/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py',
                         required_modules_as_list=None, num_of_cpus=1, executable='python', done_path=None):
    required_modules_as_str = 'python/python-anaconda3.6.5-orenavr2'
    if required_modules_as_list:
        # don't forget a space after the python module!!
        required_modules_as_str += ' ' + ' '.join(required_modules_as_list)
    cmds_as_str = f'module load {required_modules_as_str}'
    # the queue does not like very long commands so I use a dummy delimiter (!@#) to break the rows in q_submitter
    cmds_as_str += new_line_delimiter

    if executable is None:
        executable = ''

    for params in params_lists:
        cmds_as_str += ' '.join([executable, script_path, *[str(param) for param in params]] + (['-v'] if verbose else [])).lstrip() + ';'
        cmds_as_str += new_line_delimiter

    example_cmd = ' '.join([executable, script_path, *[str(param) for param in params]] + (['-v'] if verbose else [])).lstrip() + ';'

    # GENERATE DONE FILE
    # write an empty string (like "touch" command)
    # cmds_as_str += ' '.join(['python', done_files_script_path, os.path.join(tmp_dir, job_name + '.done'), 'done'])+';'
    # cmds_as_str += new_line_delimiter

    cmds_as_str += '\t' + job_name + '\n'
    cmds_path = os.path.join(tmp_dir, f'{job_name}.cmds')
    if os.path.exists(cmds_path):
        cmds_path = os.path.join(tmp_dir, f'{job_name}_{time()}.cmds')
    with open(cmds_path, 'w') as f:
        f.write(cmds_as_str)

    # process_str = f'{q_submitter_script_path} {cmds_path} {tmp_dir} -q {queue_name} --cpu {num_of_cpus}'
    process = [q_submitter_script_path, cmds_path, tmp_dir, '-q', queue_name, '--cpu', str(num_of_cpus)]
    logger.info(f'Calling:\n{" ".join(process)}')
    # if True: return
    run(process)
    return example_cmd


def build_args(executable, script_path, params, verbose):
    args = []
    if executable:
        args.append(executable)
    if script_path:
        args.append(script_path)
    args += [str(x) for x in params]
    if verbose:
        args.append('-v')
    return ' '.join(args)


def create_command(script_path, params_lists, tmp_dir, job_name, queue_name, verbose, new_line_delimiter='!@#',
                   q_submitter_script_path='/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py',
                   required_modules_as_list=None, num_of_cpus=1, executable='python', done_path=None):
    new_line_delimiter = '\n'
    cmds_as_str = ''
    
    for params in params_lists:
        cmds_as_str += build_args(executable, script_path, params, verbose)
        cmds_as_str += new_line_delimiter

    example_cmd = build_args(executable, script_path, params, verbose)

    cmds_path = os.path.join(tmp_dir, f'{job_name}.cmds')
    if os.path.exists(cmds_path):
        cmds_path = os.path.join(tmp_dir, f'{job_name}_{time()}.cmds')
    with open(cmds_path, 'w') as f:
        f.write(cmds_as_str)

    if global_params.local_command_prefix:
        process = f'{global_params.local_command_prefix} {cmds_path}'
    else:
        process = cmds_path
    return process, example_cmd


def submit_pipeline_step_to_celery(script_path, params_lists, tmp_dir, job_name, queue_name, verbose, new_line_delimiter='!@#',
                         q_submitter_script_path='/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py',
                         required_modules_as_list=None, num_of_cpus=1, executable='python', done_path=None):
    process, example_cmd = create_command(**locals())
    from worker import submit
    logger.info(f'Calling using celery:\n{process}')
    submit.delay(process, shell=True)
    return example_cmd


def run_step_locally(script_path, params_lists, tmp_dir, job_name, queue_name, verbose, new_line_delimiter='!@#',
                         q_submitter_script_path='/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py',
                         required_modules_as_list=None, num_of_cpus=1, executable='python', done_path=None):
    process, example_cmd = create_command(**locals())
    logger.info(f'Calling:\n{process}')
    if global_params.run_local_in_parallel_mode:
        Popen(process, shell=True)
    else:
        run(process, shell=True)
    return example_cmd

def submit_pipeline_step(script_path, params_lists, tmp_dir, job_name, queue_name, verbose, new_line_delimiter='!@#',
                         q_submitter_script_path='/bioseq/bioSequence_scripts_and_constants/q_submitter_power.py',
                         required_modules_as_list=None, num_of_cpus=1, executable='python', done_path = None):
    if done_path and os.path.exists(done_path):
        process, example_cmd = create_command(**locals())
        logger.info(f'Skipping "{example_cmd}" as "{done_path}" exists')
        return example_cmd
    if global_params.run_using_celery:
        return submit_pipeline_step_to_celery(**locals())
    if global_params.is_run_on_cluster:
        return submit_pipeline_step_to_cluster(**locals())
    
    return run_step_locally(**locals())


def fetch_cmd(script_name, parameters, verbose, error_path, done_path=None):
    cmd = f'python3 {script_name} ' + ' '.join(parameters + (['-v'] if verbose else []))
    if done_path and os.path.exists(done_path):
        logger.info(f'Skipping "{cmd}" as "{done_path}" exists')
        return
    logger.info(f'Executing:\n{cmd}')
    # try:
    run(cmd, shell=True)
    # logger.info(f'Finished:\n{cmd}')
    # except Exception as e:
    #     fail(error_path, e)



def load_table_to_dict(table_path, error_msg, delimiter ='\t'):
    table = {}
    with open(table_path) as f:
        for line in f:
            if line.isspace():  # empty line
                continue
            if line.startswith('#'):  # ignore symbol
                continue
            key, value = line.strip().split(delimiter)
            if key in table:
                assert False, error_msg.replace('{}', key)  # TODO: write to a global error log file
            table[key] = value
    return table


def fail(error_path, e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    with open(error_path, 'w') as f:
        f.write(f'\n{"$"*100}\n\n{fname}: {exc_type}, '
                f'at line: {exc_tb.tb_lineno}\n\ne.args: {e.args}\n\n{"$"*100}')
    raise e


def get_cluster_rank_from(header):
    return int(header.split('clusterRank_')[-1].split('_')[0])


def get_unique_members_from(header):
    unique_members = header.split('uniqueMembers_')[-1].split('_')[0]
    if unique_members.startswith('top'):
        unique_members = unique_members[3:]
    return int(unique_members)


def get_cluster_size_from_name(path):
    return float(os.path.splitext(path)[0].split('clusterSize_')[1])


def get_count_from(header):
    # e.g., >seq_1_lib_C10C_len_12_counts_325350.363668618
    if 'Type' not in header:
        return float(header.split('_')[-1])
    # e.g., >1_Length_12_Repeats_318612.4098079804_Type_C10C
    return float(header.split('_')[-3])


def get_configuration_from(header):
    if 'Type' not in header:
        # e.g., >seq_1_lib_C10C_len_12_counts_325350.363668618
        return header.split('_')[3]
    # e.g., >1_Length_12_Repeats_318612.4098079804_Type_C10C
    return header.split('_')[-1]


def remove_redundant_newlines_from_fasta(input_file_path, output_file_path):
    # remove redundant newlines (MAFFT puts only 100 chars per line), i.e., this:
    #>seq_1_lib_C10C_len_12_counts_297739.4827131018
    #----------------------C--------------H------GKTGASFL----Q---
    #C---------------------
    # will turn into this:
    #>seq_1_lib_C10C_len_12_counts_297739.4827131018
    #----------------------C--------------H------GKTGASFL----Q---C---------------------
    result = ''
    sequence = ''
    with open(input_file_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if sequence != '':  # not the first header
                    result += f'{sequence}\n'
                sequence = ''
                result += f'{line}\n'
            else:
                sequence += line
    result += f'{sequence}\n'

    with open(output_file_path, 'w') as f:
        f.write(result)


def count_memes(path):
    p = Popen(f'cat {path} | grep MOTIF | wc -l', stdout=PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    count = int(output) if p_status == 0 else 0
    print(f'Found {count} memes in {path}')
    return count
