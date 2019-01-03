#!/usr/local/bin/python3.6

import json
import os
import subprocess
import sys
import time
import signal
import socket
import glob

from contextlib import contextmanager

CONST = {
    "gethostname_ld_preload": "/libchangehostname.so",
    "model_output_dir": "/opt/ml/model",
    "train_data_dir": "/opt/ml/input/data/train",
    "resource_config": "/opt/ml/input/config/resourceconfig.json",
    "hyperparameters_config": "/opt/ml/input/config/hyperparameters.json"
}

def setup():

    # Read info that SageMaker provides
    rc_path = CONST["resource_config"]
    with open(rc_path, 'r') as f:
        resources = json.load(f)
    current_host = resources["current_host"]
    all_hosts = resources["hosts"]

    hosts = list(all_hosts)

    # Apply gethostname() patch for OpenMPI
    _change_hostname(current_host)

    # Enable SSH connections between containers
    _start_ssh_daemon()

    if current_host == _get_master_host_name(hosts):
        _wait_for_worker_nodes_to_start_sshd(hosts)


class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)


def _change_hostname(current_host):
    """Compiles a shared library to correct the behavior of the gethostname system call,
        which OpenMPI depends on.

    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("change-hostname.sh {}".format(current_host))


def _get_master_host_name(hosts):
    return sorted(hosts)[0]

def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])

def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            print("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port, s):
    try:
        print("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        print("can connect to host %s", host)
        return True
    except socket.error:
        print("can't connect to host %s", host)
        return False


def wait_for_training_processes_to_appear_and_finish(proccess_id_string):

    training_process_started = False
    while True:
        time.sleep(60)
        training_process_ps = subprocess.check_output(f'ps -elf | grep "{proccess_id_string}"', encoding='utf-8', shell=True)
        print(training_process_ps)
        training_process_count = subprocess.check_output(f'ps -elf | grep "{proccess_id_string}" | wc -l', encoding='utf-8', shell=True)
        training_process_count_str = training_process_count.replace("\n", "").strip()
        print(training_process_count_str)
        training_process_running = int(training_process_count_str) > 2
        if training_process_started:
            print("training process still running")
            if not training_process_running:
                print("Process done. Exiting after 5s")
                time.sleep(5)
                sys.exit(0)

        if not training_process_started:
            if training_process_running:
                print("training started this iter")
                training_process_started = True

def build_host_arg(host_list, gpu_per_host):
    if len(host_list) == 1:
        return f'localhost:{gpu_per_host}'
    arg = ""
    for ind, host in enumerate(host_list):
        if ind != 0:
            arg += ","
        arg += f'{host}:{gpu_per_host}'
    return arg


def train():

    outdir = CONST["model_output_dir"]
    train_data_dir = CONST["train_data_dir"]

    gethostname_ld_preload = CONST["gethostname_ld_preload"]
    print("pre-setup check")
    setup()

    rc_path = "/opt/ml/input/config/resourceconfig.json"
    with open(rc_path, 'r') as f:
        resources = json.load(f)

    print("--- resourceconfig.json ---")
    print(json.dumps(resources, indent=4))
    print("--- END resourceconfig.json ---")

    current_host = resources["current_host"]
    all_hosts = resources["hosts"]

    is_master = current_host == sorted(all_hosts)[0]
    
    if not is_master:
        process_search_term = "/usr/local/bin/python3.6 /tensorpack/examples/FasterRCNN/train.py"
        wait_for_training_processes_to_appear_and_finish(process_search_term)
        print(f'Worker {current_host} has completed')
        

    hc_path = CONST["hyperparameters_config"]
    with open(hc_path, 'r') as f:
        hyperparamters = json.load(f)

        try:
            batch_norm = hyperparamters['batch_norm']
        except KeyError:
            batch_norm = 'FreezeBN'
        
        try:
            mode_fpn = hyperparamters['mode_fpn']
        except KeyError:
            mode_fpn = "True"
        
        try:
            mode_mask = hyperparamters['mode_mask']
        except KeyError:
            mode_mask = "True"

        try:
            gpus_per_host = hyperparamters['gpus_per_host']
        except KeyError:
            gpus_per_host = 8

        try:
            eval_period = hyperparamters['eval_period']
        except KeyError:
            eval_period = 10

        try:
            steps_per_epoch = hyperparamters['steps_per_epoch']
        except KeyError:
            steps_per_epoch = 1875

        try:
            lr_schedule = hyperparamters['lr_schedule']
        except KeyError:
            lr_schedule = '[120000, 160000, 180000]'
        
        numprocesses = len(all_hosts) * int(gpus_per_host)

        mpirun_cmd = f"""HOROVOD_CYCLE_TIME=0.5 \\
HOROVOD_FUSION_THRESHOLD=67108864 \\
mpirun -np {numprocesses} \\
--host {build_host_arg(all_hosts, gpus_per_host)} \\
--allow-run-as-root \\
--display-map \\
--tag-output \\
-mca btl_tcp_if_include ethwe \\
-mca oob_tcp_if_include ethwe \\
-x NCCL_SOCKET_IFNAME=ethwe \\
--mca plm_rsh_no_tree_spawn 1 \\
-bind-to none -map-by slot \\
-mca pml ob1 -mca btl ^openib \\
-mca orte_abort_on_non_zero_status 1 \\
-x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO \\
-x HOROVOD_CYCLE_TIME -x HOROVOD_FUSION_THRESHOLD \\
-x LD_LIBRARY_PATH -x PATH \\
-x LD_PRELOAD={gethostname_ld_preload} \\
--output-filename {outdir} \\
/usr/local/bin/python3.6 /tensorpack/examples/FasterRCNN/train.py \
--logdir {outdir}/train_log/maskrcnn \
--config DATA.BASEDIR={train_data_dir} \
MODE_FPN={mode_fpn} \
MODE_MASK={mode_mask} \
BACKBONE.WEIGHTS={train_data_dir}/pretrained-models/COCO-R50FPN-MaskRCNN-Standard.npz \
BACKBONE.NORM={batch_norm} \
DATA.TRAIN='["train2017"]' \
DATA.VAL='val2017' \
TRAIN.STEPS_PER_EPOCH={steps_per_epoch} \
TRAIN.EVAL_PERIOD={eval_period} \
TRAIN.LR_SCHEDULE='{lr_schedule}' \
TRAINER=horovod"""

        print("--------Begin MPI Run Command----------")
        print(mpirun_cmd)
        print("--------End MPI Run Comamnd------------")
        try:
            process = subprocess.Popen(mpirun_cmd, encoding='utf-8', shell=True, 
                stdout=subprocess.PIPE)

            while True:
                if process.poll() != None:
                    break
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                    
            exitcode = process.poll()
        except Exception as e:
            print("train exception occured")
            exitcode = 1
            print(str(e))

        sys.exit(exitcode)

if __name__ == "__main__":
    train()
