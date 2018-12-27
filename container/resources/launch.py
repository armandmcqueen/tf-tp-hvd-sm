#!/usr/local/bin/python3.6

import json
import subprocess
import sys


CONST = {
    "hyperparameters_config": "/opt/ml/input/config/hyperparameters.json"
}

def launch_training():

    # Read info that SageMaker provides
    hc_path = CONST["hyperparameters_config"]
    exitcode = 0
      
    try:
        with open(hc_path, 'r') as f:
            hyperparamters = json.load(f)
            train_script = hyperparamters["train_script"]
            print(f"Hyperparameter train_script: {train_script}")

            subprocess.check_call(f"chmod u+x {train_script}",  shell=True)
            process = subprocess.Popen([train_script], encoding='utf-8', stdout=subprocess.PIPE)
        
            while True:
                if process.poll() != None:
                    break
                output = process.stdout.readline()
                if output:
                    print(output.strip())

            exitcode = process.poll()
            print(f"Training script exit code: {exitcode}")
    except Exception as e:
        print("Launch training exception occured")
        exitcode = 1
        print(str(e))

    sys.exit(exitcode)

if __name__ == "__main__":
    launch_training()