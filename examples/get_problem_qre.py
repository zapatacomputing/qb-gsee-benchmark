################################################################################
# © Copyright 2024 Zapata Computing Inc.
################################################################################

import gzip
import json
import os
import shutil
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import paramiko
import requests
from benchq.resource_estimators.openfermion_estimator import openfermion_estimator
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyscf.tools import fcidump

from qb_gsee_benchmark.qre import get_df_qpe_circuit


def fetch_file_from_sftp(
    url=None, local_path=None, ppk_path=None, username=None, port=None
):

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    remote_path = parsed_url.path.lstrip("/")

    with paramiko.SSHClient() as client:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(hostname)
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            key_filename=ppk_path,
        )

        with client.open_sftp() as sftp:
            sftp.get(remote_path, local_path)


port = 22
ppk_path = "/Users/maxradin/.ssh/darpa-qb-key.ppk"
username = "darpa-qb"


url = "https://raw.githubusercontent.com/jp7745/qb-gsee-problem-instances/main/problem_instances/problem_instance.mn_mono.cb40f3f7-ffe8-40e8-4544-f26aad5a8bd8.json"
response = requests.get(url)
if response.status_code != 200:
    raise RuntimeError(f"Failed to retrieve {url}. Status code: {response.status_code}")

problem_instance = response.json()

solution_data: list[dict[str, Any]] = []


def retrieve_fcidump(url: str) -> dict:
    filename = os.path.basename(urlparse(url).path)
    fetch_file_from_sftp(
        url=url, username=username, ppk_path=ppk_path, local_path=filename, port=port
    )
    fcidump_filename = filename.replace(".gz", "")
    with gzip.open(filename, "rb") as f_in:
        with open(fcidump_filename.replace(".gz", ""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return fcidump.read(filename=fcidump_filename)


for instance in problem_instance["instance_data"]:
    fci = retrieve_fcidump(instance["instance_data_object_url"])

    start = time.time()
    circuit, n_shots, allowable_error = get_df_qpe_circuit(
        fci=fci,
        error_tolerance=1e-3,
        failure_tolerance=1e-2,
        square_overlap=0.8**2,
        df_threshold=1e-3,
    )
    preprocessing_time = time.time() - start
    print(f"Time to get circuit: {preprocessing_time}")

    logical_resources = estimate_resources(circuit.circuit)

    physical_resources = openfermion_estimator(
        num_logical_qubits=logical_resources["LogicalQubits"],
        num_t=logical_resources["T"],
    )
    algorithm_run_time = n_shots * physical_resources.total_time_in_seconds

    solution_data.append(
        {
            "instance_data_object_uuid": instance["instance_data_object_uuid"],
            "run_time": {
                "overall_time": {
                    "seconds": preprocessing_time + algorithm_run_time,
                },
                "preprocessing_time": {
                    "seconds": preprocessing_time,
                },
                "algorithm_run_time": {
                    "seconds": algorithm_run_time,
                },
                "postprocessing_time": {
                    "seconds": 0,
                },
            },
        }
    )


compute_details = {
    "description": "Double factorized QPE resource estimates based on methodology of arXiv:2406.06335. Uses PyLIQTR logical resource estimates with BenchQ footprint analysis. Ground-state overlap assumed to be 0.8 and double-factorized truncation threshold to be 1e-3 Ha. Note that the truncation error is not included in the error bounds and that the SCF compute time is not included in the preprocessing time."
}

solution_uuid = str(uuid4())
current_time = datetime.now(UTC)
current_time_string = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


results = {
    "$schema": "https://raw.githubusercontent.com/jp7745/qb-file-schemas/main/schemas/solution.schema.0.0.1.json",
    "solution_uuid": solution_uuid,
    "problem_instance_uuid": problem_instance["problem_instance_uuid"],
    "creation_timestamp": current_time_string,
    "short_name": "QPE",
    "contact_info": [
        {
            "name": "Max Radin",
            "email": "max.radin@zapata.ai",
            "institution": "Zapata AI",
        }
    ],
    "solution_data": solution_data,
    "compute_hardware_type": "quantum_computer",
    "compute_details": compute_details,
    "digital_signature": None,
}

with open("qpe_results.json", "w") as f:
    f.write(json.dumps(results, indent=4))
