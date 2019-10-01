import h2o
import os
import subprocess
import pandas as pd
import time


def get_ip_from_node_name(name):
    check_ip = subprocess.run(('getent hosts ' + name).split(' '), stdout=subprocess.PIPE)
    ip = check_ip.stdout.decode('utf-8').split(' ')[0]
    return ip


def setup_h2o_cluster_sge(cluster_name):
    temp_folder = os.getenv('TMPDIR')
    current_nodes = pd.read_csv(temp_folder + '/machines', sep=':', header=None)
    current_nodes.columns = ['node_name', 'n_cores']
    current_nodes['node_ip'] = current_nodes['node_name'].apply(get_ip_from_node_name)
    print(current_nodes)

    h2o_location = h2o.__file__.split('h2o/__init__.py')[0] + 'h2o/backend/bin/h2o.jar'
    ip_list = ''

    for ip in current_nodes['node_ip']:
        ip_list = ip_list + ip + r'/32' + ','
    ip_list = ip_list[:-1]

    run_h2o_arg = "java -Xmx14g -jar " + h2o_location + " -name " + cluster_name + " -network " + ip_list

    for ip, node in zip(current_nodes['node_ip'], current_nodes['node_name']):
        print('Creating server on node: ' + node)
        argument_ssh = "ssh " + ip + ' ' + run_h2o_arg
        argument_ssh = argument_ssh + r' &> '
        file_stdout = "h2o_out_" + cluster_name + '_' + node + ".txt"
        argument_ssh = argument_ssh + file_stdout
        run = subprocess.Popen(argument_ssh, shell=True)
        print('ssh result: ' + str(run))
        time.sleep(2)

    print('Created server on all nodes')
    time.sleep(4)
