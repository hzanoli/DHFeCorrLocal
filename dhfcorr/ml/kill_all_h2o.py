import pandas as pd
import subprocess
from dhfcorr.ml.setup_sge_h2o import get_ip_from_node_name

current_nodes = pd.DataFrame(['node{:03d}'.format(n) for n in range(2, 13)])
current_nodes.columns = ['node_name']
current_nodes = current_nodes.append(pd.DataFrame(['quark'], columns=['node_name']), ignore_index=True)
current_nodes['node_ip'] = current_nodes['node_name'].apply(get_ip_from_node_name)

for ip, node in zip(current_nodes['node_ip'], current_nodes['node_name']):
    print('Killing server on node: ' + node)
    # run = subprocess.call(['ssh', ip, 'python3 -c "import h2o; h2o.connect(); h2o.cluster().shutdown()" '])
    run = subprocess.call(['ssh', ip, 'pkill java '])
    print('ssh result: ' + str(run))
