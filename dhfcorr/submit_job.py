import dhfcorr.definitions as definitions


def get_job_command(name, script, arguments=None, cluster='stbc', queue=definitions.JOB_QUEUE):
    if cluster == 'stbc':
        qsub_cmd = 'qsub '
        if arguments is not None:
            qsub_cmd += '-F \"' + arguments + '\"'
        qsub_cmd += ' -q ' + queue + ' -j oe -V -N ' + name
        qsub_cmd += ' ' + script
        return qsub_cmd

    elif cluster == 'quark':
        qsub_args = '-V -cwd'

    elif cluster == 'local':
        cmd = 'python ' + script
        if arguments is not None:
            cmd = ' ' + str(arguments)
        return cmd
    else:
        raise ValueError("The selected cluster is not defined.")
