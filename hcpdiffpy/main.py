from pathlib import Path
import argparse
import logging
from typing import Union

import pandas as pd
import nipype.pipeline as pe
from nipype import config

from hcpdiffpy.interfaces.data import InitDiffusionData
from hcpdiffpy.interfaces.preproc import HCPMinProc
from hcpdiffpy.utilities.utilities import SimgCmd

logging.getLogger('datalad').setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Multimodal neuroimaging feature extraction',
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument('dataset', type=str, help='Dataset (HCP-YA, HCP-A, HCP-D, ABCD)')
    parser.add_argument('sublist', type=str, help='Absolute path to the subject list (.csv).')
    parser.add_argument(
        '--diffusion', dest='diffusion', action='store_true',
        help='Process diffusion data and extract diffusion features')
    parser.add_argument(
        '--diff_int_dir', type=Path, dest='int_dir', default=None,
        help='Directory containing diffusion intermediate files from previous runs')
    parser.add_argument(
        '--workdir', type=Path, dest='work_dir', default=Path.cwd(), help='Work directory')
    parser.add_argument(
        '--output_dir', type=Path, dest='output_dir', default=Path.cwd(), help='Output directory')
    parser.add_argument(
        '--simg', type=Path, dest='simg', default=None,
        help='singularity image to use for command line functions from FSL / FreeSurfer / '
             'Connectome Workbench / MRTrix3.')
    parser.add_argument(
        '--overwrite', dest='overwrite', action="store_true", help='Overwrite existing results')
    parser.add_argument(
        '--condordag', dest='condordag', action='store_true',
        help='Submit graph workflow to HTCondor')
    parser.add_argument(
        '--wrapper', type=str, dest='wrapper', default='', help='Wrapper script for HTCondor')
    parser.add_argument(
        '--debug', dest='debug', action="store_true", help='Use debug configuration')
    args = parser.parse_args()

    simg_cmd = SimgCmd(
        simg=args.simg, work_dir=args.work_dir, out_dir=args.output_dir, int_dir=args.int_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sublist = pd.read_csv(args.sublist, header=None).squeeze('columns')
    for subject in sublist:
        output_file = Path(args.output_dir, f'{args.dataset}_{subject}.h5')
        if not output_file.is_file() or args.overwrite:
            subject_wf = init_d_wf(
                args.dataset, str(subject), args.work_dir, args.output_dir, simg_cmd,
                args.overwrite, args.int_dir)

            subject_wf.config['execution']['try_hard_link_datasink'] = 'false'
            subject_wf.config['execution']['crashfile_format'] = 'txt'
            subject_wf.config['execution']['stop_on_first_crash'] = 'true'
            subject_wf.config['monitoring']['enabled'] = 'true'
            if args.debug:
                config.enable_debug_mode()

            subject_wf.write_graph()
            if args.condordag:
                subject_wf.run(
                    plugin='CondorDAGMan',
                    plugin_args={
                        'dagman_args': f'-outfile_dir {args.work_dir}', 'wrapper_cmd': args.wrapper,
                        'dagman_args': '-import_env',
                        'override_specs': 'request_memory = 5 GB\nrequest_cpus = 1'})
            else:
                subject_wf.run()


def init_d_wf(
        dataset: str, subject: str, work_dir: Path, output_dir: Path, simg_cmd: SimgCmd,
        overwrite: bool, int_dir: Union[Path, None]) -> pe.Workflow:
    d_wf = pe.Workflow(f'subject_{subject}_diffusion_wf', base_dir=work_dir)
    work_curr = Path(work_dir, f'subject_{subject}_diffusion_wf')
    init_data = pe.Node(
        InitDiffusionData(
            dataset=dataset, work_dir=work_curr, subject=subject, output_dir=output_dir),
        name='init_data')
    if (dataset == 'HCP-A' or dataset == 'HCP-D') and int_dir is None:
        hcp_proc = HCPMinProc(
            dataset=dataset, work_dir=work_curr, subject=subject, simg_cmd=simg_cmd).run()
        hcp_proc_wf = hcp_proc.outputs.hcp_proc_wf
        d_wf.connect([
            (init_data, hcp_proc_wf, [
                ('d_files', 'inputnode.d_files'), ('t1_files', 'inputnode.t1_files'),
                ('fs_files', 'inputnode.fs_files'), ('fs_dir', 'inputnode.fs_dir')])])

    return d_wf


if __name__ == '__main__':
    main()
