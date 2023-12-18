from pathlib import Path
import logging

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import datalad.api as dl


logging.getLogger('datalad').setLevel(logging.WARNING)

dataset_url = {
    'HCP-YA': 'git@github.com:datalad-datasets/human-connectome-project-openaccess.git',
    'HCP-A': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git',
    'HCP-D': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git',
    'ABCD': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git'}


class _InitDiffusionDataInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    subject = traits.Str(mandatory=True, desc='subject ID')
    int_dir = traits.Directory(desc='directory containing intermediate files from previous runs')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')


class _InitDiffusionDataOutputSpec(TraitedSpec):
    d_files = traits.Dict(dtype=Path, desc='filenames of diffusion data')
    t1_files = traits.Dict(dtype=Path, desc='filenames of T1w data')
    fs_files = traits.Dict(dtype=Path, desc='filenames of FreeSurfer outputs')
    dataset_dir = traits.Directory(desc='absolute path to installed root dataset')
    fs_dir = traits.Directory(desc='FreeSurfer subject directory')


class InitDiffusionData(SimpleInterface):
    """Instal and get subject-specific diffusion data"""
    input_spec = _InitDiffusionDataInputSpec
    output_spec = _InitDiffusionDataOutputSpec

    def _run_interface(self, runtime):
        self._results['dataset_dir'] = Path(self.inputs.work_dir, self.inputs.subject)
        dataset_dirs = {
            'HCP-YA': Path(self._results['dataset_dir']),
            'HCP-A': Path(self._results['dataset_dir'], 'original', 'hcp', 'hcp_aging'),
            'HCP-D': Path(self._results['dataset_dir'], 'original', 'hcp', 'hcp_development'),
            'ABCD': Path(self._results['dataset_dir'], 'original', 'abcd', 'abcd-hcp-pipeline')}
        dataset_dir = dataset_dirs[self.inputs.dataset]

        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            source = 'inm7-storage'
        else:
            source = None

        # install datasets
        dl.install(
            path=self._results['dataset_dir'], source=dataset_url[self.inputs.dataset],
            on_failure='stop')
        dl.get(
            path=dataset_dir, dataset=self._results['dataset_dir'], get_data=False, source=source,
            on_failure='stop')
        if self.inputs.dataset == 'HCP-YA':
            subject_dir = Path(dataset_dir, 'HCP1200', self.inputs.subject)
        else:
            subject_dir = Path(dataset_dir, self.inputs.subject)
        dl.get(
            path=subject_dir, dataset=dataset_dir, get_data=False, source=source,
            on_failure='stop')

        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            if self.inputs.int_dir:
                d_dir = self.inputs.int_dir
                d_files = {
                    'data': Path(d_dir, 'data.nii.gz'), 'bval': Path(d_dir, 'bvals'),
                    'bvec': Path(d_dir, 'bvecs'), 'mask': Path(d_dir, 'nodif_brain_mask.nii.gz')}
            else:
                d_dir = Path(subject_dir, 'unprocessed', 'Diffusion')
                dl.get(
                    path=d_dir.parent, dataset=dataset_dir, get_data=False, source=source,
                    on_failure='stop')
                d_files = {}
                for dirs in [98, 99]:
                    for phase in ['AP', 'PA']:
                        for ftype in ['.nii.gz', '.bval', '.bvec']:
                            key = f'dir{dirs}_{phase}{ftype}'
                            d_files[key] = Path(d_dir, f'{self.inputs.subject}_dMRI_{key}')
        elif self.inputs.dataset == 'HCP-YA':
            d_dir = Path(subject_dir, 'T1w', 'Diffusion')
            dl.get(
                path=d_dir.parent, dataset=dataset_dir, get_data=False, source=source,
                on_failure='stop')
            d_files = {
                'data': Path(d_dir, 'data.nii.gz'), 'bval': Path(d_dir, 'bvals'),
                'bvec': Path(d_dir, 'bvecs'), 'mask': Path(d_dir, 'nodif_brain_mask.nii.gz')}

        self._results['d_files'] = d_files.copy()
        for key, val in d_files.items():
            if val.is_symlink():
                dl.get(path=val, dataset=d_dir.parent, source=source, on_failure='stop')
            elif not self.inputs.int_dir:
                self._results['d_files'][key] = ''

        anat_dir = Path(subject_dir, 'T1w')
        fs_dir = Path(subject_dir, 'T1w', self.inputs.subject)
        mni_dir = Path(subject_dir, 'MNINonLinear')
        dl.get(
            path=anat_dir, dataset=dataset_dir, get_data=False, source=source, on_failure='stop')
        dl.get(
            path=mni_dir, dataset=dataset_dir, get_data=False, source=source, on_failure='stop')

        fs_files = {
            'lh_aparc': Path(fs_dir, 'label', 'lh.aparc.annot'),
            'rh_aparc': Path(fs_dir, 'label', 'rh.aparc.annot'),
            'lh_pial': Path(fs_dir, 'surf', 'lh.pial'),
            'rh_pial': Path(fs_dir, 'surf', 'rh.pial'),
            'lh_white': Path(fs_dir, 'surf', 'lh.white'),
            'rh_white': Path(fs_dir, 'surf', 'rh.white'),
            'lh_white_deformed': Path(fs_dir, 'surf', 'lh.white.deformed'),
            'rh_white_deformed': Path(fs_dir, 'surf', 'rh.white.deformed'),
            'lh_reg': Path(fs_dir, 'surf', 'lh.sphere.reg'),
            'rh_reg': Path(fs_dir, 'surf', 'rh.sphere.reg'),
            'lh_cort_label': Path(fs_dir, 'label', 'lh.cortex.label'),
            'rh_cort_label': Path(fs_dir, 'label', 'rh.cortex.label'),
            'lh_ribbon': Path(fs_dir, 'mri', 'lh.ribbon.mgz'),
            'rh_ribbon': Path(fs_dir, 'mri', 'rh.ribbon.mgz'),
            'ribbon': Path(fs_dir, 'mri', 'ribbon.mgz'),
            'aseg': Path(fs_dir, 'mri', 'aseg.mgz'),
            'aparc_aseg': Path(fs_dir, 'mri', 'aparc+aseg.mgz'),
            'orig': Path(fs_dir, 'mri', 'orig.mgz'),
            'brain_mask': Path(fs_dir, 'mri', 'brainmask.mgz'),
            'talaraich_xfm': Path(fs_dir, 'mri', 'transforms', 'talairach.xfm'),
            'norm': Path(fs_dir, 'mri', 'norm.mgz'),
            'eye': Path(fs_dir, 'mri', 'transforms', 'eye.dat'),
            'lh_thickness': Path(fs_dir, 'surf', 'lh.thickness'),
            'rh_thickness': Path(fs_dir, 'surf', 'rh.thickness')}
        self._results['fs_dir'] = fs_dir

        self._results['fs_files'] = fs_files.copy()
        for key, val in fs_files.items():
            if val.is_symlink():
                dl.get(path=val, dataset=anat_dir, source=source, on_failure='stop')
            else:
                self._results['fs_files'][key] = ''

        t1_files = {
            't1': Path(anat_dir, 'T1w_acpc_dc.nii.gz'),
            't1_restore': Path(anat_dir, 'T1w_acpc_dc_restore.nii.gz'),
            't1_restore_brain': Path(anat_dir, 'T1w_acpc_dc_restore_brain.nii.gz'),
            'bias': Path(anat_dir, 'BiasField_acpc_dc.nii.gz'),
            'fs_mask': Path(anat_dir, 'brainmask_fs.nii.gz'),
            't1_to_mni': Path(mni_dir, 'xfms', 'acpc_dc2standard.nii.gz'),
            'aseg': Path(mni_dir, 'mri', 'aseg.mgz'),
            'aparc_aseg': Path(mni_dir, 'mri', 'aparc+aseg.mgz'),
            'brainmask': Path(mni_dir, 'mri', 'brainmask.mgz'),
            'talairach_xfm': Path(mni_dir, 'mri', 'xfms', 'talairach.xfm'),
            'norm': Path(mni_dir, 'mri', 'norm.mgz')}

        self._results['t1_files'] = t1_files.copy()
        for key, val in t1_files.items():
            if val.is_symlink():
                if key == 't1_to_mni' or key == 'aseg':
                    dl.get(path=val, dataset=mni_dir, source=source, on_failure='stop')
                else:
                    dl.get(path=val, dataset=anat_dir, source=source, on_failure='stop')
            else:
                self._results['t1_files'][key] = ''

        return runtime
