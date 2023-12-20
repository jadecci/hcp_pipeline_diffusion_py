from pathlib import Path
from shutil import copyfile
import logging

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import datalad.api as dl

logging.getLogger('datalad').setLevel(logging.WARNING)


class _InitDataInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")


class _InitDataOutputSpec(TraitedSpec):
    d_files = traits.Dict(dtype=Path, desc='filenames of diffusion data')
    t1_files = traits.Dict(dtype=Path, desc='filenames of T1w data')
    fs_files = traits.Dict(dtype=Path, desc='filenames of FreeSurfer outputs')


class InitData(SimpleInterface):
    """Get subject-specific diffusion data"""
    input_spec = _InitDataInputSpec
    output_spec = _InitDataOutputSpec

    def _run_interface(self, runtime):
        sub_dir = self.inputs.config["subject_dir"]
        dataset_dir = self.inputs.config["dataset_dir"]
        source = self.inputs.config["source"]

        d_dir = Path(sub_dir, "unprocessed", "Diffusion")
        anat_dir = Path(sub_dir, "T1w")
        fs_dir = Path(anat_dir, self.inputs.config["subject"])
        if dataset_dir is not None:
            for dir in sub_dir, d_dir.parent, anat_dir:
                dl.get(
                    path=dir, dataset=dataset_dir, get_data=False, source=source, on_failure='stop')

        d_files = {}
        for ndir in self.inputs.config["ndirs"]:
            for phase in self.inputs.config["phases"]:
                for file_type in [".nii.gz", ".bval", ".bvec"]:
                    key = f"dir{ndir}_{phase}{file_type}"
                    d_files[key] = Path(d_dir, f"{self.inputs.subject}_dMRI_{key}")
        self._results["d_files"] = d_files.copy()
        if dataset_dir is not None:
            for key, val in d_files.items():
                dl.get(path=val, dataset=d_dir.parent, source=source, on_failure='stop')

        fs_files = {
            'lh_white_deformed': Path(fs_dir, 'surf', 'lh.white.deformed'),
            'rh_white_deformed': Path(fs_dir, 'surf', 'rh.white.deformed'),
            'eye': Path(fs_dir, 'mri', 'transforms', 'eye.dat')}
        self._results['fs_files'] = fs_files.copy()
        if dataset_dir is not None:
            for key, val in fs_files.items():
                dl.get(path=val, dataset=anat_dir, source=source, on_failure='stop')

        t1_files = {
            't1': Path(anat_dir, 'T1w_acpc_dc.nii.gz'),
            't1_restore': Path(anat_dir, 'T1w_acpc_dc_restore.nii.gz'),
            't1_restore_brain': Path(anat_dir, 'T1w_acpc_dc_restore_brain.nii.gz'),
            'bias': Path(anat_dir, 'BiasField_acpc_dc.nii.gz'),
            'fs_mask': Path(anat_dir, 'brainmask_fs.nii.gz')}
        self._results['t1_files'] = t1_files.copy()
        if dataset_dir is not None:
            for key, val in t1_files.items():
                dl.get(path=val, dataset=anat_dir, source=source, on_failure='stop')

        return runtime


class _SaveDataInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    data_file = traits.File(mandatory=True, exists=True, desc="Diffusion data file")
    bval_file = traits.File(mandatory=True, exists=True, desc="b value file")
    bvec_file = traits.File(mandatory=True, exists=True, desc="b vector file")
    mask_file = traits.File(mandatory=True, exists=True, desc="mask file")


class SaveData(SimpleInterface):
    """Save processed diffusion data to output folder"""
    input_spec = _SaveDataInputSpec

    def _run_interface(self, runtime):
        out_dir = self.inputs.config["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        copyfile(self.inputs.data_file, Path(out_dir, "data.nii.gz"))
        copyfile(self.inputs.bval_file, Path(out_dir, "bvals"))
        copyfile(self.inputs.bvec_file, Path(out_dir, "bvecs"))
        copyfile(self.inputs.mask_file, Path(out_dir, "nodif_brain_mask.nii.gz"))

        return runtime
