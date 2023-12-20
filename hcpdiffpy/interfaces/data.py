from pathlib import Path
from shutil import copyfile
import logging

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits

logging.getLogger('datalad').setLevel(logging.WARNING)


class _InitDataInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")


class _InitDataOutputSpec(TraitedSpec):
    d_files = traits.Dict(dtype=Path, desc='filenames of diffusion data')
    t1_file = traits.File(exists=True, desc="T1 file")
    t1_restore_file = traits.File(exists=True, desc="T1 restored file")
    t1_brain_file = traits.File(exists=True, desc="T1 brain restored file")
    bias_file = traits.File(exists=True, desc="Bias file")
    mask_file = traits.File(exists=True, desc="FreeSurfer mask file")
    eye_file = traits.File(exists=True, desc="FreeSurfer eye file")


class InitData(SimpleInterface):
    """Get subject-specific diffusion data"""
    input_spec = _InitDataInputSpec
    output_spec = _InitDataOutputSpec

    def _run_interface(self, runtime):
        sub_dir = self.inputs.config["subject_dir"]
        d_dir = Path(sub_dir, "unprocessed", "Diffusion")
        self._results["d_files"] = {}
        for ndir in self.inputs.config["ndirs"]:
            for phase in self.inputs.config["phases"]:
                for file_type in [".nii.gz", ".bval", ".bvec"]:
                    key = f"dir{ndir}_{phase}{file_type}"
                    self._results["d_files"][key] = Path(d_dir, f"{self.inputs.subject}_dMRI_{key}")

        anat_dir = Path(sub_dir, "T1w")
        self._results["t1_file"] = Path(anat_dir, 'T1w_acpc_dc.nii.gz')
        self._results["t1_restore_file"] = Path(anat_dir, 'T1w_acpc_dc_restore.nii.gz')
        self._results["t1_restore_brain"] = Path(anat_dir, 'T1w_acpc_dc_restore_brain.nii.gz')
        self._results["bias_file"] = Path(anat_dir, 'BiasField_acpc_dc.nii.gz')
        self._results["mask_file"] = Path(anat_dir, 'brainmask_fs.nii.gz')

        fs_dir = Path(anat_dir, self.inputs.config["subject"])
        self._results["eye_file"] = Path(fs_dir, 'mri', 'transforms', 'eye.dat')

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
