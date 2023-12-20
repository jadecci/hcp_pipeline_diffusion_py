import itertools
import subprocess

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import nibabel as nib


class _PickDiffFilesInputSpec(BaseInterfaceInputSpec):
    d_files = traits.Dict(mandatory=True, desc="collection of diffusion files")
    ndir = traits.Str(mandatory=True, desc="Number of directions to pick")
    phase = traits.Str(mandatory=True, desc="Phase encoding direction to pick")


class _PickDiffFilesOutputSpec(TraitedSpec):
    data_file = traits.File(exists=True, desc="Diffusion data files")
    bval_file = traits.File(exists=True, desc="b value files")
    bvec_file = traits.File(exists=True, desc="b vector files")


class PickDiffFiles(SimpleInterface):
    """Pick the diffusion files for one number of directions and phase encoding direction"""
    input_spec = _PickDiffFilesInputSpec
    output_spec = _PickDiffFilesOutputSpec

    def _run_interface(self, runtime):
        key = f"dir{self.inputs.ndir}_{self.inputs.phase}"
        self._results["data_file"] = self.inputs.d_files[f"{key}.nii.gz"]
        self._results["bval_file"] = self.inputs.d_files[f"{key}.bval"]
        self._results["bvec_file"] = self.inputs.d_files[f"{key}.bvec"]
        return runtime


class _UpdateDiffFilesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    d_files = traits.Dict(mandatory=True, desc="Collection of diffusion files")
    data_files = traits.List(mandatory=True, desc="New data files")


class _UpdateDiffFilesOutputSpec(TraitedSpec):
    d_files = traits.Dict(mandatory=True, desc="Collection of diffusion files")


class UpdateDiffFiles(SimpleInterface):
    """Update diffusion files collection with new data files"""
    input_spec = _UpdateDiffFilesInputSpec
    output_spec = _UpdateDiffFilesOutputSpec

    def _run_interface(self, runtime):
        self._results["d_files"] = self.inputs.d_files.copy()
        for key in self.inputs.config["keys"]:
            dwi_key = [
                d_key for d_key in self.inputs.d_files if key in d_key and '.nii.gz' in d_key]
            dwi_replace = [d_file for d_file in self.inputs.dwi_replacements if key in str(d_file)]
            self._results["d_files"][dwi_key[0]] = dwi_replace[0]
        return runtime


class _SplitDiffFilesInputSpec(BaseInterfaceInputSpec):
    d_files = traits.Dict(mandatory=True, desc="Collection of diffusion files")


class _SplitDiffFilesOutputSpec(TraitedSpec):
    data_files = traits.List(exists=True, desc="Diffusion data files")
    bval_files = traits.List(exists=True, desc="b value files")
    bvec_files = traits.List(exists=True, desc="b vector files")


class SplitDiffFiles(SimpleInterface):
    """Split diffusion files collection by file types"""
    input_spec = _SplitDiffFilesInputSpec
    output_spec = _SplitDiffFilesOutputSpec

    def _run_interface(self, runtime):
        self._results["data_files"] = [
            self.inputs.d_files[key] for key in self.inputs.d_files if ".nii.gz" in key]
        self._results["bval_files"] = [
            self.inputs.d_files[key] for key in self.inputs.d_files if ".bval" in key]
        self._results["bvec_files"] = [
            self.inputs.d_files[key] for key in self.inputs.d_files if ".bvec" in key]
        return runtime


class _SplitT1FilesInputSpec(BaseInterfaceInputSpec):
    t1_files = traits.Dict(mandatory=True, desc="Collection of T1 files")


class _SplitT1FilesOutputSpec(TraitedSpec):
    t1_file = traits.File(exists=True, desc="T1 file")
    t1_restore_file = traits.File(exists=True, desc="T1 restored file")
    t1_brain_file = traits.File(exists=True, desc="T1 brain restored file")
    bias_file = traits.File(exists=True, desc="Bias file")
    mask_file = traits.File(exists=True, desc="FreeSurfer mask file")


class SplitT1Files(SimpleInterface):
    """Split T1 files collection by file types"""
    input_spec = _SplitT1FilesInputSpec
    output_spec = _SplitT1FilesOutputSpec

    def _run_interface(self, runtime):
        self._results["t1_file"] = self.inputs.t1_files["t1"]
        self._results["t1_restore_file"] = self.inputs.t1_files["t1_restore"]
        self._results["t1_brain_file"] = self.inputs.t1_files["t1_restore_brain"]
        self._results["bias_file"] = self.inputs.t1_files["bias"]
        self._results["mask_file"] = self.inputs.t1_files["fs_mask"]
        return runtime


class FSEyeInputSpec(BaseInterfaceInputSpec):
    fs_files = traits.Dict(mandatory=True, desc="Collection of FreeSurfer files")


class FSEyeOutputSpec(TraitedSpec):
    eye_file = traits.File(exists=True, desc="FreeSurfer eye file")


class FSEye(SimpleInterface):
    """Get the eye file from FreeSurfer files collection"""
    input_spec = FSEyeInputSpec
    output_spec = FSEyeOutputSpec

    def _run_interface(self, runtime):
        self._results["eye_file"] = self.inputs.fs_files["eye"]
        return runtime


class _FLIRTSchInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(mandatory=True, exists=True, desc="Input file")
    reference = traits.File(mandatory=True, exists=True, desc="Reference file")
    wm_seg = traits.File(exists=True, desc="White matter segmentation volume")
    in_matrix_file = traits.File(exists=True, desc="Input affine matrix")
    schedule = traits.File(desc="Replaces default schedule")
    out_matrix_file = traits.File(desc="Output affine matrix")
    fsl_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _FLIRTSchOutputSpec(TraitedSpec):
    out_matrix_file = traits.File(exists=True, desc="Output affine matrix")


class Flirtsch(SimpleInterface):
    """Run FLIRT with non-default schedule file which may exist in a container"""
    input_spec = _FLIRTSchInputSpec
    output_spec = _FLIRTSchOutputSpec

    def _run_interface(self, runtime):
        subprocess.run(
            self.inputs.fsl_cmd.cmd("flirt").split() + [
                "-in", self.inputs.in_file, "-ref", self.inputs.reference,
                "-wmseg", self.inputs.wm_seg, "-init", self.inputs.in_matrix_file,
                "-omat", self.inputs.out_matrix_file, "-dof", "6", "-cost", "bbr",
                "-schedule", self.inputs.schedule],
            check=True)
        self._results["out_matrix_file"] = self.inputs.out_matrix_file
        return runtime


class _DiffResInputSpec(BaseInterfaceInputSpec):
    data_file = traits.File(mandatory=True, exists=True, desc="Diffusion data file")


class _DiffResOutputSpec(TraitedSpec):
    res = traits.Int(desc="Diffusion data resolution (assuming isotropic)")
    dilate = traits.Int(desc="Dilation range")


class DiffRes(SimpleInterface):
    """Get resolution and dilation range of diffusion data"""
    input_spec = _DiffResInputSpec
    output_spec = _DiffResOutputSpec

    def _run_interface(self, runtime):
        self._results["res"] = nib.load(self.inputs.data_file).header.get_zooms()[0]
        self._results["dilate"] = int(self._results["res"] * 4)
        return runtime


class _ListInputSpec(BaseInterfaceInputSpec):
    input = traits.List(mandatory=True, desc="Input list")


class _ListOutputSpec(TraitedSpec):
    output = traits.List(desc="Output list")


class FlattenList(SimpleInterface):
    """Flatten nested lists"""
    input_spec = _ListInputSpec
    output_spec = _ListOutputSpec

    def _run_interface(self, runtime):
        self._results["output"] = list(itertools.chain.from_iterable(self.inputs.input))
        return runtime


class _CreateListInputSpec(BaseInterfaceInputSpec):
    input1 = traits.Any(mandatory=True, desc="Input item 1")
    input2 = traits.Any(mandatory=True, desc="Input item 2")


class CreateList(SimpleInterface):
    """Flatten nested lists"""
    input_spec = _CreateListInputSpec
    output_spec = _ListOutputSpec

    def _run_interface(self, runtime):
        self._results["output"] = [self.inputs.input1, self.inputs.input2]
        return runtime


class _CombineStringsInputSpec(BaseInterfaceInputSpec):
    input1 = traits.Str(mandatory=True, desc="Input string 1")
    input2 = traits.Str(mandatory=True, desc="Input string 2")
    input3 = traits.Str("", desc="Input string 3")
    input4 = traits.Str("", desc="Input string 4")


class _StringOutputSpec(TraitedSpec):
    output = traits.Str(desc="Output string")


class CombineStrings(SimpleInterface):
    """Combine multile strings into one"""
    input_spec = _CombineStringsInputSpec
    output_spec = _StringOutputSpec

    def _run_interface(self, runtime):
        self._results["output"] = f"{self.inputs.input1}{self.inputs.input2}" \
                                  f"{self.inputs.input3}{self.inputs.input4}"
        return runtime


class _ListItemInputSpec(BaseInterfaceInputSpec):
    input = traits.List(mandatory=True, desc="Input list")
    index = traits.Any(mandatory=True, desc="Index to pick from")


class ListItem(SimpleInterface):
    """Pick a string item from a list by index"""
    input_spec = _ListItemInputSpec
    output_spec = _StringOutputSpec

    def _run_interface(self, runtime):
        self._results["output"] = str(self.inputs.input[self.inputs.index])
        return runtime
