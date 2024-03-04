from pathlib import Path
from typing import Union
import subprocess

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
import pandas as pd
from nipype.interfaces import fsl, workbench
import nibabel as nib


class _ExtractB0InputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    bval_file = traits.File(mandatory=True, desc="b value file")
    data_file = traits.File(mandatory=True, desc="Diffusion data file")
    rescale = traits.Bool(False, desc="if b0dist should be applied on rescaled data")
    fsl_cmd = traits.Any(mandatory=True, desc="FSL command for using singularity image (or not)")


class _ExtractB0OutputSpec(TraitedSpec):
    roi_files = traits.List(dtpye=Path, desc="filenames of B0 files")
    pos_files = traits.List(dtpye=Path, desc="filenames of B0 files with positive phase encoding")
    neg_files = traits.List(dtpye=Path, desc="filenames of B0 files with negative phase encoding")


class ExtractB0(SimpleInterface):
    """Extract b0 slices"""
    input_spec = _ExtractB0InputSpec
    output_spec = _ExtractB0OutputSpec

    def _extract_b0(self, b0dist: Union[int, None] = None) -> list:
        b0maxbval = 50  # values below this will be considered as b0s
        bvals = pd.read_csv(
            self.inputs.bval_file, header=None, delim_whitespace=True).squeeze("rows")

        if b0dist is None:
            dist_count = 0
            roi_files = [self.inputs.data_file]
        else:
            dist_count = b0dist + 1
            roi_files = []
        dim4 = nib.load(self.inputs.data_file).header.get_data_shape()[3]
        vol_count = 0

        for b in bvals:
            roi_file = Path(
                self.inputs.config["work_dir"],
                f"roi{vol_count}_{Path(self.inputs.data_file).name}")
            if b < b0maxbval and b0dist is None:
                roi = fsl.ExtractROI(
                    command=self.inputs.fsl_cmd.cmd("fslroi"),
                    in_file=self.inputs.data_file, t_min=dist_count, t_size=1, roi_file=roi_file)
                roi.run()
                roi_files.append(roi_file)
            elif b < b0maxbval and vol_count < dim4 and dist_count > b0dist:
                roi = fsl.ExtractROI(
                    command=self.inputs.fsl_cmd.cmd("fslroi"),
                    in_file=self.inputs.data_file, t_min=vol_count, t_size=1, roi_file=roi_file)
                roi.run()
                roi_files.append(roi_file)
                dist_count = 0
            dist_count = dist_count + 1
            vol_count = vol_count + 1

        return roi_files

    def _split_pos_neg(self, roi_files: list) -> tuple[list, ...]:
        phases = sorted(self.inputs.config["phases"])
        pos_files = [roi_file for roi_file in roi_files if phases[0] in str(roi_file)]
        neg_files = [roi_file for roi_file in roi_files if phases[1] in str(roi_file)]

        return pos_files, neg_files

    def _run_interface(self, runtime):
        if not self.inputs.rescale:
            self._results["roi_files"] = self._extract_b0()
        else:
            b0dist = 45  # minimum distance between b0s
            self._results["roi_files"] = self._extract_b0(b0dist=b0dist)
            self._results["pos_files"], self._results["neg_files"] = self._split_pos_neg(
                self._results["roi_files"])

        return runtime


class _RescaleInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    scale_files = traits.List(mandatory=True, dtype=str, desc="filenames of scale files")
    d_files = traits.Dict(mandatory=True, dtype=Path, desc="filenames of diffusion data")
    fsl_cmd = traits.Any(mandatory=True, desc="FSL command for using singularity image (or not)")


class _RescaleOutputSpec(TraitedSpec):
    rescaled_files = traits.List(dtype=str, desc="filenames of rescaled DWI images")


class Rescale(SimpleInterface):
    """Rescale DWI images, except the first one"""
    input_spec = _RescaleInputSpec
    output_spec = _RescaleOutputSpec

    def _run_interface(self, runtime):
        keys = self.inputs.config["keys"].copy()
        key_first = keys.pop()

        rescale_file = [s_file for s_file in self.inputs.scale_files if key_first in s_file]
        rescale = pd.read_csv(rescale_file[0], header=None).squeeze()
        self._results["rescaled_files"] = [
            self.inputs.d_files[d_key] for d_key in self.inputs.d_files if key_first in d_key]

        for key in keys:
            scale_file = [s_file for s_file in self.inputs.scale_files if key in s_file]
            scale = pd.read_csv(scale_file[0], header=None).squeeze()
            d_file = [self.inputs.d_files[d_key] for d_key in self.inputs.d_files if key in d_key]
            maths = fsl.ImageMaths(
                command=self.inputs.fsl_cmd.cmd("fslmaths"),
                in_file=d_file[0], args=f"-mul {rescale} -div {scale}")
            maths.run()
            self._results["rescaled_files"].append(maths.aggregate_outputs().out_file)

        return runtime


class _PrepareTopupInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    roi_files = traits.List(mandatory=True, dtpye=Path, desc="filenames of B0 files")
    d_files = traits.Dict(mandatory=True, dtype=Path, desc="filenames of diffusion data")
    pos_b0_file = traits.File(mandatory=True, desc="merged positive b0 file")


class _PrepareTopupOutputSpec(TraitedSpec):
    enc_dir = traits.List(dtype=str, desc="encoding directions for each b0")
    ro_time = traits.Float(desc="readout time")
    indices_t = traits.List(dtype=int, desc="indices based on time dimension of b0 files")


class PrepareTopup(SimpleInterface):
    """Prepare parameters for FSL Topup"""
    input_spec = _PrepareTopupInputSpec
    output_spec = _PrepareTopupOutputSpec

    def _run_interface(self, runtime):
        # encoding direction
        self._results['enc_dir'] = []
        phases = sorted(self.inputs.config["phases"])
        for roi_file in self.inputs.roi_files:
            if phases[0] in str(roi_file):
                self._results['enc_dir'].append('y')
            elif phases[1] in str(roi_file):
                self._results['enc_dir'].append('y-')

        # readout time
        key = f"{sorted(self.inputs.config['keys'])[0]}.nii.gz"
        dim_p = nib.load(self.inputs.d_files[key]).header.get_data_shape()[1]
        self._results['ro_time'] = round(self.inputs.config["echo_spacing"] * (dim_p - 1) / 1000, 6)

        # time dimension
        self._results['indices_t'] = [
            1, nib.load(self.inputs.pos_b0_file).header.get_data_shape()[3] + 1]

        return runtime


class _MergeBFilesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    bval_files = traits.List(mandatory=True, dtype=Path, desc="list of bval files to merge")
    bvec_files = traits.List(mandatory=True, dtype=Path, desc="list of bvec files to merge")


class _MergeBFilesOutputSpec(TraitedSpec):
    bval_merged = traits.File(exists=True, desc="merged bval file")
    bvec_merged = traits.File(exists=True, desc="merged bvec file")


class MergeBFiles(SimpleInterface):
    """Merge bval and bvec files respectively"""
    input_spec = _MergeBFilesInputSpec
    output_spec = _MergeBFilesOutputSpec

    def _run_interface(self, runtime):
        bvals = pd.DataFrame()
        bvecs = pd.DataFrame()

        for key in self.inputs.config["keys"]:
            bval_file = [b_file for b_file in self.inputs.bval_files if key in str(b_file)]
            bvec_file = [b_file for b_file in self.inputs.bvec_files if key in str(b_file)]
            bvals = pd.concat(
                [bvals, pd.read_csv(bval_file[0], delim_whitespace=True, header=None)], axis=1)
            bvecs = pd.concat(
                [bvecs, pd.read_csv(bvec_file[0], delim_whitespace=True, header=None)], axis=1)

        self._results["bval_merged"] = Path(self.inputs.config["work_dir"], "merged.bval")
        self._results["bvec_merged"] = Path(self.inputs.config["work_dir"], "merged.bvec")
        bvals.to_csv(self._results["bval_merged"], sep='\t', header=False, index=False)
        bvecs.to_csv(self._results["bvec_merged"], sep='\t', header=False, index=False)

        return runtime


class _EddyIndexInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    roi_files = traits.List(mandatory=True, dtype=Path, desc="filenames of B0 files")
    dwi_files = traits.List(mandatory=True, dtype=Path, desc="filenames of rescaled DWI images")


class _EddyIndexOutputSpec(TraitedSpec):
    index_file = traits.File(exists=True, desc="filename of index file")


class EddyIndex(SimpleInterface):
    """Create index file for eddy correction"""
    input_spec = _EddyIndexInputSpec
    output_spec = _EddyIndexOutputSpec

    def _run_interface(self, runtime):
        rois = [
            int(str(roi_file.name).lstrip('roi').split('_')[0]) for roi_file in
            self.inputs.roi_files]
        unique_rois = np.sort(np.unique(rois))
        phases = sorted(self.inputs.config["phases"])

        indices = []
        pos_count = 0
        neg_count = 0
        vol_prev = 0
        for roi_file in self.inputs.roi_files:
            key = [k for k in self.inputs.config["keys"] if k in str(roi_file)]
            dwi_file = [d_file for d_file in self.inputs.dwi_files if key[0] in str(d_file)]
            dim4 = nib.load(dwi_file[0]).header.get_data_shape()[3]

            vol_curr = int(str(roi_file.name).lstrip("roi").split('_')[0])
            for _ in range(vol_prev, vol_curr):
                if phases[0] in str(roi_file):
                    indices.append(pos_count)
                elif phases[1] in str(roi_file):
                    indices.append(neg_count)

            if phases[0] in str(roi_file):
                pos_count = pos_count + 1
            elif phases[1] in str(roi_file):
                neg_count = neg_count + 1

            if vol_curr == unique_rois[-1]:
                for _ in range(vol_curr, dim4):
                    if phases[0] in str(roi_file):
                        indices.append(pos_count)
                    elif phases[1] in str(roi_file):
                        indices.append(neg_count)
            vol_prev = vol_curr

        self._results["index_file"] = Path(self.inputs.config["work_dir"], "index.txt")
        pd.DataFrame(indices).to_csv(
            self._results["index_file"], sep="\t", header=False, index=False)

        return runtime


class _EddyPostProcInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    bval_files = traits.List(mandatory=True, dtype=Path, desc="list of bval files to merge")
    bvec_files = traits.List(mandatory=True, dtype=Path, desc="list of bvec files to merge")
    eddy_corrected_file = traits.File(
        mandatory=True, exists=True, desc="filename of eddy corrected image")
    eddy_bvecs_file = traits.File(
        mandatory=True, exists=True, desc="filename of eddy corrected bvecs")
    rescaled_files = traits.List(
        mandatory=True, dtype=Path, desc="filenames of rescaled DWI images")
    fsl_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _EddyPostProcOutputSpec(TraitedSpec):
    combined_dwi_file = traits.File(exists=True, desc="combined DWI data")
    rot_bvals = traits.File(exists=True, desc="average rotated bvals")
    rot_bvecs = traits.File(exists=True, desc="average rotated bvecs")


class EddyPostProc(SimpleInterface):
    """Post-eddy processing: combine output files and rotate bval/bvec from eddy correction"""
    input_spec = _EddyPostProcInputSpec
    output_spec = _EddyPostProcOutputSpec

    def _generate_files(
            self, keys: list, dirs: str) -> tuple[Path, Path, Path, Path, pd.DataFrame, list]:
        bvals = pd.DataFrame()
        bvecs = pd.DataFrame()
        corrvols = []
        tsizes = []
        for key in keys:
            bval_file = [b_file for b_file in self.inputs.bval_files if key in str(b_file)]
            bvec_file = [b_file for b_file in self.inputs.bvec_files if key in str(b_file)]
            bval = pd.read_csv(bval_file[0], delim_whitespace=True, header=None)
            bvals = pd.concat([bvals, bval], axis=1)
            bvecs = pd.concat(
                [bvecs, pd.read_csv(bvec_file[0], delim_whitespace=True, header=None)], axis=1)

            rescaled_file = [d_file for d_file in self.inputs.rescaled_files if key in str(d_file)]
            dim4 = nib.load(rescaled_file[0]).header.get_data_shape()[3]
            corrvols.append([dim4, dim4])
            tsizes.append(bval.shape[1])

        bval_merged = Path(self.inputs.config["work_dir"], f"{dirs}.bval")
        bvals.to_csv(bval_merged, sep='\t', header=False, index=False)
        bvec_merged = Path(self.inputs.config["work_dir"], f"{dirs}.bvec")
        bvecs.to_csv(bvec_merged, sep='\t', header=False, index=False)
        corrvols_file = Path(self.inputs.config["work_dir"], f"{dirs}_volnum.txt")
        pd.DataFrame(corrvols).to_csv(corrvols_file, sep='\t', header=False, index=False)

        bval_tsize = bvals.shape[1]
        roi_file = Path(self.inputs.config["work_dir"], f"{dirs}.nii.gz")
        extract_roi = fsl.ExtractROI(
            command=self.inputs.fsl_cmd.cmd("fslroi"), in_file=self.inputs.eddy_corrected_file,
            t_size=bval_tsize, roi_file=roi_file)
        if dirs == "pos":
            extract_roi.inputs.t_min = 0
        elif dirs == "neg":
            extract_roi.inputs.t_min = bval_tsize
        extract_roi.run()

        return roi_file, bval_merged, bvec_merged, corrvols_file, bvals, tsizes

    def _rotate_b(
            self, pos_tsize: list, neg_tsize: list, pos_bvals: pd.DataFrame,
            neg_bvals: pd.DataFrame) -> None:
        rot_bvecs = pd.read_csv(self.inputs.eddy_bvecs_file, delim_whitespace=True, header=None)
        pos_rot_bvecs = np.zeros((3, sum(pos_tsize)))
        neg_rot_bvecs = np.zeros((3, sum(neg_tsize)))
        break_pos = [pos_tsize[0], pos_tsize[0] + neg_tsize[0], sum(pos_tsize) + neg_tsize[0]]
        pos_rot_bvecs[:, :pos_tsize[0]] = rot_bvecs.iloc[:, :break_pos[0]]
        neg_rot_bvecs[:, :neg_tsize[0]] = rot_bvecs.iloc[:, break_pos[0]:break_pos[1]]
        pos_rot_bvecs[:, pos_tsize[0]:] = rot_bvecs.iloc[:, break_pos[1]:break_pos[2]]
        neg_rot_bvecs[:, neg_tsize[0]:] = rot_bvecs.iloc[:, break_pos[2]:]

        avg_bvals = np.zeros((sum(pos_tsize)), dtype="i4")
        avg_bvecs = np.zeros((3, sum(pos_tsize)))
        for i in range(sum(pos_tsize)):
            pos_bvec = np.array(
                pos_bvals.iloc[:, i]) * np.array(pos_rot_bvecs[:, i]).reshape((3, 1))
            neg_bvec = np.array(
                neg_bvals.iloc[:, i]) * np.array(neg_rot_bvecs[:, i]).reshape((3, 1))
            bvec_sum = (np.dot(pos_bvec, pos_bvec.T) + np.dot(neg_bvec, neg_bvec.T)) / 2
            eigvals, eigvecs = np.linalg.eig(bvec_sum)
            eigvalmax = np.argmax(eigvals)
            avg_bvals[i] = np.rint(eigvals[eigvalmax] ** 0.5)
            avg_bvecs[:, i] = eigvecs[:, eigvalmax]

        self._results["rot_bvals"] = Path(self.inputs.config["work_dir"], "rotated.bval")
        self._results["rot_bvecs"] = Path(self.inputs.config["work_dir"], "rotated.bvec")
        pd.DataFrame(avg_bvals).T.to_csv(
            self._results["rot_bvals"], sep=' ', header=False, index=False)
        pd.DataFrame(avg_bvecs).to_csv(
            self._results["rot_bvecs"], sep=' ', header=False, index=False, float_format="%0.16f")

    def _run_interface(self, runtime):
        phases = sorted(self.inputs.config["phases"])
        pos_keys = [key for key in self.inputs.config["keys"] if phases[0] in key]
        neg_keys = [key for key in self.inputs.config["keys"] if phases[1] in key]
        pos_dwi, pos_bval, pos_bvec, pos_corrvols, pos_bvals, pos_tsize = self._generate_files(
            pos_keys, "pos")
        neg_dwi, neg_bval, neg_bvec, neg_corrvols, neg_bvals, neg_tsize = self._generate_files(
            neg_keys, "neg")

        subprocess.run(
            self.inputs.fsl_cmd.cmd("eddy_combine").split() + [
                pos_dwi, pos_bval, pos_bvec, pos_corrvols, neg_dwi, neg_bval, neg_bvec,
                neg_corrvols, self.inputs.config["work_dir"], "1"],
            check=True)
        self._results["combined_dwi_file"] = Path(self.inputs.config["work_dir"], "data.nii.gz")
        self._rotate_b(pos_tsize, neg_tsize, pos_bvals, neg_bvals)

        return runtime


class _WBDilateInputSpec(BaseInterfaceInputSpec):
    data_file = traits.File(mandatory=True, exists=True, desc="filename of input data")
    dilate = traits.Int(mandatory=True, desc="dilate resolution")
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    wb_cmd = traits.Any(mandatory=True, desc="Workbench command for using container (or not)")


class _WBDilateOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc="dilated data output")


class WBDilate(SimpleInterface):
    """Dilate DWI data"""
    input_spec = _WBDilateInputSpec
    output_spec = _WBDilateOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = Path(self.inputs.config["work_dir"], "data_dilated.nii.gz")
        args = (
            f"-volume-dilate {self.inputs.data_file} {self.inputs.dilate} NEAREST "
            f"{self._results['out_file']}")
        wb = workbench.base.WBCommand(command=self.inputs.wb_cmd.cmd("wb_command"), args=args)
        wb.run()

        return runtime


class _DilateMaskInputSpec(BaseInterfaceInputSpec):
    mask_file = traits.File(mandatory=True, exists=True, desc="filename of input mask")
    fsl_cmd = traits.Any(mandatory=True, desc="FSL command for using container (or not)")


class _DilateMaskOutputSpec(TraitedSpec):
    dil0_file = traits.File(exists=True, desc="mask dilated once")
    out_file = traits.File(exists=True, desc="dilated mask output")


class DilateMask(SimpleInterface):
    """Dilate mask image 7 times"""
    input_spec = _DilateMaskInputSpec
    output_spec = _DilateMaskOutputSpec

    def _run_interface(self, runtime):
        args = "-kernel 3D -dilM"
        resamp_start = fsl.ImageMaths(
            command=self.inputs.fsl_cmd.cmd("fslmaths"), in_file=self.inputs.mask_file, args=args)
        resamp_start.run()
        self._results["dil0_file"] = resamp_start.aggregate_outputs().out_file
        mask_prev = self._results["dil0_file"]

        for _ in range(6):
            resamp_curr = fsl.ImageMaths(
                command=self.inputs.fsl_cmd.run_cmd("fslmaths"), in_file=mask_prev, args=args)
            resamp_curr.run()
            mask_prev = resamp_curr.aggregate_outputs().out_file
        self._results["out_file"] = mask_prev

        return runtime


class _RotateBVec2StrInputSpec(BaseInterfaceInputSpec):
    bvecs_file = traits.File(mandatory=True, exists=True, desc="filename of input (merged) bvecs")
    rot = traits.List(dtype=list, mandatory=True, desc="rotation matrix of diff2str warp")
    config = traits.Dict(mandatory=True, desc="Workflow configurations")


class _RotateBVec2StrOutputSpec(TraitedSpec):
    rotated_file = traits.File(exists=True, desc="rotated bvecs")


class RotateBVec2Str(SimpleInterface):
    """Rotate bvecs based on diffusion-to-structural warp"""
    input_spec = _RotateBVec2StrInputSpec
    output_spec = _RotateBVec2StrOutputSpec

    def _run_interface(self, runtime):
        bvecs = pd.read_csv(self.inputs.bvecs_file, delim_whitespace=True, header=None)
        rotated_bvecs = np.matmul(np.array(self.inputs.rot)[:3, :3], bvecs)
        self._results["rotated_file"] = Path(self.inputs.config["work_dir"], "rotated2str.bvec")
        pd.DataFrame(rotated_bvecs).to_csv(
            self._results["rotated_file"], sep=' ', header=False, index=False,
            float_format="%10.6f", quoting=3, escapechar=' ')

        return runtime
