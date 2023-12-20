from pathlib import Path
import argparse

from nipype.interfaces import fsl, freesurfer
import nipype.pipeline as pe

from hcpdiffpy.container import SimgCmd
from hcpdiffpy.interfaces.data import InitData, SaveData
from hcpdiffpy.interfaces.preproc import (
    EddyIndex, EddyPostProc, ExtractB0, DilateMask, MergeBFiles, Rescale, RotateBVec2Str,
    PrepareTopup, WBDilate)
from hcpdiffpy.interfaces.utilities import (
    CreateList, CombineStrings, DiffRes, FlattenList, ListItem, PickDiffFiles, SplitDiffFiles,
    UpdateDiffFiles)

base_dir = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HCP Pipeline for diffusion preprocessing",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument(
        "subject_dir", type=Path,
        help="Absolute path to the subject's data folder (organised in HCP-like structure)")
    parser.add_argument("subject", type=str, help="Subject ID")
    parser.add_argument("ndirs", nargs="+", help="List of numbers of directions")
    parser.add_argument("phases", nargs="+", help="List of 2 phase encoding directions")
    parser.add_argument("echo_spacing", type=float, help="Echo spacing used for acquisition in ms")
    parser.add_argument(
        "--workdir", type=Path, dest="work_dir", default=Path.cwd(),
        help="Absolute path to work directory")
    parser.add_argument(
        "--output_dir", type=Path, dest="output_dir", default=Path.cwd(),
        help="Absolute path to output directory")
    parser.add_argument(
        "--fsl_simg", type=Path, dest="fsl_simg", default=None,
        help="singularity image to use for command line functions from FSL")
    parser.add_argument(
        "--fs_simg", type=Path, dest="fs_simg", default=None,
        help="singularity image to use for command line functions from FreeSurfer")
    parser.add_argument(
        "--wb_simg", type=Path, dest="wb_simg", default=None,
        help="singularity image to use for command line functions from Connectome Workbench")
    parser.add_argument(
        "--condordag", dest="condordag", action="store_true",
        help="Submit workflow as DAG to HTCondor")
    config = vars(parser.parse_args())

    # Set-up
    config["tmp_dir"] = Path(config["work_dir"], "hcp_proc_tmp")
    config["tmp_dir"].mkdir(parents=True, exist_ok=True)
    config["keys"] = [
        f"dir{ndir}_{phase}" for ndir in sorted(config["ndirs"])
        for phase in sorted(config["phases"])]
    d_iterables = [("ndir", config["ndirs"]), ("phases", config["phases"])]
    fsl_cmd = SimgCmd(config, config["fsl_simg"])
    fs_cmd = SimgCmd(config, config["fs_simg"])
    wb_cmd = SimgCmd(config, config["wb_simg"])
    topup_config = Path(base_dir, "utilities", "b02b0.cnf")
    sch_file = Path(base_dir, "utilities", "bbr.sch")
    fs_dir = Path(config["subject_dir"], "T1w", config["subject"])

    # Workflow set-up
    hcpdiff_wf = pe.Workflow(f"hcpdiff_{config['subject']}_wf", base_dir=config["work_dir"])
    hcpdiff_wf.config["execution"]["try_hard_link_datasink"] = "false"
    hcpdiff_wf.config["execution"]["crashfile_format"] = "txt"
    hcpdiff_wf.config["execution"]["stop_on_first_crash"] = "true"

    # Get data
    init_data = pe.Node(InitData(config=config), "init_data")
    d_files = pe.Node(PickDiffFiles(), "split_diff_files", ietrables=d_iterables)

    hcpdiff_wf.connect([(init_data, d_files, [("d_files", "d_files")])])

    # 1. PreEddy
    # 1.1. Normalise intensity
    mean_dwi = pe.Node(
        fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-Xmean -Ymean -Zmean"), "mean_dwi")
    extract_b0s = pe.Node(ExtractB0(fsl_cmd=fsl_cmd, config=config), "extract_b0s")
    merge_b0s = pe.Node(fsl.Merge(command=fsl_cmd.cmd("fslmerge"), dimension="t"), "merge_b0s")
    mean_b0 = pe.Node(fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-Tmean"), "mean_b0")
    scale = pe.Node(fsl.ImageMeants(command=fsl_cmd.cmd("fslmeants")), "scale")
    rescale = pe.JoinNode(
        Rescale(fsl_cmd=fsl_cmd, config=config), "rescale",
        joinfield=["scale_files"], joinsource="split_diff_files")

    hcpdiff_wf.connect([
        (d_files, mean_dwi, [("data_file", "in_file")]),
        (d_files, extract_b0s, [("bval_file", "bval_file")]),
        (mean_dwi, extract_b0s, [("out_file", "data_file")]),
        (extract_b0s, merge_b0s, [("roi_files", "in_files")]),
        (merge_b0s, mean_b0, [("merged_file", "in_file")]),
        (mean_b0, scale, [("out_file", "in_file")]),
        (init_data, rescale, [("d_files", "d_files")]),
        (scale, rescale, [("out_file", "scale_file")])])

    # 1.2. Prepare b0s and index files for topup
    update_d_files = pe.Node(UpdateDiffFiles(config=config), "update_d_files")
    rescaled_d_files = pe.Node(PickDiffFiles(), "split_rescaled", iterables=d_iterables)
    rescaled_b0s = pe.Node(ExtractB0(config=config, rescale=True, fsl_cmd=fsl_cmd), "rescaled_b0s")
    b0_list = pe.JoinNode(FlattenList(), "b0_list", joinfield="input", joinsource="split_rescaled")
    pos_b0_list = pe.JoinNode(
        FlattenList(), "pos_b0_list", joinfield="input", joinsource="split_rescaled")
    neg_b0_list = pe.JoinNode(
        FlattenList(), "neg_b0_list", joinfield="input", joinsource="split_rescaled")
    merge_rescaled_b0s = pe.Node(
        fsl.Merge(command=fsl_cmd.cmd("fslmerge"), dimension="t"), "merge_rescaled_b0s")
    merge_pos_b0s = pe.Node(
        fsl.Merge(command=fsl_cmd.cmd("fslmerge"), dimension="t"), "merge_pos_b0s")
    merge_neg_b0s = pe.Node(
        fsl.Merge(command=fsl_cmd.cmd("fslmerge"), dimension="t"), "merge_neg_b0s")

    hcpdiff_wf.connect([
        (init_data, update_d_files, [("d_files", "d_files")]),
        (rescale, update_d_files, [("rescaled_files", "data_files")]),
        (update_d_files, rescaled_d_files, [("d_files", "d_files")]),
        (rescaled_d_files, rescaled_b0s, [("bval_file", "bval_file"), ("data_file", "data_file")]),
        (rescaled_b0s, b0_list, [("roi_files", "input")]),
        (rescaled_b0s, pos_b0_list, [("pos_files", "input")]),
        (rescaled_b0s, neg_b0_list, [("neg_files", "input")]),
        (b0_list, merge_rescaled_b0s, [("output", "in_files")]),
        (pos_b0_list, merge_pos_b0s, [("output", "in_files")]),
        (neg_b0_list, merge_neg_b0s, [("output", "in_files")])])

    # 1.3. Topup
    prepare_topup = pe.Node(PrepareTopup(config=config), "prepare_topup")
    topup = pe.Node(fsl.TOPUP(command=fsl_cmd.cmd("topup"), config=str(topup_config)), "topup")
    pos_b01 = pe.Node(fsl.ExtractROI(command=fsl_cmd.cmd("fslroi"), t_min=0, t_size=1), "pos_b01")
    neg_b01 = pe.Node(fsl.ExtractROI(command=fsl_cmd.cmd("fslroi"), t_min=0, t_size=1), "neg_b01")
    b01_files = pe.Node(CreateList(), "b01_files")
    apply_topup = pe.Node(
        fsl.ApplyTOPUP(command=fsl_cmd.cmd("applytopup"), method="jac"), "apply_topup")
    nodiff_mask = pe.Node(fsl.BET(command=fsl_cmd.cmd("bet"), frac=0.2, mask=True), "nodiff_mask")

    hcpdiff_wf.connect([
        (update_d_files, prepare_topup, [("d_files", "d_files")]),
        (b0_list, prepare_topup, [("output", "roi_files")]),
        (merge_pos_b0s, prepare_topup, [("merged_file", "pos_b0_file")]),
        (merge_rescaled_b0s, topup, [("merged_file", "in_file")]),
        (prepare_topup, topup, [("enc_dir", "encoding_direction"), ("ro_time", "readout_times")]),
        (merge_pos_b0s, pos_b01, [("merged_file", "in_file")]),
        (merge_neg_b0s, neg_b01, [("merged_file", "in_file")]),
        (pos_b01, b01_files, [("roi_file", "input1")]),
        (neg_b01, b01_files, [("roi_file", "input2")]),
        (prepare_topup, apply_topup, [("indices_t", "in_index")]),
        (topup, apply_topup, [
            ("out_enc_file", "encoding_file"), ("out_fieldcoef", "in_topup_fieldcoef"),
            ("out_movpar", "in_topup_movpar")]),
        (b01_files, apply_topup, [("output", "in_files")]),
        (apply_topup, nodiff_mask, [("out_corrected", "in_file")])])

    # 2. Eddy
    d_filetype = pe.Node(SplitDiffFiles(), "split_d_filetype")
    merge_bfiles = pe.Node(MergeBFiles(config=config), "merge_bfiles")
    merge_dwi = pe.Node(fsl.Merge(command=fsl_cmd.cmd("fslmerge"), dimension="t"), "merge_dwi")
    eddy_index = pe.Node(EddyIndex(config=config), "eddy_index")
    eddy = pe.Node(fsl.Eddy(command=fsl_cmd.cmd("eddy"), fwhm=0, args="-v"), name="eddy")

    hcpdiff_wf.connect([
        (update_d_files, d_filetype, [("d_files", "d_files")]),
        (d_filetype, merge_bfiles, [
            ("bval_files", "bval_files"), ("bvec_files", "bvec_files")]),
        (d_filetype, merge_dwi, [("data_files", "in_files")]),
        (b0_list, eddy_index, [("output", "roi_files")]),
        (d_filetype, eddy_index, [("data_files", "dwi_files")]),
        (merge_dwi, eddy, [("merged_file", "in_file")]),
        (merge_bfiles, eddy, [("bval_merged", "in_bval"), ("bvec_merged", "in_bvec")]),
        (topup, eddy, [("out_enc_file", "in_acqp")]),
        (eddy_index, eddy, [("index_file", "in_index")]),
        (nodiff_mask, eddy, [("mask_file", "in_mask")]),
        (topup, eddy, [
            ("out_fieldcoef", "in_topup_fieldcoef"), ("out_movpar", "in_topup_movpar")])])

    # 3. PostEddy
    # 3.1. Postproc
    postproc = pe.Node(EddyPostProc(fsl_cmd=fsl_cmd, config=config), "postproc")
    fov_mask = pe.Node(
        fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-abs -Tmin -bin -fillh"), "fov_mask")
    mask_args = pe.Node(CombineStrings(input1="-mas "), "mask_args")
    mask_data = pe.Node(fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths")), "mask_data")
    thr_data = pe.Node(fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-thr 0"), "thr_data")

    hcpdiff_wf.connect([
        (d_filetype, postproc, [
            ("bval_files", "bval_files"), ("bvec_files", "bvec_files"),
            ("data_files", "rescaled_files")]),
        (postproc, fov_mask, [("combined_dwi_file", "in_file")]),
        (fov_mask, mask_args, [("out_file", "input2")]),
        (postproc, mask_data, [("combined_dwi_file", "in_file")]),
        (mask_args, mask_data, [("output", "args")]),
        (mask_data, thr_data, [("out_file", "in_file")])])

    # 3.2. DiffusionToStructural
    # 3.2.1. nodiff-to-T1
    nodiff_brain = pe.Node(
        fsl.ExtractROI(command=fsl_cmd.cmd("fslroi"), t_min=0, t_size=1), "nodiff_brain")
    wm_seg = pe.Node(fsl.FAST(command=fsl_cmd.cmd("fast"), output_type="NIFTI_GZ"), "wm_seg")
    pve_file = pe.Node(ListItem(index=-1), "pve_file")
    wm_thr = pe.Node(
        fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-thr 0.5 -bin"), "wm_thr")
    flirt_init = pe.Node(fsl.FLIRT(command=fsl_cmd.cmd("flirt"), dof=6), "flirt_init")
    flirt_nodiff2t1 = pe.Node(
        fsl.FLIRT(command=fsl_cmd.cmd("flirt"), dof=6, cost="bbr", schedule=sch_file),
        "flirt_nodiff2t1")
    nodiff2t1 = pe.Node(
        fsl.ApplyWarp(command=fsl_cmd.cmd("applywarp"), interp="spline", relwarp=True), "nodiff2t1")
    bias_args = pe.Node(CombineStrings(input1="-div "), "bias_args")
    nodiff_bias = pe.Node(fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths")), "nodiff_bias")

    hcpdiff_wf.connect([
        (thr_data, nodiff_brain, [("out_file", "in_file")]),
        (init_data, wm_seg, [("t1_brain_file", "in_files")]),
        (wm_seg, pve_file, [("partial_volume_files", "input")]),
        (pve_file, wm_thr, [("output", "in_file")]),
        (nodiff_brain, flirt_init, [("roi_file", "in_file")]),
        (init_data, flirt_init, [("t1_brain_file", "reference")]),
        (nodiff_brain, flirt_nodiff2t1, [("roi_file", "in_file")]),
        (init_data, flirt_nodiff2t1, [("t1_file", "reference")]),
        (wm_thr, flirt_nodiff2t1, [("out_file", "wm_seg")]),
        (flirt_init, flirt_nodiff2t1, [("out_matrix_file", "in_matrix_file")]),
        (nodiff_brain, nodiff2t1, [("roi_file", "in_file")]),
        (init_data, nodiff2t1, [("t1_file", "ref_file")]),
        (flirt_nodiff2t1, nodiff2t1, [("out_matrix_file", "premat")]),
        (init_data, bias_args, [("bias_file", "input2")]),
        (nodiff2t1, nodiff_bias, [("out_file", "in_file")]),
        (bias_args, nodiff_bias, [("output", "args")])])

    # 3.2.2. diff-to-struct
    bbr_epi2t1 = pe.Node(
        freesurfer.BBRegister(
            command=fs_cmd.cmd("bbregister", options=f"--env SUBJECTS_DIR={fs_dir}"),
            contrast_type="bold", dof=6, args="--surf white.deformed", subjects_dir=fs_dir,
            subject_id=config["subject"]),
        "bbr_epi2t1")
    tkr_diff2str = pe.Node(
        freesurfer.Tkregister2(command=fs_cmd.cmd("tkregister2"), noedit=True), "tkr_diff2str")
    diff2str = pe.Node(
        fsl.ConvertXFM(command=fsl_cmd.cmd("convert_xfm"), concat_xfm=True), "diff2str")

    hcpdiff_wf.connect([
        (nodiff_bias, bbr_epi2t1, [("out_file", "source_file")]),
        (init_data, bbr_epi2t1, [("eye_file", "init_reg_file")]),
        (nodiff_bias, tkr_diff2str, [("out_file", "moving_image")]),
        (bbr_epi2t1, tkr_diff2str, [("out_reg_file", "reg_file")]),
        (init_data, tkr_diff2str, [("t1_file", "target_image")]),
        (flirt_nodiff2t1, diff2str, [("out_matrix_file", "in_file")]),
        (tkr_diff2str, diff2str, [("fsl_file", "in_file2")])])

    # 3.2.3. resampling
    res_dil = pe.Node(DiffRes(), "res_dil")
    flirt_resamp = pe.Node(fsl.FLIRT(command=fsl_cmd.cmd("flirt")), "flirt_resamp")
    t1_resamp = pe.Node(
        fsl.ApplyWarp(command=fsl_cmd.cmd("applywarp"), interp="spline", relwarp=True), "t1_resamp")
    dilate_data = pe.Node(WBDilate(config=config, wb_cmd=wb_cmd), "dilate_data")
    resamp_data = pe.Node(
        fsl.FLIRT(command=fsl_cmd.cmd("flirt"), apply_xfm=True, interp="spline"), "resamp_data")
    resamp_mask = pe.Node(
        fsl.FLIRT(command=fsl_cmd.cmd("flirt"), interp="nearestneighbour"), "resamp_mask")
    resamp_fmask = pe.Node(
        fsl.FLIRT(command=fsl_cmd.cmd("flirt"), apply_xfm=True, interp="trilinear"), "resamp_fmask")

    hcpdiff_wf.connect([
        (thr_data, res_dil, [("out_file", "data_file")]),
        (init_data, flirt_resamp, [
            ("t1_restore_file", "in_file"), ("t1_restore_file", "reference")]),
        (res_dil, flirt_resamp, [("res", "apply_isoxfm")]),
        (init_data, t1_resamp, [("t1_restore_file", "in_file")]),
        (flirt_resamp, t1_resamp, [("out_file", "ref_file")]),
        (thr_data, dilate_data, [("out_file", "data_file")]),
        (res_dil, dilate_data, [("dilate", "dilate")]),
        (dilate_data, resamp_data, [("out_file", "in_file")]),
        (t1_resamp, resamp_data, [("out_file", "reference")]),
        (diff2str, resamp_data, [("out_file", "in_matrix_file")]),
        (init_data, resamp_mask, [("mask_file", "in_file"), ("mask_file", "reference")]),
        (res_dil, resamp_mask, [("res", "apply_isoxfm")]),
        (fov_mask, resamp_fmask, [("out_file", "in_file")]),
        (t1_resamp, resamp_fmask, [("out_file", "reference")]),
        (diff2str, resamp_fmask, [("out_file", "in_matrix_file")])])

    # 3.2.4. postprocessing
    dilate_mask = pe.Node(DilateMask(fsl_cmd=fsl_cmd), "dilate_mask")
    thr_fmask = pe.Node(
        fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-thr 0.999 -bin"), "thr_fmask")
    masks_args = pe.Node(CombineStrings(input1="-mas ", input3="-mas "), "masks_args")
    mask_data = pe.Node(fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths")), "fmask_data")
    nonneg_data = pe.Node(
        fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-thr 0"), "nonneg_data")
    mean_mask = pe.Node(fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths"), args="-Tmean"), "mean_mask")
    mean_args = pe.Node(CombineStrings(input1="-mas "), "mean_args")
    mask_mask = pe.Node(fsl.ImageMaths(command=fsl_cmd.cmd("fslmaths")), "mask_mask")
    rot_matrix = pe.Node(fsl.AvScale(command=fsl_cmd.cmd("avscale")), "rot_matrix")
    rotate_bvec = pe.Node(RotateBVec2Str(config=config), "rotate_bvec")

    hcpdiff_wf.connect([
        (resamp_mask, dilate_mask, [("out_file", "mask_file")]),
        (resamp_fmask, thr_fmask, [("out_file", "in_file")]),
        (dilate_mask, masks_args, [("out_file", "input2")]),
        (thr_fmask, masks_args, [("out_file", "input4")]),
        (resamp_data, mask_data, [("out_file", "in_file")]),
        (masks_args, mask_data, [("output", "args")]),
        (mask_data, nonneg_data, [("out_file", "in_file")]),
        (nonneg_data, mean_mask, [("out_file", "in_file")]),
        (mean_mask, mean_args, [("out_file", "input2")]),
        (dilate_mask, mask_mask, [("dil0_file", "in_file")]),
        (mean_args, mask_mask, [("output", "args")]),
        (diff2str, rot_matrix, [("out_file", "mat_file")]),
        (postproc, rotate_bvec, [("rot_bvecs", "bvecs_file")]),
        (rot_matrix, rotate_bvec, [("rotation_translation_matrix", "rot")])])

    # Save data
    save_data = pe.Node(SaveData(config=config), "save_data")

    hcpdiff_wf.connect([
        (postproc, save_data, [("rot_bvals", "bval_file")]),
        (nonneg_data, save_data, [("out_file", "data_file")]),
        (mask_mask, save_data, [("out_file", "mask_file")]),
        (rotate_bvec, save_data, [("rotated_file", "bvec_file")])])

    # Run workflow
    hcpdiff_wf.write_graph()
    if config["condordag"]:
        hcpdiff_wf.run(
            plugin="CondorDAGMan",
            plugin_args={
                "dagman_args": f"-outfile_dir {config['work_dir']} -import_env",
                "wrapper_cmd": Path(base_dir, "utilities", "venv_wrapper.sh"),
                "override_specs": "request_memory = 5 GB\nrequest_cpus = 1"})
    else:
        hcpdiff_wf.run()


if __name__ == "__main__":
    main()
