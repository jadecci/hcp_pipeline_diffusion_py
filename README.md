# HCP Pipeline for Diffusion Processing

This is a Python implementation of the Diffusion Preprocessing procedure in 
[HCP Pipeline](https://github.com/Washington-University/HCPpipelines.git). Similar to the original 
implementation, the input data need to be organised with HCP-like folder structure.

## Prerequisite

The pipeline requires 3 softwares: FSL, FreeSurfer, and Connectome Workbench. To use containerised 
versions, the absolute path to the Singularity images can be passed with the `--fsl_simg`, 
`--fs_simg`, and `--wb_simg` flags respectively.

## Usage

```console
hcpdiffpy --help

usage: hcpdiffpy [-h] [--workdir WORK_DIR] [--output_dir OUTPUT_DIR] [--fsl_simg FSL_SIMG]
                 [--fs_simg FS_SIMG] [--wb_simg WB_SIMG] [--condordag]
                 subject_dir subject ndirs [ndirs ...] phases [phases ...] echo_spacing

HCP Pipeline for diffusion preprocessing

positional arguments:
  subject_dir           Absolute path to the subject's data folder (organised in HCP-like structure)
  subject               Subject ID
  ndirs                 List of numbers of directions
  phases                List of 2 phase encoding directions
  echo_spacing          Echo spacing used for acquisition in ms

options:
  -h, --help            show this help message and exit
  --workdir WORK_DIR    Absolute path to work directory (default: /data/project/hcpa_dwi_proc)
  --output_dir OUTPUT_DIR
                        Absolute path to output directory (default: /data/project/hcpa_dwi_proc)
  --fsl_simg FSL_SIMG   singularity image to use for command line functions from FSL (default: None)
  --fs_simg FS_SIMG     singularity image to use for command line functions from FreeSurfer
                        (default: None)
  --wb_simg WB_SIMG     singularity image to use for command line functions from Connectome
                        Workbench (default: None)
  --condordag           Submit workflow as DAG to HTCondor (default: False)
```

# References
Glasser, M.F., et al. 2013. The minimal preprocessing pipelines for the Human Connectome Project. 
*NeuroImage*, 80:105-24. 
DOI: [10.1016/j.neuroimage.2013.04.127](https://doi.org/10.1016/j.neuroimage.2013.04.127)
