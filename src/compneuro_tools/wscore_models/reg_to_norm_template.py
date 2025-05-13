import os
import subprocess as sp

from argparse import ArgumentParser


def _setup_parser():
    desc = ("Register a list of images to a template_GM using FSL, and "
            " merge the results into a single image.\nIMPORTANT: "
            "Your list of images must come from an 'imglob' command.")
    parser = ArgumentParser(description=desc)
    
    parser.add_argument("--list_of_images",
                        type=str,
                        required=True,
                        help=("Path to the list of images to be registered. "
                              "Must be a file containing the output of an "
                              "imglob command."))

    parser.add_argument("--template",
                        type=str,
                        required=True,
                        help=("Path to the template image to register to. "
                              "This should be a GM template created using "
                              "FSL's fslvbm_2_template command."))

    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help=("Path to the output directory where the "
                              "registered images will be saved."))

    args = parser.parse_args()
    return args


def _check_args(args):
    # Check that the list of images file exists
    if not os.path.exists(args.list_of_images):
        raise FileNotFoundError((f"List of images file {args.list_of_images} "
                                 "does not exist."))

    # Check that the template image exists
    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Template image {args.template} does not exist.")

    # Check that the output directory exists, if not, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Output directory {args.output} created.")

    return args


def _check_fsl():
    # Check if FSL is installed
    if not os.getenv("FSLDIR"):
        raise EnvironmentError(("FSLDIR environment variable is not set."
                               " Check your FSL installation. If not installed, "
                               "install FSL and set the FSLDIR environment variable."))
    else:
        return os.path.join(os.environ["FSLDIR"], "bin")


def main():
    args = _setup_parser()
    args = _check_args(args)
    FSLBIN = _check_fsl()

    # Read the list of images
    image_paths = open(args.list_of_images,"r").readline().replace("\n","").split(" ")
    template = args.template
    template_name = os.path.basename(template).split(".")[0]

    ### STEP 1: NON-LINEAR REGISTRATION OF GM IN SUBSPACE TO TEMPLATE_GM
    print(("### Registering all participant space GM segmentations to"
           f" the provided template: {template_name}"))

    for img_path in image_paths:
        print(f"\n### Processing entry: {img_path}")
        img_path = os.path.abspath(img_path)
        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]

        # Apply non-linear registration, output the jacobian
        print(f"# Registering {img_name}")
        registered_GM_path = os.path.join(img_dir,
                                          f"{img_name_no_ext}_in_{template_name}.nii.gz")
        jacobian_path = os.path.join(img_dir,
                                     f"{img_name_no_ext}_jacobian.nii.gz")
        registered_GM_mod_path = os.path.join(img_dir,
                                     f"{img_name_no_ext}_in_{template_name}_mod.nii.gz")

        cmd = (f'{FSLBIN}/fsl_reg {img_path} {template} {registered_GM_path} '
               f'-fnirt "--config=GM_2_MNI152GM_2mm.cnf --jout={jacobian_path}"')
        if not os.path.exists(registered_GM_mod_path):
            sp.run(cmd, shell=True, check=True)
            print("# Registered GM image to template")
        else:
            print(f"# GM image already registered to template: {registered_GM_mod_path}")

        # Apply the jacobian to the registered GM image
        cmd = (f'{FSLBIN}/fslmaths {registered_GM_path} -mul {jacobian_path} '
               f'{registered_GM_mod_path}')
        if not os.path.exists(registered_GM_mod_path):
            sp.run(cmd, shell=True, check=True)
            print("# Applied jacobian to registered GM image")
        else:
            print(f"# GM image already modulated: {registered_GM_mod_path}")

        # Remove jacobian and intermediate non modulated registration results
        warp_partial_path = os.path.join(img_dir,
                                 f"{img_name_no_ext}*_in_{template_name}_warp*")
        cmd = (f"rm -v {jacobian_path} {registered_GM_path}"
               f" {warp_partial_path} *_struc_GM_to_{template_name}_GM.log")
        sp.run(cmd, shell=True, check=False)

    ### STEP 2: MERGE ALL REGISTERED GM IMAGES
    output_merge_path = os.path.join(args.output,
                                     f"GM_mod_merg_in_{template_name}.nii.gz")
    print(f"\n### Merging all registered GM images to {output_merge_path}")

    cmd = (f"{FSLBIN}/fslmerge -t {output_merge_path} "
           f"`{FSLBIN}/imglob {img_dir}/*_in_{template_name}_mod.*`")
    sp.run(cmd, shell=True, check=True)

    output_merge_path_s3 = os.path.join(args.output,
                                       f"GM_mod_merg_in_{template_name}_s3.nii.gz")
    cmd = (f"{FSLBIN}/fslmaths {output_merge_path} -s 3 {output_merge_path_s3}")
    sp.run(cmd, shell=True, check=True)
    print("# Finished merging\n### DONE!")


if __name__ == "__main__":
    main()