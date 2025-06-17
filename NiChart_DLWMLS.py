import glob
import os
import shutil
from typing import Any
import logging

import numpy as np
import pandas as pd

import SimpleITK as sitk

import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform

os.environ['CURL_CA_BUNDLE'] = ''

def reorient_to_lps(input_path: str, output_path: str):
    """
    Reorients a NIfTI image to LPS (Left-Posterior-Superior) orientation.

    This function loads a NIfTI image, calculates the necessary transformation
    to reorient it to LPS, applies the transformation to the image data,
    updates the affine matrix, and saves the new image.

    Args:
        input_path (str): The file path for the input NIfTI image.
        output_path (str): The file path where the reoriented NIfTI image will be saved.
    """
    print(f"Loading image: {input_path}")
    # Load the nifti image
    img = nib.load(input_path)

    # Get the original orientation from the affine
    original_orientation = nib.aff2axcodes(img.affine)
    print(f"Original orientation: {original_orientation}")

    # Define the target orientation (LPS)
    target_orientation = ('L', 'P', 'S')
    print(f"Target orientation: {target_orientation}")

    if original_orientation == target_orientation:
        print("Image is already in the target LPS orientation. No changes needed.")
        # If you still want to save a copy, uncomment the line below
        # nib.save(img, output_path)
        return

    # Determine the transformation needed to go from original to target orientation
    # axcodes2ornt gives an orientation array, which is a tuple of axis numbers and directions
    start_ornt = nib.orientations.axcodes2ornt(original_orientation)
    end_ornt = nib.orientations.axcodes2ornt(target_orientation)

    # ornt_transform finds the transformation between two orientation arrays
    transform = nib.orientations.ornt_transform(start_ornt, end_ornt)

    # Apply the orientation transform to the image data
    print("Applying orientation transform to image data...")
    reoriented_data = nib.orientations.apply_orientation(img.get_fdata(), transform)

    # The affine matrix needs to be updated to reflect the new data orientation.
    # inv_ornt_aff corrects the affine.
    new_affine = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape)

    # Create the new NIfTI image object with the reoriented data and new affine
    new_img = nib.Nifti1Image(reoriented_data, new_affine, img.header)

    # Save the reoriented image
    print(f"Saving reoriented image to: {output_path}")
    nib.save(new_img, output_path)

    # # --- Verification (optional) ---
    # # Load the newly saved image and check its orientation
    # verify_img = nib.load(output_path)
    # final_orientation = nib.aff2axcodes(verify_img.affine)
    # print(f"Verification: New image orientation is {final_orientation}")
    # if final_orientation == target_orientation:
    #     print("Successfully reoriented to LPS.")
    # else:
    #     print("Warning: Reorientation may not have been successful.")

def run_DLWMLS(in_dir: str,
               in_suff: Any,
               out_dir: str,
               out_suff: Any,
               device: str,
               extra_args: str = "",) -> None:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    os.system(f"DLWMLS -i {in_dir} -o {out_dir} -device {device} " + extra_args)
    
    for fname in glob.glob(os.path.join(out_dir, "DLMUSE_mask_*.nii.gz")):
        new_fname = fname.replace("DLMUSE_mask_", "", 1).replace(in_suff, out_suff)
        # os.rename(fname, new_fname)
        shutil.copyfile(fname, new_fname)

def register_flair_to_t1(t1_image_path='', 
                         flair_image_path='', 
                         output_path='',
                         data_dir='',
                         mrid='', 
                         t1_image_suffix='_T1_LPS.nii.gz', 
                         fl_image_suffix='_FL_LPS.nii.gz', 
                         output_dir='',
                         ):
    """
    Registers a FLAIR image to a T1 image using SimpleITK.

    This function performs an affine registration to align the FLAIR (moving) image
    to the T1 (fixed) image. 
    
    The final transformation are saved to the specified output directory.
    """

    # Args:
    # *Option 1: entire directory mode
    # t1_image_path (str): The file path for the T1-weighted image (fixed image).
    # flair_image_path (str): The file path for the FLAIR image (moving image).
    # output_dir (str): The directory where the output files will be saved.
    if (t1_image_path != '' and flair_image_path != '' and output_path != '') and (data_dir == '' and mrid == '' and output_dir == ''):
        # output_registered_image_path = os.path.join(output_dir, "registered_flair.nii.gz")
        output_transform_path = output_path
    # *Option 2: single image mode
    # data_dir (str): The file directory to store both input/outputs
    # mrid (str): The unique idenifier of the file (must be identical header)
    # t1_image_suffix (str): Suffix indicating the T1 modality of the file
    # fl_image_suffix (str): Suffix indicating the FLAIR modality of the file
    elif (t1_image_path == '' and flair_image_path == '' and output_path == '') and (data_dir != '' and mrid != '' and output_dir != ''):
        t1_image_path = os.path.join(data_dir, "T1", mrid + t1_image_suffix)
        flair_image_path = os.path.join(data_dir, 'FLAIR', mrid + fl_image_suffix)
        output_transform_dir = os.path.join(output_dir,"TFMs")
        if not os.path.exists(output_transform_dir):
            os.makedirs(output_transform_dir)
        # output_registered_image_path = os.path.join(data_dir, mrid + "FL_rT1.nii.gz")
        output_transform_path = os.path.join(output_transform_dir, mrid + "_FL_to_T1.tfm")
    else:
        print(f"Invalid input arg combination")
        return


    # Read the images
    print("Reading images...")
    fixed_image = sitk.ReadImage(t1_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(flair_image_path, sitk.sitkFloat32)

    # Set up the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Initial transform: Use AffineTransform for affine registration
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.AffineTransform(3),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # # Connect an observer to monitor the registration process
    # def command_iteration(method) :
    #     print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    # Execute the registration
    print("Starting registration...")
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                   sitk.Cast(moving_image, sitk.sitkFloat32))

    # Post-registration analysis
    print("Registration complete.")
    print(f"Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration_method.GetMetricValue()}")

    # Resample the moving image using the final transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(final_transform)
    print(f"Saving transform to: {output_transform_path}")
    sitk.WriteTransform(final_transform, output_transform_path)

    # # Register the image
    # resampled_moving_image = resampler.Execute(moving_image)

    # Save the results
    # print(f"Saving registered image to: {output_registered_image_path}")
    # sitk.WriteImage(resampled_moving_image, output_registered_image_path)


def apply_saved_transform(fixed_image_path, moving_image_path, transform_path, output_image_path):
    """
    Applies a saved SimpleITK transformation to an image.

    Args:
        fixed_image_path (str): Path to the reference/fixed image. The output image
                                will have the same size, spacing, and origin as this image.
        moving_image_path (str): Path to the image that needs to be transformed.
        transform_path (str): Path to the .tfm file containing the transformation.
        output_image_path (str): Path to save the resulting resampled image.
    """
    # Read the fixed and moving images
    print("Reading images...")
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Read the transformation from the file
    print(f"Reading transform from {transform_path}...")
    try:
        loaded_transform = sitk.ReadTransform(transform_path)
    except Exception as e:
        print(f"Error reading transform file: {e}")
        return

    # Create a resampler
    resampler = sitk.ResampleImageFilter()

    # --- Configure the resampler ---
    # 1. Set the reference image: This defines the output image's grid (size, spacing, etc.)
    resampler.SetReferenceImage(fixed_image)

    # 2. Set the interpolator
    # Use sitk.sitkLinear for intensity images (like T1, FLAIR)
    # Use sitk.sitkNearestNeighbor for label/mask images to preserve labels
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # 3. Set the default pixel value for areas outside the moving image
    resampler.SetDefaultPixelValue(0) # Use 0 for black background

    # 4. Set the transformation
    resampler.SetTransform(loaded_transform)

    # Execute the resampling
    print("Applying transform and resampling image...")
    resampled_image = resampler.Execute(moving_image)

    # Save the output image
    print(f"Saving resampled image to: {output_image_path}")
    sitk.WriteImage(resampled_image, output_image_path)
    print("Done.")


def segment_multilabel_mask_and_calculate_volumes(mask_a_path: str, 
                                                  mask_b_path: str, 
                                                  output_path: str,
                                                  save_as_csv: bool,
                                                  csv_path: str,
                                                  mrid: str) -> None:
    """
    Segments a binary mask (A) by a multi-label mask (B), saves the result,
    and calculates the volume for each label in the intersection.

    Args:
        mask_a_path (str): File path for the binary input mask (Mask A, values 0 or 1).
        mask_b_path (str): File path for the multi-label input mask (Mask B, values 0 to N).
        output_dir (str): Directory where the output mask will be saved.
    """
    print("--- Starting Multi-Label Segmentation and Volume Calculation ---")

    # Load the NIfTI images
    print(f"Loading Mask A: {mask_a_path}")
    img_a = nib.load(mask_a_path)
    print(f"Loading Mask B: {mask_b_path}")
    img_b = nib.load(mask_b_path)

    # --- Sanity Checks ---
    if img_a.shape != img_b.shape:
        raise ValueError("Error: Input masks have different dimensions. "
                         f"Mask A: {img_a.shape}, Mask B: {img_b.shape}")
    if not np.allclose(img_a.affine, img_b.affine):
        raise ValueError("Error: Input masks have different affine transformations. "
                         "They are not in the same space.")
    print("Masks have compatible dimensions and affines.")

    # Get image data as numpy arrays
    data_a = img_a.get_fdata().astype(bool) # Mask A is binary
    data_b = img_b.get_fdata().astype(np.uint16) # Mask B is multi-label

    # Find all unique non-zero labels in Mask B
    labels_in_b = np.unique(data_b[data_b > 0])
    print(f"Found {len(labels_in_b)} non-zero labels in Mask B: {labels_in_b}")

    # --- Initialize containers for results ---
    volume_results = {}
    final_segmented_data = np.zeros_like(data_b, dtype=data_b.dtype)
    
    # Calculate voxel volume once
    voxel_volume = np.abs(np.linalg.det(img_a.affine[:3, :3]))

    # --- Iterate over each label, perform segmentation, and calculate volume ---
    for label in labels_in_b:
        # Create a temporary binary mask for the current label
        mask_for_label_b = (data_b == label)
        
        # Perform the intersection: where Mask A is true AND Mask B has the current label
        intersection_data = data_a & mask_for_label_b
        
        # Add the segmented region to our final output mask, preserving the label
        final_segmented_data[intersection_data] = label
        
        # Calculate volume for this specific intersection
        non_zero_voxel_count = np.sum(intersection_data)
        total_volume_mm3 = non_zero_voxel_count * voxel_volume
        
        # Store the result
        volume_results[label] = total_volume_mm3

    # --- Print the Results ---
    print("\n--- Volume Results ---")
    print(f"Volume of a single voxel: {voxel_volume:.4f} mm^3")
    print("-" * 25)
    if not volume_results:
        print("No overlap found between Mask A and any labels in Mask B.")
    else:
        for label, volume in volume_results.items():
            print(f"Label {label:>3}: {volume:>10.4f} mm^3")
    print("------------------------")
    
    # --- Save the Resulting Multi-Label Mask ---
    output_img = nib.Nifti1Image(final_segmented_data, img_a.affine, img_a.header)
    
    print(f"\nSaving final multi-label segmented mask to: {output_path}")
    nib.save(output_img, output_path)
    
    print(volume_results)
    # --- Save the Resulting Multi-Label Mask Volumes as CSV ---
    if save_as_csv:
        df_csv = pd.DataFrame(volume_results, index=[mrid])
        df_csv.to_csv(csv_path)

    print("\nProcess finished successfully.")


if __name__ == '__main__':

    import shutil

    t1_image_suffix='_T1_LPS.nii.gz'
    fl_image_suffix='_FL_LPS.nii.gz'
    dlmuse_t1_mask_suffix = '_T1_LPS_DLMUSE.nii.gz'
    dlwmls_fl_mask_suffix = '_FL_LPS_DLWMLS.nii.gz'
    fl_to_t1_xfm_suffix = '_FL_to_T1.tfm'
    dlwmls_to_t1_reg_suffix = '_DLWMLS_REG_to_T1.nii.gz'

    data_directory = '/home/kylebaik/Projects/DLWMLS_Segment/Sample_data/'
    t1_path = os.path.join(data_directory, "T1")
    fl_path = os.path.join(data_directory, "FLAIR")

    output_directory = '/home/kylebaik/Projects/DLWMLS_Segment/Sample_data/NiChart_DLWMLS'
    dlwmls_path = os.path.join(output_directory, 'DLWMLS')
    tfm_path = os.path.join(output_directory,'TFMs')
    dlwmls_tfmed = os.path.join(output_directory,'DLWMLS_TFM_to_T1')

    dlmuse_directory = '/home/kylebaik/Projects/DLWMLS_Segment/Sample_data/NiChart_DLMUSE'
    dlmuse_suffix = '_T1_LPS_DLMUSE.nii.gz'

    mrids = ['B10081264_009', 'B10081264_027']

    #####################################################
    ########## START NiChart_DLWMLS Pipeline ############
    #####################################################


    print(f"LPS Orienting the image")
    for mrid in mrids:
        # Reorient T1
        reorient_to_lps(input_path=os.path.join(t1_path, mrid + t1_image_suffix),
                        output_path=os.path.join(t1_path, mrid + t1_image_suffix))
        # Reorient FLAIR
        reorient_to_lps(input_path=os.path.join(fl_path, mrid + fl_image_suffix),
                        output_path=os.path.join(fl_path, mrid + fl_image_suffix))
        
    print(f"Processing DLWMLS on FLAIR folder")
    # Check if the folder exists
    if os.path.exists(dlwmls_path):
        shutil.rmtree(dlwmls_path) # Remove the directory and its contents
        print(f"All files and subdirectories in '{dlwmls_path}' have been removed.")
    else:
        print(f"Folder '{dlwmls_path}' not found.")
    run_DLWMLS(in_dir=fl_path, 
               in_suff='_FL_LPS.nii.gz', 
               out_suff='_DLWMLS.nii.gz', 
               out_dir=dlwmls_path,device='cuda')

    
    print(f"Creating transformation matrix from FL to T1, applying to the DLWMLS Masks")
    for mrid in mrids:
        register_flair_to_t1(t1_image_path=os.path.join(t1_path, mrid + t1_image_suffix),
                             flair_image_path=os.path.join(fl_path, mrid + fl_image_suffix),
                             output_path=os.path.join(tfm_path, mrid+fl_to_t1_xfm_suffix))
        
        apply_saved_transform(fixed_image_path=os.path.join(t1_path, mrid + t1_image_suffix),
                              moving_image_path=os.path.join(dlwmls_path, mrid+'_FL_LPS_WMLS.nii.gz'),
                              transform_path=os.path.join(tfm_path, mrid + fl_to_t1_xfm_suffix),
                              output_image_path=os.path.join(dlwmls_tfmed, mrid + dlwmls_to_t1_reg_suffix))
        
        segment_multilabel_mask_and_calculate_volumes(mask_a_path=os.path.join(dlwmls_tfmed, mrid + dlwmls_to_t1_reg_suffix),
                                                      mask_b_path=os.path.join(dlmuse_directory, mrid + dlmuse_suffix),
                                                      output_path=os.path.join(output_directory, 'DLWMLS_DLMUSE_Segmented', mrid + "_DLWMLS_DLMUSE_Segmented.nii.gz"),
                                                      save_as_csv=True,
                                                      csv_path=os.path.join(output_directory, mrid + '_DLWMLS_DLMUSE_Segmented_Volumes.csv'),
                                                      mrid = mrid)
