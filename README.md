# DLWMLS_Segment

Step 1: Receive inputs (LPS Oriented T1s images, DLMUSE Masks in T1 space, LPS Oriented FLAIR images)
Step 2: process DLWMLS on FLAIR image, get WMLS mask
Step 3.1: Orient FLAIR to T1 -> get transformation matrix
Step 3.2: Apply transformation matrix to the S2 WMLS mask
Step 4: Segment the DLWMLS mask using DLMUSE mask
