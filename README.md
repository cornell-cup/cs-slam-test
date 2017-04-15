#Overview

This repository contains the code for all of our SLAM preliminary tests.

##Stereo Cameras

We attempt to use stereo cameras to determine depth and ultimately distance to visual objects.

###Tools Used

[StereoVision](https://github.com/erget/StereoVision) : OpenCV wrapper for simple calibration of stereo cameras. Contains wrapper modules as well as scripts for quick calibration.

StereoVision has not been updated to support OpenCV 3. Manually updated scripts can be found in the "stereo\updated_stereo_vision_scripts" directory.

To capture images for calibration:

<code>
    python capture_chessboard camera_id_1 camera_id_2 number_of_photos photos_output_directory
</code>

Current calibration was done with 50 photos.

To get calibration files:

<code>
    python calibrate_cameras photos_input_directory files_output_directory
</code>