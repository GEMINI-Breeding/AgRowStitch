We provide four sets of images for initial testing that can be accessed here:

test_images1: https://ucdavis.box.com/s/xz60o4mpzue7tfe7aabsrmm3l32byb2z

test_images2: https://ucdavis.box.com/s/nfdc31a3y3zl3sd5u3y7959o929ezl6o

test_images3: https://ucdavis.box.com/s/h3e81mmlnximqerknphzfwsng3fom3ll

test_images4: https://ucdavis.box.com/s/gpztrrzjxwws4lhqca16rnrup867suwt

test_images6: https://ucdavis.box.com/s/fb35cc4wieo1of565yr4iwmyq77te2h8



The output of these images can be found with the corresponding labels for comparison with other methods. Poor alignment is generally due to high camera movement in the non-stitching direction, plant movement, or a frame rate too low for the amount of movement. If stitching fails, reducing the min_inliers will help ignore poor matching, but may result in lower quality. To improve stitching, try with a higher frame rate. For affine stitching with test_images6, there is a misalignment that is fixed when a spherical projection is chosen (shown here).
