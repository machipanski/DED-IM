#! /bin/bash

if [ $# -ne 1 ]; then
        echo "EXEMPLOR DE USO: $0 stl_suporte/peca_suporte_2.stl";
        exit 1;
fi

rm *.svg *.png;

make

#The options for the output are: [-2D, -3D, -video, -No]
#The options for slicing are: [-Trivial, -Incremental]
#The options for adaptive are: [true, false]. 
#The thickness is given in millimeters. 
#The options for rotation are: [true, false] (to change de Z cutting plane).
#The options for contour orientation (clockwise or counter-clockwise) are: [true, false].
#The options for write images are: [true, false].
#The dpi is the size of the image: any positive integer.

#./main -Incremental \
#    -model ./stl_models/02.femur.stl \
#    -thickness 2.0 \
#    -adaptive false \
#    -3D \
#    -rotate false \
#    -orienting_contours true \
#    -write_images true \
#    -dpi 300 \
#    -fill_triangles false

./main -Incremental \
    -model $1 \
    -thickness 2.0 \
    -adaptive false \
    -3D \
    -rotate false \
    -orienting_contours true \
    -write_images true \
    -dpi 300 \
    -fill_triangles false 
