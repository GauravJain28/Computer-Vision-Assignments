# COL780 Assignment 2

## Link to the Dataset
Click this [link](https://drive.google.com/drive/folders/1fIrqZ10V5GeFdsUsxqABONUpAlJleBkz) to access the dataset.

## Link to the Generated Outputs
Click this [link](https://drive.google.com/drive/folders/1KjR6gtnAWPBeUs_HBN0uI8aECn9-9SGl?usp=sharing) to access the generated outputs.

## How to run the code?
Run the command:
```
python assign2.py input_dir_path/ output_path/ dataset_name(1/2/3/4/5) 
```
## File description:
- ```read_frames```: Reading frames in sorted order
- ```hessian_corner_detection```: Corner Detection using Hessian of gradients (Part 1)
- ```proximity_matching```: Matching corner points using proximity and a simple sum-of-squared-difference approach (Part 2)
- ```get_dimensions```: Calculating projected corners for all frames in the first frame and calculating corners based on them (Part 3)
- ```estimate_affine_matrix```: Estimating the Affine matrix of two frames using Projective Geometry and Least Square method. (Part 3)
- ```remove_borders```: Removing black borders of the generated panorama.

## Submitted By:
Gaurav Jain - 2019CS10349