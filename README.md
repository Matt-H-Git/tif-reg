# Transform Invariant Feature Registration

This is my non-official implementation of TIF-Reg, a powerful registration model which performs well regardless of 
how much rotation is present.

As of the time of writing, there is no official implementation that is publicly available so the model architecture is purely
derived from the paper. All thanks go to the authors of the original paper for making their work public.

Sources:

[TIF-Reg paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8434253/pdf/sensors-21-05778.pdf)

[DGCNN Paper](https://arxiv.org/pdf/1801.07829)

[DGCNN Model](https://github.com/WangYueFt/dcp/blob/master/model.py#L277)

[SE Math source (FMR)](https://github.com/XiaoshuiHuang/fmr)

## Performance Evaluation (and explanation of shortcut)

TIF-Reg performs extremely well under given constraints. Assuming that the source and target point clouds are identical (but in a different position
and orientation) and contain no noise then TIF-Reg is guaranteed to converge at the global optimal solution.

 As soon as partial point clouds, or noisy point clouds are introduced then TIF-Reg's performance degrades rapidly.

### Explanation of the shortcut

For the shortcut we assume the optimal conditions of two identical point clouds used as input.

This model's architecture is broken down into 4 sections, the 3rd of which is the feature correspondence step. Here is the equation for the correspondence weights:

`W = softmax(FxFy^T)`

Where `Fx` and `Fy` are the features extracted from the source and target point clouds in the first 2 steps.

Looking at figure 2 (in the paper) displaying the tif reg architecture diagram it is known that both `x` and `y` clouds pass through the same feature extraction blocks.

Using this information, it can be determined that given identical point clouds `x`and `y`, the extracted features `Fx` and `Fy` will also be identical to each other. The correspondence weights calculation carries out a matrix multiplication before the softmax which, if using identical matrices, will result in an identity matrix. This also results in the weights being an identity matrix.

The final calculation of the 3rd step is `z = Wy` which simplifies to `z = y` when the weights matrix is identity.


# Files
## args.py

Contains global arguments which are used in all other files. Each individual file may contain its own specific args.

## train.py

Train the model. The datasets usually work by taking a pointcloud as the target,
applying a random transform (where the max rotation and translation can be set)
and using the model to solve for the inverse of that transform

## evaluate.py

Same as train except uses a pretrained model and won't modify any gradients (just inference)

## demo.py

Run visualisations of datasets using a pretrained model