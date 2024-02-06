library(tensorflow)

# Definition
# Multi-dimensional array with uniform type called tensors

# Scalar = rank 0 tensor
r0 <- as_tensor(4L)

# Vector = rank 1 tensor
r1 <- as_tensor(c(2:5))
r1 <- as_tensor(c('string1', 'word2'))
r1 <- as_tensor(c(1, 2, 4, 8.9))

# Matrix = rank 2 tensor
r2 <- as_tensor(rbind(c(1, 2, 3), c(4, 5, 6)), dtype = "float32")
r2 <- as_tensor(rbind(c(1, 2, 3), c(4, 5, 6)), dtype = tf$int8)

# Multidimensional with three axes
# Has shape of [3, 2, 5]
r3 <- as_tensor(0:29, shape = c(3, 2, 5))

# Can convert back to array
as.array(r2)
as.array(r3)

# Tensor addition
# Only works if same size/shape
a <- as_tensor(1:6, shape = c(2, 3))
b <- as_tensor(1L, shape = c(2, 3))
c <- a + b
c <- tf$add(a, b)
c <- tf$multiply(a, b)

# Or matrix multiply
# This requires the rows of `a` to match columns of `b`
# Will lead to `c` being a 2x2 matrix
a <- as_tensor(1:6, shape = c(2, 3))
b <- as_tensor(1:6, shape = c(3, 2))
c <- tf$matmul(a, b)


# Tensor shapes
# In this rank 4, axis 0 is first, axis 1 is second, axis 2 is third
# Axis "4" is actually the "-1" axis, and usually has features stored locally
r4 <- tf$zeros(shape(3, 2, 4, 5))
message("Type = ", r4$dtype) # Float or int types
message("Number of axes = ", length(dim(r4))) # Should be 4
message("Shape = ", r4$shape) # Should be 3245
message("Elements along 0 axis = ", dim(r4)[1]) # Should be 3
message("Elements along last axis = ", tail(dim(r4), 1)) # Should be 5
message("Total size = ", length(r4)) # Or tf$size()
