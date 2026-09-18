
ANN_vs_CNN_Image_Kernels_Review.ipynb_
ANN vs CNN — Image Kernels, Feature Engineering & Edge Detection

Bootcamp Review Notebook

Topics covered:

ANN vs CNN — why CNNs win for images
Image Kernels — the sliding-window math
Image Feature Engineering — before vs after CNNs (HOG, SIFT, SURF)
Vertical Edge Detection — a worked example
Key CNN Properties — padding, stride, pooling, stationarity, compositionality
Tools used: Python, NumPy, Matplotlib, TensorFlow/Keras, run in VS Code with the Jupyter extension.

 
[1]
8s
# Setup — run this firstimport numpy as npimport matplotlib.pyplot as plt# TensorFlow is only needed for the Keras layer demos in sections 1, 5.4-5.5, 6 and 7.# If it isn't installed yet, uncomment the line below and run it once:# !pip install tensorflowimport tensorflow as tfprint("TensorFlow version:", tf.__version__)

 TensorFlow version: 2.20.0
1. ANN vs CNN — Why CNN for Images?

An ANN (Artificial Neural Network, also called a fully-connected / dense network) treats every input pixel as an independent number. It has no idea that pixel (5,5) is next to pixel (5,6).

ANN — Pros: handles non-linear relationships, learns automatically, works on large datasets, does classification & regression, generalizes to unseen data, can be retrained.

ANN — Cons: needs lots of data, computationally expensive, slow to train, needs careful tuning, prone to overfitting, hard to interpret ("black box"), sensitive to feature scaling, can get stuck in local minima.

Why CNN instead, for images/video:

Automatically learns spatial features (edges, textures, shapes, objects)
Preserves spatial relationships between neighboring pixels (an ANN does not)
Uses parameter sharing — the same small kernel is reused across the whole image, so far fewer weights are needed than a fully-connected layer
Translation-invariant feature detection
Efficient on high-dimensional visual data
Quick demo: how many parameters does each approach need?

 
[2]
0s
# A tiny illustration of *why* parameter sharing matters.# Suppose we have a small 28x28 grayscale image (like MNIST digits).image_h, image_w = 28, 28hidden_units = 128  # a modest hidden layer# ---- Fully-connected (ANN) first layer ----ann_params = (image_h * image_w) * hidden_units + hidden_units  # weights + biasesprint(f"Dense/ANN layer parameters:  {ann_params:,}")# ---- CNN layer: one 3x3 kernel, 32 filters ----kernel_size = 3num_filters = 32cnn_params = (kernel_size * kernel_size) * num_filters + num_filters  # weights + biasesprint(f"Conv2D layer parameters:     {cnn_params:,}")print(f"\nThe ANN layer uses about {ann_params / cnn_params:,.0f}x more parameters")print("...for a SINGLE layer, and it still doesn't know which pixels are neighbors.")

 Dense/ANN layer parameters:  100,480
Conv2D layer parameters:     320

The ANN layer uses about 314x more parameters
...for a SINGLE layer, and it still doesn't know which pixels are neighbors.
Takeaway: the CNN layer needs far fewer parameters because the same 3×3 kernel slides across the entire image (weight sharing) instead of connecting every pixel to every neuron.

2. What Is an Image Kernel?

An image kernel (a.k.a. filter) is a small matrix of numbers that slides over an image, multiplying and summing values at each position to produce a new pixel in the feature map.

Think of it as a pattern detector.

Worked example

4×4 image:

 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14 15 16
3×3 kernel:

1  0 -1
1  0 -1
1  0 -1
Take the top-left 3×3 region of the image, multiply element-wise with the kernel, and sum:

(1×1)+(2×0)+(3×-1)
(5×1)+(6×0)+(7×-1)
(9×1)+(10×0)+(11×-1)
= (1+0-3) + (5+0-7) + (9+0-11) = -6
So the first output value is -6. Sliding the kernel across the whole image (stride = 1) gives a 2×2 feature map (since a 3×3 kernel can't fully fit past the edges of a 4×4 image).

The general equation is:

𝑦𝑖,𝑗=∑𝑘∑𝑙𝑤𝑘,𝑙⋅𝑥𝑖+𝑘,𝑗+𝑙

In words: output pixel = multiply the image pixels with the corresponding kernel values, then add them all up.

 
[3]
0s
def convolve2d(img, kernel, stride=1):    """A from-scratch 2D convolution (valid padding), so you can see exactly what Conv2D does."""    img = np.array(img, dtype=float)    kernel = np.array(kernel, dtype=float)    ih, iw = img.shape    kh, kw = kernel.shape    oh = (ih - kh) // stride + 1    ow = (iw - kw) // stride + 1    output = np.zeros((oh, ow))    for i in range(oh):        for j in range(ow):            region = img[i*stride : i*stride+kh, j*stride : j*stride+kw]            output[i, j] = np.sum(region * kernel)    return output# Reproduce the slide's example exactlyimage_4x4 = [    [1, 2, 3, 4],    [5, 6, 7, 8],    [9, 10, 11, 12],    [13, 14, 15, 16],]kernel_3x3 = [    [1, 0, -1],    [1, 0, -1],    [1, 0, -1],]feature_map = convolve2d(image_4x4, kernel_3x3, stride=1)print("Feature map:\n", feature_map)assert feature_map[0, 0] == -6, "Should match the hand-worked example!"print("\nMatches the hand-worked example (top-left value = -6).")

 Feature map:
 [[-6. -6.]
 [-6. -6.]]

Matches the hand-worked example (top-left value = -6).
 
[4]
0s
# Visualize: image -> kernel -> feature map
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im0 = axes[0].imshow(image_4x4, cmap="viridis")
axes[0].set_title("Original 4x4 Image")
for (i, j), val in np.ndenumerate(np.array(image_4x4)):
    axes[0].text(j, i, int(val), ha="center", va="center", color="white")

im1 = axes[1].imshow(kernel_3x3, cmap="coolwarm")
axes[1].set_title("3x3 Kernel (filter)")
for (i, j), val in np.ndenumerate(np.array(kernel_3x3)):
    axes[1].text(j, i, int(val), ha="center", va="center")

im2 = axes[2].imshow(feature_map, cmap="viridis")
axes[2].set_title("Resulting 2x2 Feature Map")
for (i, j), val in np.ndenumerate(feature_map):
    axes[2].text(j, i, int(val), ha="center", va="center", color="white")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

 
3. Image Feature Engineering — Before vs After CNNs

Before CNNs became popular, computers couldn't automatically learn useful image features, so humans hand-designed methods to extract them.

Method	Main Idea	Learned automatically by CNN?
Edges	Find sharp changes in pixel intensity	Yes — CNNs learn edge detectors
HOG (Histogram of Oriented Gradients)	Describe the direction of edges as a histogram	No — hand-crafted
SIFT (Scale-Invariant Feature Transform)	Find & describe distinctive keypoints (corners, blobs)	No — hand-crafted
SURF (Speeded-Up Robust Features)	Faster alternative to SIFT	No — hand-crafted
Before CNN pipeline:

Image → Hand-crafted features (HOG/SIFT/SURF) → ML algorithm (e.g. SVM) → Prediction
With CNN:

Image → CNN → Learned edges → Learned textures → Learned shapes → Prediction
The CNN learns the filters/kernels automatically during training — nobody hand-designs them.

Mini demo: a hand-crafted "edge" feature vs. a CNN-learned filter

Below, we compute a classic hand-crafted edge feature (a Sobel-style gradient, in the same spirit as HOG) on a simple synthetic image, and compare it to what an untrained CNN filter produces on the same image. Both are just numbers — the difference is who chose the numbers in the kernel: a human (hand-crafted) or gradient descent during training (CNN-learned).

 
[5]
0s
# A synthetic image: a bright square on a dark background
synthetic_img = np.zeros((10, 10))
synthetic_img[3:7, 3:7] = 1.0

# --- Hand-crafted feature: simple horizontal-gradient "edge" kernel (like a HOG building block) ---
handcrafted_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])
handcrafted_feature = convolve2d(synthetic_img, handcrafted_kernel)

# --- CNN-style: an untrained Conv2D layer with random initial weights ---
cnn_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding="valid", use_bias=False)
img_batch = synthetic_img.reshape(1, 10, 10, 1).astype("float32")
cnn_output = cnn_layer(img_batch).numpy()[0, :, :, 0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(synthetic_img, cmap="gray")
axes[0].set_title("Synthetic Image")
axes[1].imshow(handcrafted_feature, cmap="viridis")
axes[1].set_title("Hand-crafted edge feature\n(fixed kernel, like HOG)")
axes[2].imshow(cnn_output, cmap="viridis")
axes[2].set_title("CNN Conv2D output\n(kernel starts random, then\nis LEARNED during training)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

print("Key point: the hand-crafted kernel is fixed forever.")
print("The CNN's kernel starts random and is updated by training — that's 'automatic feature learning'.")

 
4. Vertical Edge Detection

The kernel

 1  0 -1
 1  0 -1
 1  0 -1
compares the left column of each 3×3 region against the right column. The middle column has weight 0, so it's simply ignored.

Left pixels → +1
Middle → 0 (ignored)
Right pixels → -1
Interpretation while sliding over an image:

If the left and right pixels are similar → result ≈ 0 → no strong edge
If left and right pixels are very different → strong positive/negative result → a vertical edge is present at that location
Let's build a synthetic image with a clear vertical edge (dark on the left, bright on the right) and run the kernel over it.

 
[6]
0s
# Synthetic image with one vertical edge down the middle
edge_img = np.zeros((8, 8))
edge_img[:, :4] = 10    # dark left half
edge_img[:, 4:] = 250   # bright right half

vertical_edge_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])

edge_feature_map = convolve2d(edge_img, vertical_edge_kernel)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(edge_img, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Image with a vertical edge\n(dark | bright)")
im = axes[1].imshow(edge_feature_map, cmap="RdBu_r")
axes[1].set_title("Feature map after the\nvertical-edge kernel")
plt.colorbar(im, ax=axes[1], fraction=0.046)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

print("Feature map values:\n", edge_feature_map)
print("\nLarge negative values line up exactly where the image jumps from dark (10) to bright (250) —")
print("that's the kernel detecting the vertical edge.")

 
5. Key CNN Properties

5.1 Padding

Padding = adding extra pixels (usually zeros) around the image border.

Valid padding = no padding → output shrinks (4×4 image + 3×3 kernel → 2×2 output)
Same padding = pad so the output stays the same size as the input
Why pad?

Preserve spatial dimensions
Prevent edge/corner information from being used less often
Allow deeper networks without the image shrinking too fast
5.2 Striding

Stride = how many pixels the kernel jumps each step.

Stride = 1 → kernel moves one pixel at a time → more overlap, bigger output
Stride = 2 → kernel jumps two pixels at a time → smaller output, less computation
5.3 Pooling

Pooling reduces the size of a feature map by summarizing small regions — it is not the same as padding or striding.

Max pooling — take the maximum value in each region
Average pooling — take the average value in each region
Why pool? Reduces size/computation/memory, keeps the important signal, and adds some robustness to small shifts in the image.

 
[7]
0s
# --- 5.1 & 5.2: Padding and stride, demonstrated with Keras Conv2D ---
dummy_image = np.random.rand(1, 4, 4, 1).astype("float32")  # batch=1, 4x4, 1 channel

valid_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="valid")
same_layer  = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same")
stride2_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, padding="valid")

print("Input shape:            ", dummy_image.shape)
print("Valid padding output:   ", valid_layer(dummy_image).shape, " (shrinks: 4x4 -> 2x2)")
print("Same padding output:    ", same_layer(dummy_image).shape,  " (stays: 4x4 -> 4x4)")
print("Stride=2 output:        ", stride2_layer(dummy_image).shape)

 Input shape:             (1, 4, 4, 1)
Valid padding output:    (1, 2, 2, 1)  (shrinks: 4x4 -> 2x2)
Same padding output:     (1, 4, 4, 1)  (stays: 4x4 -> 4x4)
Stride=2 output:         (1, 1, 1, 1)
 
[8]
0s
# --- 5.3: Max pooling vs average pooling, worked example matching the slide ---
feature_map_4x4 = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [7, 2, 8, 3],
    [4, 1, 5, 9],
], dtype=float)

def pool2d(fmap, size=2, stride=2, mode="max"):
    h, w = fmap.shape
    oh = (h - size) // stride + 1
    ow = (w - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            region = fmap[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = region.max() if mode == "max" else region.mean()
    return out

max_pooled = pool2d(feature_map_4x4, mode="max")
avg_pooled = pool2d(feature_map_4x4, mode="avg")

print("Original feature map:\n", feature_map_4x4)
print("\nMax pooled (2x2, stride 2):\n", max_pooled)
print("\nAverage pooled (2x2, stride 2):\n", avg_pooled)

# Same thing using the real Keras pooling layers, for comparison
fmap_tensor = feature_map_4x4.reshape(1, 4, 4, 1).astype("float32")
keras_max = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(fmap_tensor).numpy()[0, :, :, 0]
keras_avg = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(fmap_tensor).numpy()[0, :, :, 0]
print("\nKeras MaxPooling2D output:\n", keras_max)
print("\nKeras AveragePooling2D output:\n", keras_avg)

 Original feature map:
 [[1. 3. 2. 4.]
 [5. 6. 1. 2.]
 [7. 2. 8. 3.]
 [4. 1. 5. 9.]]

Max pooled (2x2, stride 2):
 [[6. 4.]
 [7. 9.]]

Average pooled (2x2, stride 2):
 [[3.75 2.25]
 [3.5  6.25]]

Keras MaxPooling2D output:
 [[6. 4.]
 [7. 9.]]

Keras AveragePooling2D output:
 [[3.75 2.25]
 [3.5  6.25]]
 
[9]
0s
# Visualize the pooling effect
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, data, title in zip(
    axes,
    [feature_map_4x4, max_pooled, avg_pooled],
    ["Original 4x4 Feature Map", "Max Pooled (2x2)", "Average Pooled (2x2)"],
):
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f"{val:g}", ha="center", va="center", color="white")
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

 
5.4 Stationarity Principle

The same feature can appear anywhere in an image, so the same kernel can be used to detect it everywhere — this is called weight sharing, and it's why CNNs need far fewer parameters than a fully-connected ANN (see Section 1).

5.5 Compositionality Principle

Complex visual structures are built by combining simpler features layer by layer:

Raw Image → Layer 1: Edges → Layer 2: Textures → Layer 3: Object parts → Layer 4: Full objects
Example (face recognition): Pixels → Edges → Eyes/Nose/Mouth → Face structure → Person classification

Let's put 5.1–5.5 together in one small, real Sequential CNN, and inspect the shapes to see padding, striding, and pooling working together.

 
[10]
0s
# A small end-to-end CNN, purely to illustrate how the pieces fit together (not trained on real data)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            padding="same", activation="relu"),   # Section 2 & 5.1/5.2
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                # Section 5.3
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                            padding="same", activation="relu"),   # Section 5.5: builds on layer 1's features
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),  # e.g. 10 classes
])

model.summary()

 
Reading the summary: notice how the spatial size (28→14→7) shrinks after each MaxPooling2D layer (Section 5.3), while the number of filters grows (32→64), reflecting the compositionality principle — deeper layers build more complex features from the simpler ones the earlier layers found.

6. Putting It Together: A Tiny End-to-End Exercise

Goal: load a simple built-in image, apply a hand-picked kernel (Section 2 & 4), then pool it (Section 5.3) — the same pipeline a real CNN layer performs internally, just done manually so every step is visible.

Try this yourself: change my_kernel below to other values and see how the detected feature changes.

 
[11]
0s
from tensorflow.keras.datasets import mnist

# Load one real handwritten digit to work with
(x_train, y_train), _ = mnist.load_data()
sample_digit = x_train[0].astype(float) / 255.0   # normalize to 0-1
print("Digit label:", y_train[0], " | image shape:", sample_digit.shape)

# --- Try changing this kernel! ---
my_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])  # vertical edge detector

feature_map = convolve2d(sample_digit, my_kernel, stride=1)
pooled = pool2d(feature_map, size=2, stride=2, mode="max")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(sample_digit, cmap="gray")
axes[0].set_title(f"Original digit ({y_train[0]})")
axes[1].imshow(feature_map, cmap="RdBu_r")
axes[1].set_title("After vertical-edge kernel")
axes[2].imshow(pooled, cmap="RdBu_r")
axes[2].set_title("After 2x2 max pooling")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

 
(This cell needs internet access the first time, to download MNIST via Keras. If you're offline, swap sample_digit for any small NumPy grayscale image — the synthetic images from Sections 3 and 4 work just as well.)

Summary Cheat-Sheet

Term	One-line definition
ANN	Fully-connected network; treats each pixel as an independent input, no spatial awareness
CNN	Uses sliding kernels to learn spatial features directly from images
Kernel / filter	Small matrix that slides over the image to detect a pattern
Feature map	The output produced after a kernel scans an image
HOG / SIFT / SURF	Hand-crafted (pre-CNN) feature extraction methods — not learned automatically
Vertical edge kernel	[[1,0,-1],[1,0,-1],[1,0,-1]] — compares left vs right pixels, ignores the middle
Padding	Extra border pixels (usually 0) added so the kernel can cover edges / preserve size
Stride	How many pixels the kernel jumps each step
Pooling	Shrinks a feature map by summarizing regions (max or average)
Stationarity	Same kernel reused everywhere in the image → weight sharing → fewer parameters
Compositionality	Deeper layers combine simple features (edges) into complex ones (objects)
Suggested flow for presenting this notebook to reviewers: walk through Sections 1 → 5 in order, running each cell live, and pause on the visuals in Sections 2 and 4 — they make the kernel math concrete before you get to the Keras Sequential model in Section 5.

Colab paid products
-
Cancel contracts here

