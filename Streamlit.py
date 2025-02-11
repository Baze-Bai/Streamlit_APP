import streamlit as st
import os
import numpy as np
import joblib
import lime
import lime.lime_image
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.segmentation import mark_boundaries, slic
from tensorflow.keras.models import load_model
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有 GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 禁用 TensorFlow 的警告和错误日志
import tensorflow as tf
import io
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
import numpy as np
import boto3
from io import BytesIO
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import feature

################################################################S3#####################################################################################################################################################

aws_region = "us-east-1"  

s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_KEY"),
    region_name=aws_region
)

bucket_name = os.environ.get("BUCKET_NAME")
model_key = os.environ.get("MODEL_KEY")
local_path = "final_hybrid_model.h5"
s3.download_file(bucket_name, model_key, local_path)

model_key_2 = "RandomForest_breakhis.pkl"  # Key for the second model
local_path_2 = "RandomForest_breakhis.pkl"     # Local path for the second model
s3.download_file(bucket_name, model_key_2, local_path_2)

# Load machine learning model
model_path = local_path_2
ml_model = joblib.load(model_path)

    
################################################################S3#####################################################################################################################################################


################################################################Define the structure and Load model#####################################################################################################################################################


# --------------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) block
# --------------------------------------------------------------------------------
def squeeze_excitation_block(input_tensor, reduction_ratio=16):
    channel_dim = int(input_tensor.shape[-1])
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channel_dim // reduction_ratio, activation='relu')(se)
    se = layers.Dense(channel_dim, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channel_dim))(se)
    return layers.Multiply()([input_tensor, se])

# --------------------------------------------------------------------------------
# Depthwise-Separable convolution utility
# --------------------------------------------------------------------------------
def depthwise_separable_conv(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = layers.SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# --------------------------------------------------------------------------------
# Pretrained EfficientNetV2B0 Bottleneck
# --------------------------------------------------------------------------------
def bottleneck0(inputs):
    backbone = tf.keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False)
    for layer in backbone.layers:
        layer.trainable = False
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    return x

# --------------------------------------------------------------------------------
# Neck Section
# --------------------------------------------------------------------------------
def neck_section(inputs):
    x = depthwise_separable_conv(inputs, filters=256, kernel_size=7, strides=(2, 2), padding='same')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = squeeze_excitation_block(x)
    x = layers.Flatten()(x)
    return x

# --------------------------------------------------------------------------------
# Vision Transform (patch-based) branch
# --------------------------------------------------------------------------------
def vision_transform(inputs, patch_size):
    x = layers.Reshape((patch_size[0], patch_size[1], -1))(inputs)
    x = layers.Lambda(lambda image: K.cast(image, 'float32') / 255.0)(x)
    x = depthwise_separable_conv(x, 64, kernel_size=3)
    x = depthwise_separable_conv(x, 64, kernel_size=3)
    x = layers.MaxPooling2D((2, 2))(x)
    x = depthwise_separable_conv(x, 128, kernel_size=3)
    x = depthwise_separable_conv(x, 128, kernel_size=3)
    x = layers.MaxPooling2D((2, 2))(x)
    x = squeeze_excitation_block(x)
    x = layers.Flatten()(x)
    return x

# --------------------------------------------------------------------------------
# Adaptive Branch Fusion
# --------------------------------------------------------------------------------
def adaptive_branch_fusion(*branches, hidden_dim=128):
    weighted_branches = []
    for branch in branches:
        gate = layers.Dense(hidden_dim, activation='relu')(branch)
        gate = layers.Dense(1, activation='sigmoid')(gate)
        weighted = layers.Multiply()([branch, gate])
        weighted_branches.append(weighted)
    return layers.Concatenate()(weighted_branches)

# --------------------------------------------------------------------------------
# Build the Full Hybrid Model
# --------------------------------------------------------------------------------
def build_hybrid_model(input_shape, num_classes, patch_size):
    inputs = layers.Input(shape=input_shape)
    efficientnet_features = bottleneck0(inputs)
    patches_features = vision_transform(inputs, patch_size)
    neck_features = neck_section(inputs)
    fused = adaptive_branch_fusion(efficientnet_features, patches_features, neck_features, hidden_dim=64)
    outputs = layers.Dense(num_classes, activation='softmax')(fused)
    return models.Model(inputs=inputs, outputs=outputs)


# Load deep learning model
dl_model = build_hybrid_model(
    input_shape=(128, 128, 3),  # 确保使用 input_shape 而不是 batch_shape
    num_classes=2,
    patch_size=(32, 32)
)

dl_model_path = local_path
dl_model.load_weights(dl_model_path)

################################################################Define the structure and Load model#####################################################################################################################################################


################################################################Deep learning part#####################################################################################################################################################

def df_preprocess(image, img_size=(128, 128)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    
    return np.expand_dims(img, axis=0)

def DL_classify(image, model):
    img = df_preprocess(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    return "Malignant (Yes)" if predicted_class == 1 else "Benign (No)"

def normalize_map(heatmap):
    """Normalize a heatmap to the [0,1] range."""
    heatmap -= heatmap.min()
    denom = (heatmap.max() - heatmap.min()) + 1e-8
    heatmap /= denom
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.5, cmap='jet'):
    """
    Overlay `heatmap` on `img`.
    """
    colormap = plt.cm.get_cmap(cmap)
    colored_heatmap = colormap(heatmap)[..., :3]
    overlay = (1 - alpha) * img + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay

def integrated_gradients(model, x, baseline=None, steps=50, class_idx=0):
    """
    Computes Integrated Gradients 
    """
    x = tf.cast(x, tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(x)
    else:
        baseline = tf.cast(baseline, tf.float32)

    B, H, W, C = x.shape
    
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps+1), [steps+1, 1, 1, 1, 1])
    x_expanded = tf.expand_dims(x, axis=0)
    baseline_expanded = tf.expand_dims(baseline, axis=0)
    interpolated = baseline_expanded + alphas * (x_expanded - baseline_expanded)

    with tf.GradientTape() as tape:
        interpolated_reshaped = tf.reshape(interpolated, [(steps+1)*B, H, W, C])
        tape.watch(interpolated_reshaped)
        preds = model(interpolated_reshaped)
        preds_for_class = preds[:, class_idx]
        loss = tf.reduce_sum(preds_for_class)
    
    grads = tape.gradient(loss, interpolated_reshaped)
    if grads is None:
        raise ValueError("Gradient is None. Not differentiable.")
    
    grads_reshaped = tf.reshape(grads, [steps+1, B, H, W, C])
    avg_grads = tf.reduce_mean(grads_reshaped[1:], axis=0)
    ig = (x - baseline) * avg_grads
    return ig.numpy()


def DL_explainability(model, image, class_idx=1):
    img_batch = df_preprocess(image)
    ig_map = integrated_gradients(model, img_batch, class_idx=class_idx)
    
    original_img = img_batch[0]
    normalized_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
    
    ig_map_single = ig_map[0]
    ig_map_2d = np.mean(ig_map_single, axis=-1)
    ig_map_norm = normalize_map(ig_map_2d)
    overlay_img = overlay_heatmap(normalized_img, ig_map_norm, alpha=0.5, cmap='jet')
     
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(normalized_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # IG heatmap
    im = axes[1].imshow(ig_map_norm, cmap="jet")
    axes[1].set_title("IG Map (2D Mean)")
    plt.colorbar(im, ax=axes[1])
    axes[1].axis("off")
    
    # Overlay images
    axes[2].imshow(overlay_img)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    # Adjust layout 
    plt.tight_layout()

    # load image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)  
    plt.close(fig)  

    return img_buffer

################################################################Deep learning part#####################################################################################################################################################




################################################################machine learning part#####################################################################################################################################################

# LBP Feature Extraction
def extract_lbp_features(image, num_points=24, radius=8):
    gray = rgb2gray(image)
    lbp = feature.local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# ✅ Function to preprocess image and extract LBP features
def process_image(image):
    image_resized = resize(image, (128, 128))  # Resize to required size
    features = extract_lbp_features(image_resized)
    return np.array(features).reshape(1, -1)  # Convert to 2D array

# ✅ Function to predict using model
def classify_image(image, model):
    features = process_image(image)
    prediction = model.predict(features)[0]
    return "Malignant (Yes)" if prediction == 1 else "Benign (No)"

# ✅ **Better LIME prediction function**
def model_predict(image_batch):
    processed_features = []
    for img in image_batch:
        img_resized = resize(img, (128, 128))
        features = extract_lbp_features(img_resized)
        processed_features.append(features)
    
    return ml_model.predict_proba(np.array(processed_features))  # ✅ LIME now passes LBP features!

# ✅ **Improved LIME Explanation**
def generate_lime_explanation(image):
    explainer = lime.lime_image.LimeImageExplainer()

    # ✅ Use SLIC segmentation with more superpixels
    segmentation_fn = lambda x: slic(x, n_segments=300, compactness=15, sigma=1)  

    explanation = explainer.explain_instance(
        image, 
        model_predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000,
        segmentation_fn=segmentation_fn  # Use improved segmentation
    )

    # ✅ **Extract important positive regions**
    temp_pos, mask_pos = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=15, hide_rest=False
    )

    # ✅ **Extract both positive & negative regions**
    temp_neg, mask_neg = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=15, hide_rest=False
    )

    return temp_pos, mask_pos, temp_neg, mask_neg

################################################################machine learning part#####################################################################################################################################################

st.set_page_config(page_title="Breast Cancer Classification", layout="wide", page_icon="🔬")

# Title Section
st.title("🎯 Machine Learning vs Deep Learning Classification")
st.markdown(
    """
    Compare **Machine Learning** and **Deep Learning** techniques for breast cancer image classification. 
    Upload an image and select a model type to see predictions and explainability visualizations.
    """
)

# Sidebar for Model Selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose a model type:", ["Machine Learning Model", "Deep Learning Model"])

# Image Upload
uploaded_file = st.file_uploader(
    "📤 Upload an image file (jpg, jpeg, png):",
    type=["jpg", "jpeg", "png"],
)

# Main Content
if uploaded_file is not None:
    # Read and display the uploaded image
    image = imread(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("---")  # Horizontal line for separation

    if model_choice == "Machine Learning Model":
        st.header("🧠 Machine Learning Classification")
        
        # Classification Prediction
        prediction = classify_image(image, ml_model)
        st.success(f"**Prediction:** {prediction}")

        # LIME Explainability
        with st.spinner("🔍 Generating LIME Explainability..."):
            temp_pos, mask_pos, temp_neg, mask_neg = generate_lime_explanation(image)

        # Display LIME Explanation
        st.subheader("LIME Explanation")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Important Regions Only**")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.imshow(mark_boundaries(temp_pos, mask_pos, color=(1, 1, 0), mode="thick"))
            ax1.axis("off")
            st.pyplot(fig1)

        with col2:
            st.markdown("**Positive & Negative Regions**")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(mark_boundaries(temp_neg, mask_neg, color=(1, 1, 0), mode="thick"))
            ax2.axis("off")
            st.pyplot(fig2)

    elif model_choice == "Deep Learning Model":
        st.header("🤖 Deep Learning Classification")
        
        # Classification Prediction
        prediction = DL_classify(image, dl_model)
        st.success(f"**Prediction:** {prediction}")

        # Integrated Gradients Explainability
        with st.spinner("✨ Generating Integrated Gradients Explainability..."):
            X_image = DL_explainability(dl_model, image)

        st.subheader("Integrated Gradients Visualization")
        st.image(X_image, caption="Integrated Gradients Visualization", use_container_width=True)
else:
    st.info("💡 Please upload an image to proceed.")
