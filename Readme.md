# 🌸 Fine-Tuning VGG16 for Fine-Grained Flower Classification - A Transfer Learning Approach

This project fine-tunes VGG16, a deep convolutional neural network, to classify images from the Flowers102 dataset. The Flowers102 dataset features 102 categories of flowers, making it a popular benchmark for fine-grained image classification tasks. This project leverages transfer learning to achieve high accuracy with limited training data.

🚀 **Features**

    ✅ Fine-tunes VGG16 using transfer learning.
    ✅ Uses the Flowers102 dataset with 102 flower species.
    ✅ Downloads pretrained weights via gdown.
    ✅ Provides a ready-to-use classifier.py for inference on new images.

📦 **Setup**

    #clone the repo
    git clone https://github.com/kandarpa02/Fine-Tuning-VGG16-for-Fine-Grained-Flower-Classification-A-Transfer-Learning-Approach.git

    #install depenencies
    cd requirements
    pip install -r requirements.txt

    #download data and the weights
    cd classifier
    python data_dwnld.py
    python download_weights.py

    #run inference
    # batch_index is a number that picks a specific batch from the training data
    python classifier.py <batch_index>


🧩 **Dataset - Flowers102**

    📚 102 Classes — Each representing a different flower species.
    🖼️ 8189 Images Total
    📥 Official Source: [Flowers102-Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)


🏗️ **Model Architecture - VGG16**
Key Modifications

    ✅ Pretrained on ImageNet.
    ✅ Fully connected layer adjusted to output 102 flower classes.
    ✅ Initial layers frozen to preserve pretrained features, while the classifier head is fine-tuned.


**Final Thoughts:**

    The model performs pretty well, I shows 74% accuracy with very less amount of training data, although I think by tuning hyper-parameters
    the performance can be increased.


If you found this project helpful, please star this repository :) ⭐


