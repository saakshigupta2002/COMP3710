{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1v2jByvIpcLhZ7P5T1SObGxzageUQ4_eU",
      "authorship_tag": "ABX9TyOgkwlrKIoOif9LuEE0ILXM",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/saakshigupta2002/COMP3710/blob/main/DEMO3/train.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6YlbG8DPi-Ey"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "from module import build_alzheimer_model\n",
        "from dataset import load_and_preprocess_data\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Define hyperparameters\n",
        "input_shape = (224, 224, 3)  # Adjust image size as needed\n",
        "epochs = 15\n",
        "model_save_path = 'alzheimer_model.h5'\n",
        "\n",
        "# Load and preprocess the data\n",
        "train_data_gen, test_data_gen = load_and_preprocess_data(img_height=224, img_width=224, batch_size=16)\n",
        "\n",
        "# Build the Alzheimer's disease classification model\n",
        "model = build_alzheimer_model(input_shape)\n",
        "\n",
        "# Compile the model\n",
        "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "# Train the model\n",
        "history = model.fit(train_data_gen, epochs=epochs, validation_data=test_data_gen)\n",
        "\n",
        "# Save the trained model\n",
        "model.save(model_save_path)\n",
        "\n",
        "# Plot training history (loss and accuracy)\n",
        "plt.plot(history.history['loss'], label='Training Loss')\n",
        "plt.plot(history.history['val_loss'], label='Validation Loss')\n",
        "plt.legend()\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Loss')\n",
        "plt.show()\n",
        "\n",
        "plt.plot(history.history['accuracy'], label='Training Accuracy')\n",
        "plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
        "plt.legend()\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.show()\n"
      ]
    }
  ]
}