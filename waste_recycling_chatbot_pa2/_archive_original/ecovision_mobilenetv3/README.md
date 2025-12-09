---
license: mit
tags:
- image-classification
- garbage-classification
- pytorch
datasets:
- garbage-classification-v2
- garbage-classification
metrics:
- accuracy
---

# Ecovision MobilenetV3

This model is a fine-tuned MobileNetV3 Large model for garbage classification. It has been trained on the Garbage Classification V2 and Garbage Classification datasets and achieves an overall accuracy of around 95% on the test set.

## Model Description

This model is based on the MobileNetV3 Large architecture and has been fine-tuned for garbage classification. It takes an image as input and outputs the predicted class label.

The model was trained using PyTorch and the following hyperparameters:

* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 32
* Number of epochs: 10
* dataset training on kaggle : [sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) 
* dataset test on kaggle : [mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

## Intended Uses & Limitations

This model is intended to be used for classifying garbage into different categories. It can be used for a variety of applications, such as waste sorting and recycling.

However, it is important to note that the model is limited to the classes it was trained on. It may not be able to accurately classify images of garbage that are not in those classes. Additionally, the model's accuracy may vary depending on the quality of the input images.

## Training and Evaluation Data


### Training Procedure

The following steps were followed to train the model:

1. Load the Garbage Classification V2 and Garbage Classification datasets.
2. Preprocess the images by resizing them and normalizing their pixel values.
3. Split the dataset into training and validation sets.
4. Load the MobileNetV3 Large model and modify the classifier to output ten classes.
5. Train the model using the Adam optimizer and cross-entropy loss function.
6. Evaluate the model on the validation set and save the best model weights.
7. Test the model on the test set and report the results.

### Training Logs
The following table summarizes the training process:

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|------------|----------------|----------|--------------|
| 1     | 0.5056     | 83.59%         | 0.3849   | 87.71%       |
| 2     | 0.3289     | 89.43%         | 0.3087   | 90.47%       |
| 3     | 0.2692     | 91.24%         | 0.3045   | 90.27%       |
| 4     | 0.1377     | 95.39%         | 0.1322   | 95.55%       |
| 5     | 0.0860     | 97.03%         | 0.1148   | 96.03%       |
| 6     | 0.0677     | 97.72%         | 0.1184   | 96.16%       |
| 7     | 0.0540     | 98.03%         | 0.1150   | 96.39%       |
| 8     | 0.0505     | 98.34%         | 0.1122   | 96.49%       |
| 9     | 0.0470     | 98.41%         | 0.1112   | 96.44%       |
| 10    | 0.0507     | 98.30%         | 0.1110   | 96.46%       |

### Dataset Overview
The model was trained using the [sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) dataset and evaluated on the [mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) dataset. These datasets collectively contain approximately 19,762 images of garbage classified into ten categories:

| Class | Accuracy |
|---|---|
| battery | 99.36% |
| biological | 98.27% |
| cardboard | 97.64% |
| clothes | 99,34% |
| glass | 98,01% |
| metal | 96,87% |
| paper | 96,38% |
| plastic | 91,79% |
| shoes | 99,54% |
| trash | 98,99% |


## How to Use

You can explore and interact with the **EcoVision MobileNetV3** model directly on [Hugging Face Spaces](https://huggingface.co/spaces/AmadFR/ecovision_mobilenetv3) where it is hosted via an intuitive **Gradio** interface.

![Gradio Example](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/Gradio_Exemple.png)

## Results

The Ecovision MobileNetV3 model demonstrated strong performance in classifying garbage, achieving an overall accuracy of approximately 95% on the test dataset. Detailed performance metrics are outlined below:

Overall Performance
* Total Correct Predictions: 15,247
* Total Incorrect Predictions: 268
* Overall Accuracy: ~95%

Per-Class Performance
The table below summarizes the model's performance for each class, showing the number of correct and incorrect predictions:
| **Class**       | **Correct Predictions** | **Incorrect Predictions** | **Example Highest**                                                                                 | **Example Lowest**                                                                                 |
|------------------|--------------------------|----------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Battery**      | 939                      | 6                          | ![Example Battery Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/batt1.png) | ![Example Battery Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/batt2.png) |
| **Biological**   | 968                      | 17                         | ![Example Biological Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/biologi1.png) | ![Example Biological Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/biologi2.png) |
| **Cardboard**    | 870                      | 21                         | ![Example Cardboard Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/cardboard1.png) | ![Example Cardboard Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/cardboard2.png) |
| **Clothes**      | 5,290                    | 35                         | ![Example Clothes Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/cloth1.png) | ![Example Clothes Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/cloth2.png) |
| **Glass**        | 1,971                    | 40                         | ![Example Glass Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/glass1.png) | ![Example Glass Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/glass2.png) |
| **Metal**        | 745                      | 24                         | ![Example Metal Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/metal1.png) | ![Example Metal Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/metal2.png) |
| **Paper**        | 1,012                    | 38                         | ![Example Paper Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/paper1.png) | ![Example Paper Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/paper2.png) |
| **Plastic**      | 794                      | 71                         | ![Example Plastic Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/plastic1.png) | ![Example Plastic Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/plastic2.png) |
| **Shoes**        | 1,968                    | 9                          | ![Example Shoes Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/shoes1.png) | ![Example Shoes Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/shoes2.png) |
| **Trash**        | 690                      | 7                          | ![Example Trash Best](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/trash1.png) | ![Example Trash Worst](https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/main/Image/trash2.png) |

## Observations
* The Clothes and Glass categories achieved high accuracy due to their distinct features, resulting in fewer misclassifications.
* The Plastic category had the highest number of incorrect predictions, likely due to variability in plastic item appearances and similarities with other materials.
* The model's accuracy across all classes indicates its robustness and potential for practical waste classification applications.


## Limitations

The model is limited to the classes it was trained on. It may not be able to accurately classify images of garbage that are not in those classes. Additionally, the model's accuracy may vary depending on the quality of the input images.

## Future Work

Future work could focus on improving the model's accuracy by training it on a larger and more diverse dataset. Additionally, the model could be adapted to classify other types of objects.