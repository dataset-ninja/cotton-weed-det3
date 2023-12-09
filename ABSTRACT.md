The authors of the **CottonWeedDet3: A 3-Class Weed Detection Dataset for Cotton Cropping Systems** delve into the application of alternative non-chemical or chemical-reduced methods for weed control, emphasizing their significance, particularly in addressing herbicide-resistant weeds. With a focus on machine vision technology, specifically weed detection and localization, the authors aim to facilitate site-and species-specific treatments for individual weed plants. However, the complexities of unstructured field circumstances and the biological variability of weeds pose challenges for achieving robust and accurate weed detection.

Weeds pose a substantial threat to crop yield, competing with crops for vital resources such as water, nutrients, light, and space. This competition impedes crop growth and development, resulting in estimated global crop yield losses of 43%. Current strategies, primarily herbicide application, while widely adopted, face challenges such as increased herbicide-resistant weeds, negative environmental impacts, and rising management costs.

<img src="https://github.com/dataset-ninja/cotton-weed-det3/assets/78355358/61ebc455-49e1-4c73-b30f-3c621e02588e" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Weed detection pipeline.</span>

Detecting weeds robustly and accurately in natural field conditions remains a challenging task due to various factors. These factors include intra- and inter-species variability in weed appearance characteristics, the similarity of weeds with crops, and variations in field light conditions and soil backgrounds.

## **Image Acquisition**

RGB images of weeds in cotton fields were acquired using either a smartphone camera or a handheld digital camera. The average dimension of the captured images was 4442 × 4335 pixels, saved in .jpeg format. To ensure data diversity, images were captured from different view angles under natural field light conditions across the U.S. cotton belt states, primarily in North Carolina and Mississippi, during the growth seasons of 2020 and 2021. A total of 5187 images of fifteen common weed species were collected during this period.

## **Data Pre-processing**

Acquired images were meticulously annotated for weed plants using the [SuperAnnotate platform](https://www.superannotate.com/), allowing the definition of weed instances through bounding boxes. The annotations were exported in the COCO dataset format and converted to the VIA annotation format, a popular and free annotation tool. The resultant annotation files include information about the source image file name, bounding box coordinates, and associated weed class.

<img src="https://github.com/dataset-ninja/cotton-weed-det3/assets/78355358/d0388892-c7a0-42a0-a810-53444c617bd0" alt="image" width="600">

Efforts were made to ensure data quality through the removal of low-quality annotations. Annotations that were too small, out-of-focus, or mislabeled were discarded. The maximum number of bounding boxes in a single image was limited to 10 to maintain data standardization. The cleaned dataset includes 848 images and 1532 bounding boxes.

There is no predefined splitting: the dataset was randomly divided into train, validation, and test sets in a ratio of 60%: 20%: 20%, ensuring stratified partitioning to maintain similar class proportions in each split.
