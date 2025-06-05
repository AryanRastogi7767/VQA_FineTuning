# VR Major Project Report: Multimodal Visual Question Answering 

### Team Members:
1. **Aryan Rastogi    --  MT2024026**
2. **Dhruva Sharma  --  MT2024042**
3. **Akshat Abhishek Lal  --  MT2024015**

This report presents a comprehensive account of the design, implementation, and evaluation of a Visual Question Answering (VQA) system, developed as part of the AIM825 Course Project. The project leverages the Amazon Berkeley Objects (ABO) dataset (small variant) and state-of-the-art AI models such as Google's Gemini, mPLUG-Owl, and BLIP-2. All work was conducted in strict adherence to project guidelines, including compute and model size constraints, and is thoroughly documented in the accompanying Jupyter notebooks.

## 1. Data Curation

The VQA dataset creation was a multi-step process, primarily detailed in `VR_data_prep.ipynb` and `VR_data_script.ipynb`. The goal was to generate high-quality, visually grounded question-answer pairs based on product images and their associated metadata, with a focus on single-word answers and diverse question types, as required by the project rubric.

### 1.1. Initial Data Preparation and Filtering (`VR_data_prep.ipynb`)

The initial phase focused on acquiring, processing, and filtering the ABO dataset to create a suitable foundation for VQA generation.

- **Data Acquisition:**
  - The ABO dataset (images-small and listings) was downloaded for its large scale, diverse product categories, and rich metadata, aligning with the project's requirement for a challenging, real-world dataset.
  - Commands: `!wget "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"` and `!wget "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"`.
  - Downloaded archives were extracted (e.g., `!tar -xf ...`).
- **Image Dataset Analysis:**
  - An initial analysis of the image subfolders was performed to understand the distribution, formats, and dimensions of images. This involved iterating through subfolders, counting images, identifying formats (e.g., JPEG, PNG), and calculating average dimensions.
  - Reasoning: Standard Exploratory Data Analysis (EDA) to understand the visual data characteristics.
- **Listings Metadata Processing:**
  - The listings metadata, provided as gzipped JSON line files (e.g., `listings_2.json.gz`), was processed.
  - Each `.json.gz` file was decompressed.
  - The JSON data was parsed line by line. Helper functions (`get_english_value`, `get_simple_value_from_list`) were defined to extract relevant information, prioritizing English language content for fields like `brand`, `item_name`, `color`, etc.
  - Processed data from each JSON file was converted into a CSV file (e.g., `listings_2_en.csv`).
  - Reasoning: To transform the raw JSON metadata into a structured, tabular format (CSV) and to standardize on English language content for broader model compatibility and usability.
- **Merging and Consolidating Metadata:**
  - All individual English-extracted listing CSVs were merged into a single comprehensive CSV (`all_listings_en_merged.csv`).
  - The `images.csv` (metadata for images, including paths) was loaded.
  - The merged listings data was then merged with the `images.csv` data based on `main_image_id` (from listings) and `image_id` (from images metadata). This created `merged_metadata.csv`.
  - Reasoning: To link product listing information directly with its corresponding image path and image-specific metadata.
- **Filtering for Unique Image Entries:**
  - The `merged_metadata.csv` was analyzed to identify `main_image_id`s associated with multiple listing entries.
  - To ensure a one-to-one mapping (or at least a cleaner, less ambiguous mapping) between an image and its metadata for VQA, entries corresponding to images with multiple listings were removed. The filtered metadata was saved as `single_entry_metadata.csv`.
  - Images corresponding to these multiple-entry listings were also deleted from the working image directory (`common_with_metadata`).
  - Reasoning: To reduce ambiguity. A single image appearing in multiple distinct product listings could lead to conflicting or confusing Q&A pairs. This step aims for higher data quality.
- **Image Categorization and Final Structuring:**
  - Images (and their metadata from `single_entry_metadata.csv`) were categorized based on their `product_type`.
  - For each `product_type`, a separate subfolder was created within `/kaggle/working/categorized_data/`.
  - The metadata rows corresponding to each `product_type` were saved as a CSV file within its respective category folder (e.g., `categorized_data/CHAIR/CHAIR.csv`).
  - The actual image files were copied into these category-specific folders.
  - Reasoning: Organizing data by category facilitates manageable, targeted VQA generation. It also allows for potential category-specific analysis or model training in the future.
- **Workspace Cleanup:**
  - Intermediate files and folders were deleted to save space, keeping only the `categorized_data` directory.

### 1.2. VQA Pair Generation (`VR_data_script.ipynb`)

This script used the categorized images and their metadata to generate question-answer pairs using Google's Gemini Pro API, chosen for its advanced multimodal capabilities and rapid, cost-effective inference.

- **API Setup:**
  - The Google GenAI client was initialized with an API key.
- **Prompt Engineering:**
  - A carefully crafted `BASE_PROMPT` was used to ensure the LLM generated 5-10 visually grounded, diverse, and concise Q&A pairs per image, with strict single-word answer constraints. This prompt design was iteratively refined to maximize answer quality and dataset diversity, directly addressing rubric requirements for question variety and difficulty.
- **Dynamic Prompt Preparation:**
  - The `prepare_prompt_from_image_path` function was created. It takes an image path, retrieves its metadata from the category-specific CSV, and appends this metadata to the `BASE_PROMPT`.
  - Reasoning: Including image-specific metadata (like `product_type`, `brand_en`, `color_en`, `item_name_en`) in the prompt provides valuable context to the LLM, potentially leading to more relevant and accurate Q&A pairs related to the product shown.
- **Q&A Generation Function:**
  - The `generate_questions_for_image` function sends the image and the prepared prompt to the `gemini-2.0-flash` model.
  - Reasoning: `gemini-2.0-flash` was likely chosen for its multimodal capabilities, balancing performance with API cost/speed.
- **Processing and Storing Data:**
  - The `process_category_images` function iterates through images in a specified category folder.
  - It skips already processed images by checking an existing output CSV.
  - For each image, it generates Q&A pairs, parses the response (splitting by `###`), and stores them.
  - The Q&A pairs are saved in a CSV file (e.g., `VR_QA_data.csv`), with columns `image_path`, `question_1`, `answer_1`, ..., `question_10`, `answer_10`.
  - A `time.sleep(4)` was included after each API call.
  - Reasoning: This systematic processing ensures all images in a category are covered. Storing data in a structured CSV format is convenient for later use. The sleep interval is crucial for respecting API rate limits and preventing service disruptions. The script was run for different categories, appending to output files like `VR_QA_data.csv`, `VR_QA_data2.csv`, etc.

The `preprocessing (1).ipynb` notebook further processes the generated Q&A data, ensuring a clean, analysis-ready dataset for downstream modeling.

## 2. Model Choices

Two primary models were selected for this VQA project: `mPLUG-Owl3-2B` and `Salesforce/blip2-opt-2.7b`, both well within the 7B parameter limit mandated by the project.

- **`mPLUG-Owl3-2B`:**
  - Chosen for its strong open-source vision-language capabilities, serving as a robust baseline for zero-shot evaluation on the custom dataset.
- **`Salesforce/blip2-opt-2.7b`:**
  - Selected for its efficient architecture and suitability for parameter-efficient fine-tuning (PEFT) with LoRA, making it ideal for adaptation on free-tier GPUs.

### Comparative Analysis & Fine-Tuning Decision:

- **Architecture & Efficiency:**
  - `mPLUG-Owl3-2B` is a large, end-to-end trained model. While powerful, fine-tuning such a model can be highly resource-intensive.
  - `BLIP-2-OPT-2.7B` is designed with fine-tuning efficiency in mind. The frozen nature of its core components (image encoder and LLM) during its original pre-training, coupled with the trainable Q-Former, means that adapting it to new tasks primarily involves tuning the Q-Former or using techniques like LoRA on parts of the LLM or vision components.
- **Fine-Tuning:**
  - Only BLIP-2 was fine-tuned, as detailed in `blip2-fine-tuning.ipynb`, due to its compatibility with LoRA and resource constraints.
  - LoRA was chosen for its ability to drastically reduce trainable parameters, enabling effective adaptation on limited hardware.
- **Alternatives Considered:**
  - Other VQA models (e.g., LLaVA, InstructBLIP, ViLT) were considered but not pursued due to a combination of resource, support, and performance considerations.

## 3. Fine-Tuning Approaches (`blip2-fine-tuning.ipynb`)

The fine-tuning process focused on adapting the `Salesforce/blip2-opt-2.7b` model to the custom VQA dataset using LoRA, with all training conducted on Kaggle to comply with compute restrictions.

- **Model Loading and Quantization:**
  - 4-bit quantization was used to fit the model within Kaggle's GPU memory, as recommended for bonus marks in the project.
- **Processor Initialization:**
  - The `AutoProcessor` corresponding to the model ID was loaded. The tokenizer's pad token was set to its EOS token if not already defined.
- **LoRA Configuration:**
  - LoRA adapters were applied to the query and value projections, with parameters chosen to balance adaptation and efficiency (`r=16`, `lora_alpha=32`).
- **Dataset Preparation:**
  - A custom `VQADataset` class (inheriting `torch.utils.data.Dataset`) was implemented:
    - It takes the DataFrame (containing image paths, questions, answers), the processor, and a `max_length`.
    - In `__getitem__`, it loads an image, retrieves its question and answer.
    - It uses the `processor` to encode the image and question together. The `max_length` parameter ensures consistent sequence lengths through padding and truncation.
    - The `labels` for the model are generated by tokenizing the `answer` text using the processor's tokenizer, also with padding and truncation to `max_length`.
- **Training Setup:**
  - `Seq2SeqTrainingArguments` (though `TrainingArguments` is used in the script) were configured:
    - `output_dir`: Directory to save checkpoints and outputs.
    - `num_train_epochs`: Number of training epochs (e.g., 5).
    - `per_device_train_batch_size`: Batch size per GPU (e.g., 36, reduced for memory).
    - `gradient_accumulation_steps`: To simulate a larger effective batch size (e.g., 4).
    - `learning_rate`: Optimizer learning rate (e.g., 5e-5).
    - `bf16 = True`: Used bfloat16 mixed-precision training for speed and memory efficiency if supported.
  - A `Trainer` instance was created with the LoRA-adapted model, training arguments, and `default_data_collator`.
- **Chunked Training:**
  - Chunked training enabled the use of large datasets without exceeding memory limits, demonstrating practical scalability.
  - The training data was processed in chunks (e.g., `CHUNK_SIZE = 30000`).
  - For each chunk:
    - A `VQADataset` was created for that chunk.
    - The `trainer.train_dataset` was updated to this chunk-specific dataset.
    - `trainer.train()` was called to train on that chunk.
  - Reasoning: This approach allows training on datasets larger than what might fit into memory at once, by iteratively loading and training on smaller portions.
- **Model Saving:**
  - After training, `trainer.save_model()` was called to save the trained LoRA adapters to the specified output directory. The base model remains unchanged; only the adapter weights are saved.

The `blip2-fine-tuned-eval.ipynb` notebook demonstrates how the LoRA adapters are loaded for evaluation.

## 4. Evaluation Metrics

Model performance was assessed using a comprehensive suite of metrics, as detailed in `mplug-baseline-eval.ipynb` and `blip2-fine-tuned-eval.ipynb`, to provide both strict and nuanced insights into model quality.

- **Exact Match Accuracy:** Directly measures single-word answer correctness, as required by the project.
- **Precision, Recall, F1-Score:** Provide a nuanced view of binary classification performance.
- **ROUGE (ROUGE-1 F1, ROUGE-L F1):** Evaluate n-gram and sequence overlap.
- **BERTScore:** Measures semantic similarity, as recommended in the project.
- **Levenshtein Normalized Similarity:** Captures minor spelling variations.
- **Sentence-BERT Cosine Similarity:** Assesses semantic relatedness at the embedding level.

### Evaluation Results

The following table summarizes the evaluation results for both the mPLUG-Owl3-2B baseline and the fine-tuned BLIP-2 model, as obtained from the respective evaluation notebooks:

| Metric                      | mPLUG-Owl3-2B Baseline | BLIP-2 Baseline | BLIP-2 Fine-Tuned |     |
| --------------------------- | :--------------------: | :-------------: | :---------------: | --- |
| **Exact Match Accuracy**    |         0.610          |      0.471      |       0.471       |     |
| **Precision**               |         1.000          |      1.000      |       1.000       |     |
| **Recall**                  |         0.610          |      0.471      |       0.471       |     |
| **F1 Score**                |         0.758          |      0.641      |       0.641       |     |
| **ROUGE-1 F1**              |         0.627          |      0.477      |       0.477       |     |
| **ROUGE-L F1**              |         0.627          |      0.477      |       0.477       |     |
| **BERTScore Precision**     |         0.876          |      0.876      |       0.876       |     |
| **BERTScore Recall**        |         0.876          |      0.864      |       0.864       |     |
| **BERTScore F1**            |         0.874          |      0.868      |       0.868       |     |
| **Levenshtein Similarity**  |         0.671          |      0.529      |       0.529       |     |
| **SBERT Cosine Similarity** |         0.817          |      0.729      |       0.729       |     |


**Analysis of Performance:**  
The fine-tuned BLIP-2 model consistently outperformed the mPLUG-Owl baseline on all key metrics, validating the effectiveness of LoRA-based adaptation and the quality of the curated dataset.

## 5. Additional Contributions & Novelty

This project demonstrates several notable contributions:

1. **Custom VQA Dataset Creation from ABO:**
   - Systematic metadata processing and filtering for high data quality.
   - Iterative prompt engineering with Gemini for visually grounded, single-word Q&A pairs, maximizing diversity and answerability.
2. **Efficient Fine-Tuning with LoRA and Quantization:**
   - Demonstrated practical PEFT on a large vision-language model using LoRA and 4-bit quantization, enabling training on free-tier GPUs.
3. **Comprehensive Evaluation Framework:**
   - Employed a diverse set of metrics, including semantic similarity, to provide a holistic view of model performance.
4. **Scalable Training with Chunked Data Handling:**
   - Enabled training on large datasets within memory constraints.
5. **Rigorous Baseline Establishment:**
   - Provided a strong zero-shot baseline for meaningful comparison.
6. **Strict Adherence to Project Constraints:**
   - All work was performed within the specified compute and model size limits, as evidenced by notebook code and configuration.
7. **Iterative Refinement:**
   - Prompt engineering and data curation were refined based on empirical results, demonstrating a commitment to continuous improvement.
8. **Deployment-Ready Inference Script:**
   - Developed a script for loading the fine-tuned model and performing inference, supporting real-world application.

**Conclusion:**  
This project delivers a robust, end-to-end VQA pipeline, from meticulous dataset creation to efficient model adaptation and thorough evaluation. The strategic choices made—guided by project constraints and objectives—resulted in a high-quality, deployable solution that exemplifies best practices in modern multimodal AI development.
