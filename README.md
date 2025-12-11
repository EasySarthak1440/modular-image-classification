
---

# **Modular Image Classification Pipeline (PyTorch)**

## **1️⃣ Project Overview**

A modular and flexible image classification pipeline built with PyTorch, supporting:

* Custom CNN or other architectures
* Custom callback for learning rate adjustment and early stopping
* Automatic best model saving
* Interactive prompts to continue or halt training
* Layer freezing for transfer learning scenarios
* Logging metrics to CSV

---

## **2️⃣ Project Structure**

| Module               | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `data_module.py`     | Loads and preprocesses datasets (CIFAR-10 by default)                  |
| `model_module.py`    | Defines model architecture; supports layer freezing                    |
| `callback_module.py` | Custom callback (`LRA`) for dynamic LR adjustment & early stopping     |
| `train_module.py`    | Training loop integrating callback, CSV logging, and best model saving |
| `utils.py`           | Utilities for saving metrics and models                                |
| `main.py`            | Entry point to run training                                            |
| `requirements.txt`   | Required packages                                                      |
| `README.md`          | Instructions for usage                                                 |

---

## **3️⃣ Execution Steps**

1. **Clone the repository**

   ```bash
   git clone <repo_url>
   cd modular_image_classification
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run training**

   ```bash
   python main.py
   ```

4. **Training behavior**

   * The custom callback automatically reduces the learning rate if validation loss does not improve.
   * Best model weights are restored if `dwell=True`.
   * Interactive prompts appear after `ask_epoch` epochs.
   * Metrics (train/val loss & accuracy) are saved to `training_metrics.csv`.
   * Best model is saved to `best_model.pth`.

5. **Optional settings**

   * Freeze layers: `get_model(freeze=True)`
   * Modify learning rate, patience, threshold in `LRA` callback
   * Adjust batch size, epochs in `main.py`

---

## **4️⃣ Visual Workflow**

```
+------------------+       +--------------------+
|  Data Module     | --->  |  Model Module       |
| (Load & Transform)|       | (Define CNN/Freeze)|
+------------------+       +--------------------+
           |                        |
           v                        v
      +-----------------------------------+
      |        Train Module               |
      | - Training Loop                   |
      | - Validation                      |
      | - Metrics Logging (CSV)           |
      | - Callback for LR Adjustment      |
      +-----------------------------------+
                        |
                        v
            +-----------------------+
            | Callback Module (LRA) |
            | - Reduce LR           |
            | - Restore Best Weights|
            | - Early Stop          |
            +-----------------------+
                        |
                        v
                Best Model Saved
                 `best_model.pth`
```

---

## **5️⃣ Notes**

* Fully GPU compatible (uses CUDA if available)
* Easy to swap architectures or datasets
* Modular design encourages code reuse and clarity

---


