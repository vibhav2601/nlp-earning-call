# 🧠 FinBERT Tone Classification on Earnings Call Transcripts

This repository implements a binary classification model using **FinBERT** to analyze the tone of earnings call transcripts and predict market sentiment (0 = negative/neutral, 1 = positive).

---

## 🔍 Overview

- **Task**: Binary sentiment classification on earnings call transcripts  
- **Model**: `ProsusAI/finbert` from Hugging Face  
- **Loss**: Weighted `BCEWithLogitsLoss` to address class imbalance  
- **Data**: Custom cleaned JSONL from earnings calls; optional weak labels  

---


---

## ⚙️ Setup & Requirements

```bash
pip install req2.txt
pip install jsonlines

```
## 🏃‍♂️ How to Run
```bash
python generate_gold_labels.py
python train.py
```

## ✅Accuracy
![image](https://github.com/user-attachments/assets/8eefd1f2-02fe-46f7-9341-99487ba35f4e)


# 📚 Citation
We used ProsusAI/FinBERT and custom cleaned transcripts from the lamini earnings-calls-qa dataset (transcripts only).




