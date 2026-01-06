# Training Scripts

## סקירה כללית

4 סקריפטים עיקריים לאימון YOLOv11 על VisDrone:

---

## 1. train_baseline.py

**מטרה:** בדיקה ראשונית שהכל עובד

**מתי להריץ:** פעם ראשונה, לפני Optuna

**מה עושה:**
- אימון קצר (10 epochs)
- פרמטרים default
- בדיקה שהdata טוב

**Output:** `runs/baseline/`

**זמן:** ~30 דקות

---

## 2. train_optuna.py

**מטרה:** חיפוש hyperparameters אוטומטי

**מתי להריץ:** אחרי baseline עובד

**מה עושה:**
- 20 trials
- כל trial = 30 epochs
- Optuna מחפש lr, freeze, batch
- WandB logging*****

**Output:** 
- `runs/optuna/`
- `optuna_results.csv`
- Best params בconsole

**זמן:** ~15-20 שעות

---

## 3. train_final.py

**מטרה:** אימון סופי עם best params

**מתי להריץ:** אחרי Optuna מצא פרמטרים

**מה עושה:**
- אימון אחד ארוך (100 epochs)
- פרמטרים מOptuna
- מודל production

**Output:** `runs/final/best.pt`

**זמן:** ~5-8 שעות

---

## 4. evaluate.py

**מטרה:** הערכת ביצועים

**מתי להריץ:** אחרי אימון

**מה עושה:**
- Validation metrics (mAP, Precision, Recall)
- Confusion matrix
- Per-class performance
- Test על MATRIX

**Output:** דוח ביצועים + plots

**זמן:** ~10 דקות

---

## תהליך עבודה מומלץ
```
1. train_baseline.py    → בדוק שעובד
2. train_optuna.py      → מצא פרמטרים
3. train_final.py       → אמן מודל סופי
4. evaluate.py          → הערך ביצועים
```

---

## דרישות

- Python 3.8+
- ultralytics
- optuna
- wandb*********
- GPU מומלץ

---

## הערות

- כל הסקריפטים משתמשים ב-`configs/training_config.yaml`
- outputs ב-`runs/` (לא מועלה לGit)
- Models נשמרים ב-`models/trained/`