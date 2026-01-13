# B. README.md

## Telco Customer Churn – MLOps-Ready ML Pipeline

### 1. What I chose to implement and why

I chose to extend the IBM Telco Customer Churn project by improving the model and implementing a full ML lifecycle workflow suitable for production. The original project provided the dataset, preprocessing, and baseline model. I enhanced it by adding multiple models including a Decision Tree, an advanced XGBoost classifier, and a Neural Network. I also implemented model comparison, a model registry, and a conditional deployment logic to promote only the best-performing model. Finally, I wrapped the production model in a FastAPI inference API. This approach was selected to demonstrate real-world ML engineering practices and align with modern MLOps workflows.

### 2. How to run the code

1. **Install required packages**:

   ```bash
   pip install pandas numpy scikit-learn xgboost joblib fastapi uvicorn seaborn matplotlib
   ```
2. **Prepare the dataset**:

   * Place `telco_churn.csv` in a folder named `data/`.
3. **Run the notebook/script step by step**:

   * Preprocessing, EDA, feature engineering, and model training are included in the notebook.
4. **Compare models and select the best**:

   * The notebook automatically evaluates Decision Tree, XGBoost, and Neural Network models and updates the production model in `models/production_model.pkl`.
5. **Deploy the production model**:

   * Save the FastAPI deployment code as `deploy.py` and run:

     ```bash
     uvicorn deploy:app --host 0.0.0.0 --port 8000
     ```
   * Use ngrok if running in Colab to expose a public endpoint.

### 3. Assumptions or limitations

* The model registry uses a local JSON file for simplicity.
* Only F1-score is used to determine the best model.
* Some preprocessing assumes consistent categorical values in the dataset.
* The FastAPI deployment does not include authentication or logging for simplicity.
* The project is designed for demonstration and educational purposes; full-scale production deployment would require CI/CD and cloud integration.

---

# C. Reflection

During this project, using the coding assistant greatly helped me move faster by providing structured templates for preprocessing, modeling, and deployment. It suggested approaches for multi-model comparison, model registry, and FastAPI integration that I could adapt directly. Occasionally, some suggestions, such as handling categorical features, required adjustments because the dataset had nuances not fully captured in the generic code. The assistant was most useful in giving end-to-end workflow guidance and clean code structuring, and least useful when fine-tuning model hyperparameters, which still required my domain knowledge. Overall, it streamlined the workflow, helped me avoid common pitfalls, and allowed me to focus on applying MLOps principles and explaining decisions in a professional manner. This experience reinforced the importance of combining automated guidance with human oversight in ML engineering tasks.
