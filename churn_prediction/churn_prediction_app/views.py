import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from .forms import ChurnPredictionForm

# Load Models
log_reg_model = joblib.load('churn_prediction_app/ml_models/churn_logistic_regression_model_for_deployment.pkl')
xgb_model = joblib.load('churn_prediction_app/ml_models/churn_xgb_model_for_deployment.pkl')

# Load Zip Code Data
zip_code_map_df = pd.read_csv("churn_prediction_app/data/zip_code_map_df.csv", index_col='Zip Code')

# Prediction Function
def predict_churn(request):
    if request.method == 'POST':
        form = ChurnPredictionForm(request.POST)
        
        if form.is_valid():
            # Extract form data
            zip_code = int(form.cleaned_data['zip_code'])
            age = int(form.cleaned_data['age'])
            tenure = int(form.cleaned_data['tenure'])
            contract = form.cleaned_data['contract']
            internet_service = form.cleaned_data['internet_service']
            offer = form.cleaned_data['offer']
            payment_method = form.cleaned_data['payment_method']
            online_security = int(form.cleaned_data['online_security'])
            tech_support = int(form.cleaned_data['tech_support'])
            paperless_billing = int(form.cleaned_data['paperless_billing'])
            married = int(form.cleaned_data['married'])
            referred_a_friend = int(form.cleaned_data['referred_a_friend'])
            total_revenue = float(form.cleaned_data['total_revenue'])
            satisfaction_score = int(form.cleaned_data['satisfaction_score'])

            # Handle zip code validation
            if zip_code not in zip_code_map_df.index:
                zip_code = 90201  # Default to max population zip code if invalid

            # Zip code related data
            latitude = zip_code_map_df.loc[zip_code]['Latitude']
            longitude = zip_code_map_df.loc[zip_code]['Longitude']
            population = zip_code_map_df.loc[zip_code]['Population']

            # Derived features
            senior_citizen = int(age > 65)
            under_30 = int(age < 30)
            tenure_bins_0_12 = int(0 < tenure <= 12)
            tenure_bins_12_24 = int(12 < tenure <= 24)
            tenure_bins_24_48 = int(24 < tenure <= 48)
            tenure_bins_48_60 = int(48 < tenure <= 60)
            tenure_bins_60_72 = int(tenure > 60)
            
            # One-hot encoding for categorical fields
            offer_one_hot = [
                int(offer == "Offer A"),
                int(offer == "Offer B"),
                int(offer == "Offer C"),
                int(offer == "Offer D"),
                int(offer == "Offer E")
            ]
            
            payment_method_one_hot = [
                int(payment_method == "Bank transfer (automatic)"),
                int(payment_method == "Credit card (automatic)"),
                int(payment_method == "Electronic check"),
                int(payment_method == "Mailed check")
            ]

            contract_one_hot = [
                int(contract == "Month-to-month"),
                int(contract == "One year"),
                int(contract == "Two year")
            ]

            internet_service_one_hot = [
                int(internet_service == "DSL"),
                int(internet_service == "Fiber Optic"),
                int(internet_service == "No")
            ]

            # Feature array for the model
            data = [
                latitude, longitude, senior_citizen, online_security, tech_support, paperless_billing,
                age, under_30, married, population, referred_a_friend, total_revenue, satisfaction_score,
                tenure_bins_0_12, tenure_bins_12_24, tenure_bins_24_48, tenure_bins_48_60, tenure_bins_60_72,
                *offer_one_hot, *payment_method_one_hot, *contract_one_hot, *internet_service_one_hot
            ]

            # Convert to 2D array for model input
            data = np.array(data).reshape(1, -1)

            # Make predictions
            output_probab_log_reg = log_reg_model.predict_proba(data)[0][1]
            output_probab_xgb = xgb_model.predict_proba(data)[0][1]

            # Average the predictions from both models
            api_output_probab = (output_probab_xgb + output_probab_log_reg) / 2.0
            pred = "Churn" if api_output_probab > 0.4 else "Not Churn"

            # Render the result template
            return render(request, 'churn_prediction_app/result.html', {
                'form': form,
                'prediction': pred,
                'probability': round(api_output_probab, 4)
            })

    # Render form on GET request
    else:
        form = ChurnPredictionForm()
    
    return render(request, 'churn_prediction_app/predict_churn.html', {'form': form})
