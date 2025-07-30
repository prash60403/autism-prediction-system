from django.shortcuts import render
from .forms import AutismForm
from .models import AutismScreening
import pickle
import numpy as np

# Load model and scaler
with open('autism_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def index(request):
    if request.method == 'POST':
        form = AutismForm(request.POST)
        if form.is_valid():
            cleaned = form.cleaned_data

            # Sum score and index features
            sum_score = sum(int(cleaned[f"A{i}_Score"]) for i in range(1, 11))
            ind = int(cleaned['jaundice']) + int(cleaned['gender']) + int(cleaned['ethnicity'])

            features = [
                int(cleaned[f"A{i}_Score"]) for i in range(1, 11)
            ] + [
                float(cleaned['age']),
                int(cleaned['jaundice']),
                int(cleaned['gender']),
                int(cleaned['ethnicity']),
                sum_score,
                ind
            ]

            # Prepare for prediction
            features = np.array(features).reshape(1, -1)
            scaled_features = scaler.transform(features)
            pred = model.predict(scaled_features)[0]
            result_text = "ASD Positive" if pred == 1 else "ASD Negative"

            # Save to DB
            AutismScreening.objects.create(
                **{f"A{i}_Score": cleaned[f"A{i}_Score"] for i in range(1, 11)},
                age=cleaned['age'],
                jaundice=cleaned['jaundice'],
                gender=cleaned['gender'],
                ethnicity=cleaned['ethnicity'],
                sum_score=sum_score,
                ind=ind,
                prediction=result_text
            )

            return render(request, 'result.html', {'result': result_text})
    else:
        form = AutismForm()

    return render(request, 'index.html', {'form': form})
