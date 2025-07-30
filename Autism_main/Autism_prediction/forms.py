from django import forms

class AutismForm(forms.Form):
    A1_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A2_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A3_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A4_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A5_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A6_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A7_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A8_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A9_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    A10_Score = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    age = forms.FloatField()
    jaundice = forms.ChoiceField(choices=[(1, 'Yes'), (0, 'No')])
    gender = forms.ChoiceField(choices=[(1, 'Male'), (0, 'Female')])
    ethnicity = forms.IntegerField(min_value=0, max_value=20)
