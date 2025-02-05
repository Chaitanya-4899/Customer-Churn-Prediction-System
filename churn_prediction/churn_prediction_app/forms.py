from django import forms

class ChurnPredictionForm(forms.Form):
    # Numeric fields
    age = forms.IntegerField(label='Age', widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter age'}))
    zip_code = forms.CharField(label='Zip Code', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter zip code'}))
    tenure = forms.IntegerField(label='Tenure in Months', widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter tenure'}))
    total_revenue = forms.FloatField(label='Total Revenue in USD $', widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter total revenue'}))
    satisfaction_score = forms.ChoiceField(
        label='Customer Satisfaction Score',
        choices=[('1', '1 (Very Dissatisfied)'), ('2', '2 (Dissatisfied)'), ('3', '3 (Neutral)'), ('4', '4 (Satisfied)'), ('5', '5 (Very Satisfied)')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Radio fields
    married = forms.ChoiceField(
        label='Is Married',
        choices=[('1', 'Yes'), ('0', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    tech_support = forms.ChoiceField(
        label='Tech Support Available',
        choices=[('1', 'Yes'), ('0', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    online_security = forms.ChoiceField(
        label='Online Security Available',
        choices=[('1', 'Yes'), ('0', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    paperless_billing = forms.ChoiceField(
        label='Paperless Billing',
        choices=[('1', 'Yes'), ('0', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    referred_a_friend = forms.ChoiceField(
        label='Referred a Friend',
        choices=[('1', 'Yes'), ('0', 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    
    # Dropdown fields
    contract = forms.ChoiceField(
        label='Contract Type',
        choices=[
            ('Month-to-month', 'Month-to-month'),
            ('One year', 'One year'),
            ('Two year', 'Two year')
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    internet_service = forms.ChoiceField(
        label='Type of Internet Service',
        choices=[
            ('DSL', 'DSL'),
            ('Fiber Optic', 'Fiber Optic'),
            ('No', 'No')
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    offer = forms.ChoiceField(
        label='Offer Available',
        choices=[
            ('Offer A', 'Offer A'),
            ('Offer B', 'Offer B'),
            ('Offer C', 'Offer C'),
            ('Offer D', 'Offer D'),
            ('Offer E', 'Offer E')
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    payment_method = forms.ChoiceField(
        label='Payment Method',
        choices=[
            ('Bank transfer (automatic)', 'Bank Transfer (automatic)'),
            ('Credit card (automatic)', 'Credit Card (automatic)'),
            ('Electronic check', 'Electronic Check'),
            ('Mailed check', 'Mailed Check')
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
