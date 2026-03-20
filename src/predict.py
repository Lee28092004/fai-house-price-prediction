import joblib
import pandas as pd


def main():
    model = joblib.load("models/saved_model.pkl")

    sample_data = pd.DataFrame([{
        "Bedroom": 3,
        "Bathroom": 2,
        "Property Size": 1200,
        "Category": "Sale",
        "Tenure Type": "Freehold",
        "Completion Year": 2020,
        "Property Type": "Apartment",
        "Parking Lot": 1
    }])

    prediction = model.predict(sample_data)
    print(f"Predicted House Price: RM {prediction[0]:,.2f}")


if __name__ == "__main__":
    main()