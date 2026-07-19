from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        medinc = float(request.form["MedInc"])
        houseage = float(request.form["HouseAge"])
        averooms = float(request.form["AveRooms"])
        avebedrooms = float(request.form["AveBedrms"])
        population = float(request.form["Population"])
        aveoccup = float(request.form["AveOccup"])
        latitude = float(request.form["Latitude"])
        longitude = float(request.form["Longitude"])

        features = [[
            medinc,
            houseage,
            averooms,
            avebedrooms,
            population,
            aveoccup,
            latitude,
            longitude
        ]]

        prediction = model.predict(features)[0]

        predicted_price = prediction * 100000

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ${predicted_price:,.2f}"
        )

    except ValueError:
        return render_template(
            "index.html",
            prediction_text="❌ Please enter valid numeric values."
        )

if __name__ == "__main__":
    app.run(debug=True)
    
    