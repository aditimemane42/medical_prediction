from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "xgboost.pkl"

model = joblib.load(MODEL_PATH)


# ==============================
# HTML + CSS DESIGN
# ==============================
HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Insurance Cost Predictor</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 30px;
        }

        .navbar {
            width: 100%;
            max-width: 1100px;
            margin: auto;
            background: white;
            padding: 18px 30px;
            border-radius: 15px;

            display: flex;
            justify-content: space-between;
            align-items: center;

            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }

        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #5b4bc4;
        }

        .badge {
            background: #eeeaff;
            color: #5b4bc4;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
        }

        .hero {
            max-width: 900px;
            margin: 55px auto 30px;
            text-align: center;
            color: white;
        }

        .hero h1 {
            font-size: 48px;
            margin-bottom: 15px;
        }

        .hero p {
            font-size: 17px;
            line-height: 1.6;
            opacity: 0.9;
        }

        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 40px;
            border-radius: 25px;

            box-shadow: 0 20px 50px rgba(0,0,0,0.2);
        }

        .title {
            margin-bottom: 25px;
        }

        .title h2 {
            color: #222;
            margin-bottom: 7px;
        }

        .title p {
            color: #777;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
        }

        .input-box label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }

        input,
        select {
            width: 100%;
            padding: 14px;
            border: 1px solid #ddd;
            border-radius: 10px;
            font-size: 15px;
            outline: none;
            background: #fafafa;
        }

        input:focus,
        select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.15);
        }

        .button {
            grid-column: 1 / -1;
        }

        button {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 12px;

            background: linear-gradient(
                90deg,
                #667eea,
                #764ba2
            );

            color: white;
            font-size: 17px;
            font-weight: bold;

            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102,126,234,0.35);
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            text-align: center;

            background: #ecfdf5;
            border: 1px solid #bbf7d0;

            border-radius: 15px;
        }

        .result-title {
            color: #15803d;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .result-value {
            font-size: 35px;
            font-weight: bold;
            color: #166534;
        }

        .error {
            margin-top: 20px;
            padding: 15px;

            background: #fee2e2;
            color: #b91c1c;

            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }

        .info {
            margin-top: 25px;
            padding: 15px;

            background: #f8fafc;
            border-radius: 10px;

            color: #64748b;
            font-size: 13px;
            line-height: 1.6;
        }

        footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            font-size: 13px;
        }

        @media(max-width: 700px) {

            body {
                padding: 15px;
            }

            .hero h1 {
                font-size: 35px;
            }

            .container {
                padding: 25px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .button {
                grid-column: auto;
            }

            .navbar {
                padding: 15px;
            }

            .logo {
                font-size: 19px;
            }

        }

    </style>

</head>


<body>


    <!-- NAVBAR -->

    <div class="navbar">

        <div class="logo">
            XGBoost Predictor
        </div>

        <div class="badge">
            Machine Learning
        </div>

    </div>


    <!-- HERO -->

    <div class="hero">

        <h1>
            Insurance Cost Predictor
        </h1>

        <p>
            Enter customer information below and use the
            trained XGBoost machine learning model to
            predict the insurance cost.
        </p>

    </div>


    <!-- FORM -->

    <div class="container">

        <div class="title">

            <h2>
                Customer Information
            </h2>

            <p>
                Enter all details to generate prediction.
            </p>

        </div>


        <form method="POST">

            <div class="form-grid">


                <!-- AGE -->

                <div class="input-box">

                    <label>
                        Age
                    </label>

                    <input
                        type="number"
                        name="age"
                        placeholder="Enter age"
                        min="1"
                        max="120"
                        value="{{ values.age }}"
                        required
                    >

                </div>


                <!-- SEX -->

                <div class="input-box">

                    <label>
                        Sex
                    </label>

                    <select name="sex" required>

                        <option value="">
                            Select Sex
                        </option>

                        <option
                            value="0"
                            {% if values.sex == "0" %}
                            selected
                            {% endif %}
                        >
                            Female
                        </option>

                        <option
                            value="1"
                            {% if values.sex == "1" %}
                            selected
                            {% endif %}
                        >
                            Male
                        </option>

                    </select>

                </div>


                <!-- BMI -->

                <div class="input-box">

                    <label>
                        BMI
                    </label>

                    <input
                        type="number"
                        name="bmi"
                        step="0.1"
                        placeholder="Enter BMI"
                        value="{{ values.bmi }}"
                        required
                    >

                </div>


                <!-- CHILDREN -->

                <div class="input-box">

                    <label>
                        Number of Children
                    </label>

                    <input
                        type="number"
                        name="children"
                        min="0"
                        placeholder="Enter number of children"
                        value="{{ values.children }}"
                        required
                    >

                </div>


                <!-- SMOKER -->

                <div class="input-box">

                    <label>
                        Smoker
                    </label>

                    <select name="smoker" required>

                        <option value="">
                            Select
                        </option>

                        <option
                            value="0"
                            {% if values.smoker == "0" %}
                            selected
                            {% endif %}
                        >
                            No
                        </option>

                        <option
                            value="1"
                            {% if values.smoker == "1" %}
                            selected
                            {% endif %}
                        >
                            Yes
                        </option>

                    </select>

                </div>


                <!-- REGION -->

                <div class="input-box">

                    <label>
                        Region
                    </label>

                    <select name="region" required>

                        <option value="">
                            Select Region
                        </option>

                        <option
                            value="0"
                            {% if values.region == "0" %}
                            selected
                            {% endif %}
                        >
                            Northeast
                        </option>

                        <option
                            value="1"
                            {% if values.region == "1" %}
                            selected
                            {% endif %}
                        >
                            Northwest
                        </option>

                        <option
                            value="2"
                            {% if values.region == "2" %}
                            selected
                            {% endif %}
                        >
                            Southeast
                        </option>

                        <option
                            value="3"
                            {% if values.region == "3" %}
                            selected
                            {% endif %}
                        >
                            Southwest
                        </option>

                    </select>

                </div>


                <!-- BUTTON -->

                <div class="button">

                    <button type="submit">

                        Predict Insurance Cost

                    </button>

                </div>


            </div>

        </form>


        <!-- RESULT -->

        {% if prediction is not none %}

        <div class="result">

            <div class="result-title">
                Estimated Insurance Cost
            </div>

            <div class="result-value">
                ${{ "{:,.2f}".format(prediction) }}
            </div>

        </div>

        {% endif %}


        <!-- ERROR -->

        {% if error %}

        <div class="error">

            {{ error }}

        </div>

        {% endif %}


        <!-- INFORMATION -->

        <div class="info">

            <b>Model Features:</b>

            Age, Sex, BMI, Children, Smoker and Region.

            <br><br>

            The categorical values are converted into
            numerical values before sending them to the
            trained XGBoost model.

        </div>

    </div>


    <footer>

        XGBoost Machine Learning Prediction App

    </footer>


</body>

</html>
"""


# ==============================
# HOME ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    values = {
        "age": "",
        "sex": "",
        "bmi": "",
        "children": "",
        "smoker": "",
        "region": ""
    }

    prediction = None
    error = None


    if request.method == "POST":

        try:

            # Get form values

            values["age"] = request.form.get("age")
            values["sex"] = request.form.get("sex")
            values["bmi"] = request.form.get("bmi")
            values["children"] = request.form.get("children")
            values["smoker"] = request.form.get("smoker")
            values["region"] = request.form.get("region")


            # Convert values

            age = int(values["age"])

            sex = int(values["sex"])

            bmi = float(values["bmi"])

            children = int(values["children"])

            smoker = int(values["smoker"])

            region = int(values["region"])


            # Create DataFrame

            input_data = pd.DataFrame(
                [[
                    age,
                    sex,
                    bmi,
                    children,
                    smoker,
                    region
                ]],
                columns=[
                    "Age",
                    "Sex",
                    "BMI",
                    "Children",
                    "Smoker",
                    "Region"
                ]
            )


            # Prediction

            prediction = model.predict(input_data)[0]

            prediction = float(prediction)


        except Exception as e:

            print("ERROR:", e)

            error = (
                "Prediction failed. "
                "Please check your input values "
                "and model file."
            )


    return render_template_string(
        HTML,
        prediction=prediction,
        error=error,
        values=values
    )


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
