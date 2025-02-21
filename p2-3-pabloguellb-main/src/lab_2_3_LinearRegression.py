# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
            """
        X = np.array(X)
        y = np.array(y)

        # Asegurarse de que X sea un vector unidimensional
        if X.ndim > 1:
            X = X.flatten()

        # Calcular las medias de X y y
        x_promedio = np.mean(X)
        y_promedio = np.mean(y)

        # Calcular la covarianza y la varianza
        covarianza = np.sum((X - x_promedio) * (y - y_promedio))
        varianza = np.sum((X - x_promedio) ** 2)

        # Estimar el coeficiente (pendiente) y la intersección (término independiente)
        self.slope = covarianza / varianza
        self.intercept = y_promedio - self.slope * x_promedio

    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        X = np.array(X)
        y = np.array(y)

        # Incorporar una columna de unos para el término de sesgo
        X_aug = np.column_stack((np.ones(X.shape[0]), X))

        # Resolver la ecuación normal para obtener los parámetros del modelo
        params = np.linalg.inv(X_aug.T @ X_aug) @ (X_aug.T @ y)

        self.intercept = params[0]
        self.coefficients = params[1:]

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        X = np.array(X)

        # Verificar si el modelo ya fue entrenado
        if self.coefficients is None or self.intercept is None:
            raise ValueError("El modelo aún no ha sido ajustado")

        # Realizar predicciones según la dimensión de X
        if X.ndim == 1:
            # Caso de una única variable
            resultado = self.intercept + self.coefficients * X
        else:
            # Caso para múltiples variables
            resultado = self.intercept + np.dot(X, self.coefficients)

        return resultado


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # Convertir las listas de valores verdaderos y predichos en arrays de NumPy
    y_actual = np.array(y_true)
    y_estimado = np.array(y_pred)

    # Calcular la variación total de y_actual
    total_var = np.sum((y_actual - np.mean(y_actual)) ** 2)
    # Calcular la suma de errores al cuadrado
    error_cuadrado = np.sum((y_actual - y_estimado) ** 2)
    # Calcular R^2
    r2_score = 1 - (error_cuadrado / total_var)

    # Calcular la raíz del error cuadrático medio (RMSE)
    rmse_value = np.sqrt(np.mean((y_actual - y_estimado) ** 2))

    # Calcular el error absoluto medio (MAE)
    mae_value = np.mean(np.abs(y_actual - y_estimado))

    # Devolver un diccionario con las métricas calculadas
    resultados = {"R2": r2_score, "RMSE": rmse_value, "MAE": mae_value}
    return resultados

# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    """Compares a custom linear regression model with scikit-learn's LinearRegression.

    Args:
        x (numpy.ndarray): The input feature data (1D array).
        y (numpy.ndarray): The target values (1D array).
        linreg (object): An instance of a custom linear regression model. Must have
            attributes `coefficients` and `intercept`.

    Returns:
        dict: A dictionary containing the coefficients and intercepts of both the
            custom model and the scikit-learn model. Keys are:
            - "custom_coefficient": Coefficient of the custom model.
            - "custom_intercept": Intercept of the custom model.
            - "sklearn_coefficient": Coefficient of the scikit-learn model.
            - "sklearn_intercept": Intercept of the scikit-learn model.
    """
    ### Compare your model with sklearn linear regression model
    # TODO : Import Linear regression from sklearn
    from sklearn.linear_model import LinearRegression  

    x_reshaped = np.array(x).reshape(-1, 1)

    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)


    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    """Loads Anscombe's quartet, fits custom linear regression models, and evaluates performance.

    Returns:
        tuple: A tuple containing:
            - anscombe (pandas.DataFrame): The Anscombe's quartet dataset.
            - datasets (list): A list of unique dataset identifiers in Anscombe's quartet.
            - models (dict): A dictionary where keys are dataset identifiers and values
              are the fitted custom linear regression models.
            - results (dict): A dictionary containing evaluation metrics (R2, RMSE, MAE)
              for each dataset.
    """
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    # TODO: Construct an array that contains, for each entry, the identifier of each dataset
    datasets = anscombe["dataset"].unique()

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}

    for dataset in datasets:
        # Filter the data for the current dataset
        data = anscombe.loc[anscombe["dataset"] == dataset]

        # Create a linear regression model
        model = LinearRegressor()

        # Fit the model using the predictor and response from the current dataset
        X = data["x"].values  # Predictor, make it 1D for your custom model
        y = data["y"].values  # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        y_pred = model.predict(X)

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}")

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}")
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])

    return anscombe, datasets, models, results

# Go to the notebook to visualize the results
