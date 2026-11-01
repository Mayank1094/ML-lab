import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

data = pd.read_csv("heart.csv")
data = data[["age", "gender", "cp", "target"]]

model = DiscreteBayesianNetwork([
    ("age", "target"),
    ("gender", "target"),
    ("cp", "target")
])

model.fit(data)

infer = VariableElimination(model)
result = infer.query(["target"], evidence={"age": 50, "gender": 1, "cp": 2})

print("Heart Disease Diagnosis:")
print(result)
