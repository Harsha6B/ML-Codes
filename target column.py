##if last column is the target
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column = output

##if not the last column is the target
y = df["result"]
X = df.drop("result", axis=1)
