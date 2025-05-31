import pandas as pd

def double_iris():
    df = pd.read_csv('data/iris.csv')
    df_doubled = pd.concat([df, df], ignore_index=True)
    df_doubled.to_csv('data/iris.csv', index=False)

if __name__ == "__main__":
    double_iris()
