import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import json

    C = "/tmp/tahoe_cache_combined/tahoe/5_holdout"
    Y = np.load(f"{C}/Y_train.npy", mmap_mode="r")
    m = json.load(open(f"{C}/meta.json"))
    return Y, m


@app.cell
def _(m):
    m
    return


@app.cell
def _(Y):
    Y
    return


@app.cell
def _(Y):
    Y.shape
    return


if __name__ == "__main__":
    app.run()
