#%%
def coletar_dados():
    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Input data files are available in the read-only "../input/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    import os

    import kagglehub
    DATASET_PATH = os.environ.get("DATASET_PATH") or kagglehub.dataset_download(
        'hkayan/industrial-robotic-arm-imu-data-casper-1-and-2') + '/'

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

    import re
    normalPattern = "IMU_(.*)Hz.csv"

    normal = ("IMU_10Hz.csv", ("label", 0))
    faulty =("IMU_hitting_platform.csv", ("label", 1))

    df = pd.read_csv(DATASET_PATH + normal[0])
    df['label'] = 0

    faultydf = pd.read_csv(DATASET_PATH+faulty[0])
    faultydf['label'] = 1

    return df, faultydf

# %%
