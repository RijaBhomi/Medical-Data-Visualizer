import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv("medical_examination.csv")

# 2. Add 'overweight' column
# BMI = weight / (height in meters)^2
# If BMI > 25 → overweight = 1, else 0
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalize cholesterol and gluc
# - If value == 1 → good → set to 0
# - If value > 1 (2 or 3) → bad → set to 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Categorical Plot function
def draw_cat_plot():
    # 5. Melt the DataFrame into long format
    # 'cardio' stays as ID variable
    # selected lifestyle/health columns become 'variable'
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # 6. Group and count
    # For each combination of cardio (0/1), variable, and value (0/1),
    # count how many rows exist.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Draw the seaborn catplot
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    ).fig

    # 8. Save figure
    fig.savefig('catplot.png')
    return fig


# 10. Heat Map function
def draw_heat_map():
    # 11. Clean the data
    # Remove incorrect or extreme values:
    # - diastolic pressure (ap_lo) must be ≤ systolic (ap_hi)
    # - height must be within 2.5th and 97.5th percentiles
    # - weight must be within 2.5th and 97.5th percentiles
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle (to avoid duplicate values)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15. Draw the heatmap
    # - annot=True → show numbers inside cells
    # - fmt=".1f" → show numbers with 1 decimal place
    # - center=0 → center colors around 0
    # - square=True → square cells
    # - cbar_kws={'shrink': .5} → shrink color bar for readability
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        cbar_kws={'shrink': .5}
    )

    # 16. Save figure
    fig.savefig('heatmap.png')
    return fig
