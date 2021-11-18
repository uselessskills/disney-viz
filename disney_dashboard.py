import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import hiplot as hip
from scipy import stats
import statsmodels.api as sm

sns.set_theme()

st.title('Disney dataset')

logo = Image.open('data/mosaic.jpeg')
st.image(logo, caption='Disney mosaic from posters, https://github.com/codebox/mosaic')

data = pd.read_csv("data/202111 - DisneyDataset.csv", index_col=0)
data.index.name = "id"
data = data.reset_index()

def percentile(s, margin=0.01):
    s = s.dropna()
    min_, max_ = np.percentile(s, [100*margin, 100 * (1-margin)])
    return ((s < min_) | (s > max_)), min_, max_

# releases over time
with st.container():
    df = data.copy()
    df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
    n_years = len(df["Release date (datetime)"].apply(lambda x: x.year).unique())

    f, ax = plt.subplots(figsize=(10, 3))
    sns.histplot(df["Release date (datetime)"], kde=False, bins=n_years, ax=ax)
    q = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)[["id"]].count()
    q = q.dropna().rolling(5).mean()
    sns.lineplot(data=q.iloc[:,0].rename('# releases (5 year moving avg)'), ax=ax)
    ax.set_ylabel('Number of movies released')
    ax.set_xlabel('')
    f.legend(bbox_to_anchor=(0.61, 0.87), facecolor='white', fontsize=9,
        labels=["# releases (5 year mov. avg.)"]
    )
    
    st.subheader("Number of movies has grown over time")
    st.pyplot(f)

# quality 
with st.container():
    col1, col2= st.columns(2)
    with col1: 
        df = data.copy()
        df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
        df = df[df["Release date (datetime)"] < "2021"] 

        f, ax = plt.subplots(figsize=(6, 3))
        q = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)[["imdb"]].mean()
        q = q.dropna().rolling(5).mean().dropna()
        sns.lineplot(data=q["imdb"].rename('IMDB rating (5 year mov.avg.)'), ax=ax, linewidth=1)
        ax.set_ylabel('IMDB rating (5 year mov.avg.)')
        ax.set_xlabel('')
        st.subheader("Quality is volatile")
        st.pyplot(f)
    
    with col2:
        df = data.copy()
        df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
        df = df[df["Release date (datetime)"] < "2021"]

        f, ax = plt.subplots(figsize=(6, 3))
        q = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)["imdb"].mean()
        q = q.dropna().rolling(1).mean().dropna()
        sm.graphics.tsa.plot_acf(q.values, lags=30, ax=ax)
        ax.set_title("Autocorrelation of IMDB avg. annual rating", fontsize=12)
        ax.set_xlabel('Year')

        st.subheader("... and it is hard to preserve it over many years")
        st.pyplot(f)

# correlation between releases and rating
with st.container():
    df = data.copy()
    df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
    df = df[df["Release date (datetime)"] < "2021"] 

    n_years = len(df["Release date (datetime)"].apply(lambda x: x.year).unique())

    f, ax = plt.subplots(figsize=(10, 3))
    sns.histplot(df["Release date (datetime)"], kde=False, bins=n_years, ax=ax)
    ax.set_ylabel('Number of movies released')
    ax.set_xlabel('')

    # correlation
    df = data.copy()
    df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
    df = df[df["Release date (datetime)"] < "2021"] 

    q = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)["id"].count()
    q = q.dropna().rolling(1).mean()
    q = q.diff().dropna()
    q = q.rename("releases")

    q2 = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)["imdb"].mean()
    q2 = q2.dropna().rolling(1).mean()
    q2 = q2.diff().dropna()
    q2 = q2.rename("rating")

    qq = q.to_frame().join(q2.to_frame()).dropna()
    corr = qq["releases"].rolling(10).corr(qq["rating"])

    ax2 = plt.twinx()
    sns.lineplot(data=corr, ax=ax2, color="#CE056A", linewidth=2)
    ax2.set_ylabel('Rolling 10-year correlation\n of # releases and IMDB rating', color="#CE056A")

    st.subheader("Over time growth of releases attributes with decline in avg. quality")
    st.markdown('Pearson corr. is calculated between 1st order differences of #releases and imdb avg. rating')
    st.pyplot(f)


# runtime
with st.container():
    col1, col2= st.columns(2)
    with col1: 
        df = data.copy()

        df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
        q = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)[["Running time (int)"]].mean()
        q = q.dropna().rolling(5).mean()

        f, ax = plt.subplots(figsize=(5, 3))
        sns.lineplot(data=q["Running time (int)"], ax=ax)

        st.subheader("Movies become longer")
        st.pyplot(f)

    with col2:
        df = data.copy()
        df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
        df = df[df["Release date (datetime)"] < "2021"] 
        q = df[["Running time (int)", "imdb"]].dropna()

        g = sns.jointplot(
            x="Running time (int)", y="imdb", data=q, kind="reg", truncate=False,
            height=6
        )
        st.subheader("... but it has no relation with quality")
        slope, intercept, r_value, p_value, std_err = stats.linregress(q["Running time (int)"], q["imdb"])
        st.text(f"r2: {r_value**2:.2f}, p-value: {p_value:.4f}")
        st.pyplot(g)
        
# Box office vs Budget
with st.container():
    df = data.copy()
    df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
    q = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)["Budget (float)", "Box office (float)"].mean()

    f, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=q["Budget (float)"], ax=ax)

    sns.lineplot(data=q["Box office (float)"], ax=ax, color="#008C95")
    ax.set(yscale='log')

    q = q.diff().dropna()
    corr = q["Budget (float)"].rolling(10).corr(q["Box office (float)"])

    ax2 = plt.twinx()
    sns.lineplot(data=corr, ax=ax2, linewidth=1, color="#CE056A")
    ax2.lines[0].set_linestyle("--")

    ax.set_ylabel('$')
    ax2.set_ylabel('Correlation', color="#CE056A")
    ax.set_xlabel('')

    f.legend(bbox_to_anchor=(0.4, 0.87), facecolor='white', fontsize=9,
            labels=["Budget", "Box office", "10-year rolling correlation between\n 1st order diffs of Budget and Box office"]
    )

    st.subheader("Profitability mechanics improved significantly over time")
    st.pyplot(f)

# ROI
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        df = data.copy()
        df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
        df['roi'] = df["Box office (float)"]/df["Budget (float)"]-1

        q = df['roi'].dropna()
        q = q.clip(*percentile(q, 0.05)[1:])

        f, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(data=q, kde=True, ax=ax)
        text = "Outliers (two-tailed 5% percentile) are \ngrouped at boundaries"
        ax.text(0.3, 0.87, text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        st.subheader("Movie making returns are right-skewed with avg. return of 166%")
        st.text(f"median: {q.median():.2f}, mean: {q.mean():.2f}")
        st.pyplot(f)

    with col2:
        df = data.copy()
        df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
        df['roi'] = df["Box office (float)"]/df["Budget (float)"]-1
        q = df.groupby(pd.Grouper(key='Release date (datetime)', freq='Y'), as_index=True)[["roi"]].mean()

        q = q.iloc[40:]
        c = "roi"
        q[c] = q[c].clip(*percentile(q[c], 0.025)[1:])

        f, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(data=q["roi"], ax=ax)
        ax.set_xlabel('')

        st.subheader("Traditional movie economics is robust, but has a drop in 2020")
        st.pyplot(f)

# roi vs quality
with st.container():
    df = data.copy()
    df['roi'] = df["Box office (float)"]/df["Budget (float)"]-1
    q = df[["roi", "imdb"]].dropna()
    # q = q[q["roi"] < q["roi"].mean() + 1*q["roi"].std()]
    q = q[q["roi"] < 101]

    fig = px.density_contour(
        q, x="roi", y="imdb", marginal_x="histogram", marginal_y="histogram",
        nbinsx=50
    )

    st.subheader("Profitability has no correlation with IMDB rating")
    slope, intercept, r_value, p_value, std_err = stats.linregress(q["roi"], q["imdb"])
    st.text(f"r2: {r_value**2:.2f}, p-value: {p_value:.4f}")
    st.plotly_chart(fig, use_container_width=True)

# parallel coordinates
with st.container():
    df = data.copy()
    df["Release date (datetime)"] = pd.to_datetime(df["Release date (datetime)"])
    df["Release year"] = df["Release date (datetime)"].apply(lambda x: x.year)
    df['roi'] = round(df["Box office (float)"]/df["Budget (float)"]-1, 1)
    df["title length"] = df["title"].apply(lambda x: len(x))

    q = df.copy()
    q = q[q["roi"] < 500]
    
    cols = [
        "title", "Release year",
        "roi", "title length", "imdb", "Running time (int)", "Box office (float)", "Budget (float)",
    ]
    q = q[cols].dropna()

    xp = hip.Experiment.from_dataframe(q)
    xp.display_data(hip.Displays.PARALLEL_PLOT).update({
        'hide': ['id', 'title'],
        'order': ['imdb', "Release year", "roi", "Budget (float)", "Box office (float)", "title length"],
    })
    xp.display_data(hip.Displays.TABLE).update({
        'hide': ['uid', 'from_uid', "id"],
        'order_by': [['imdb']],
        'order': ['title'],
    })

    st.subheader("One chart to bring them all")
    xp.to_streamlit(key="id").display()
