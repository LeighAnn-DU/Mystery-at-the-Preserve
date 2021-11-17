#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 08:09:51 2021

@author: leighannkudloff
"""

import streamlit            as st
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image 
import re

st.sidebar.subheader('Table of Contents')
st.sidebar.write('1. ','<a href=#introduction>Introduction</a>', unsafe_allow_html=True)
#st.sidebar.write('2. ','<a href=#data-description>Data Description</a>', unsafe_allow_html=True)
st.sidebar.write('2. ','<a href=#data-description>Data Description</a>', unsafe_allow_html=True)
st.sidebar.write('3. ','<a href=#data-processing>Data Processing</a>', unsafe_allow_html=True)
st.sidebar.write('4. ','<a href=#data-visualization>Data Visualization</a>', unsafe_allow_html=True)
st.sidebar.write('5. ','<a href=#goal-1-results>Goal 1 Results</a>', unsafe_allow_html=True)
st.sidebar.write('6. ','<a href=#machine-learning-models>Machine Learning Models</a>', unsafe_allow_html=True)
st.sidebar.write('7. ','<a href=#goal-2-results>Goal 2 Results</a>', unsafe_allow_html=True)
st.sidebar.write('8. ','<a href=#goal-3-results>Goal 3 Results</a>', unsafe_allow_html=True)
st.sidebar.write('9. ','<a href=#key-insights-and-conclusion>Key Insights and Conclusion</a>', unsafe_allow_html=True)
st.sidebar.write('10. ','<a href=#questions>Questions</a>', unsafe_allow_html=True)

st.header("*Introduction*")
st.markdown('# "Cheep" Shots:   Suspense at the Wildlife Preserve')
st.image("RCBPpic.png")

audio_file = open("Rose-Crested-Blue-Pipit-162563.mp3", "rb")
st.audio(audio_file.read())

st.markdown("#### Paul O'Leary")
st.markdown("#### Leigh Ann Kudloff")
st.markdown("#### November 2021")
st.markdown("#")

st.header("**Scenario**")

st.markdown("### -Possible endangerment of the Rose-Crested Blue Pipit near Mistford and the Boonsong Lekagul Wildlife Preserve")
st.markdown("### -Investigation implicating Kasios Office Furniture, a manufacturing firm")
st.markdown("* Kasios presents itself as extremely eco-friendly")
st.markdown("* Kasios supposedly used banned substance, Methylosmolene")
st.markdown("* Kasios dumped waste in NW region of preserve")
st.markdown("* Methylosmolene detected in smokestack emissions")
st.markdown("### -Kasios claims the investigation analysis is flawed and biased.")
st.markdown("* Kasios reports plenty of Rose-Crested Blue Pipits")
st.markdown("* Kasios provided set of recent recorded bird calls with locations")
st.markdown("### -Pangera Ornithology Conservation Society concerned and suspicious")
st.markdown("### -Town of Mistford and preserve rangers satisfied recordings back Kasios’ claims")
st.markdown("### -Mistford College does not have a Pipit expert and provides collection of bird calls vetted by ornithology groups")
st.markdown("#")

st.header("**Scenario Goals**")
st.markdown("*  Using the bird call collection and the included map of the Wildlife Preserve, characterize the patterns of all of the bird species in the Preserve over the time of the collection. Please assume we have a reasonable distribution of sensors and human collectors providing the recordings, so that the patterns are reasonably representative of the bird locations across the area. Do you detect any trends or anomalies in the patterns?")
st.markdown("*  Turn your attention to the set of bird calls supplied by Kasios. Does this set support the claim of Pipits being found across the Preserve? A machine learning approach using the bird call library may help your investigation. What is the role of visualization in your analysis of the Kasios bird calls?")
st.markdown("*  Formulate a hypotheses concerning the state of the Rose Crested Blue Pipit. What are your primary pieces of evidence to support your assertion? What next steps should be taken in the investigation to either support or refute the Kasios claim that the Pipits are actually thriving across the Boonsong Lekagul Wildlife Preserve?")

#st.header("*Data Description*")
st.header("*Data Description*")
st.header("**Bird Call Collection**")

st.markdown("### -Mistford College dataset has 2081 rows with these features:")
st.markdown("* ID")
st.markdown("* Name of Bird—19 different bird varieties")
st.markdown("* Vocalization Type (Song or Call)")
st.markdown("* Quality of Recording (A, B, C, D, E, none)")
st.markdown("* Time of Day")
st.markdown("* Date—1983 to 2018")
st.markdown("* Location (X and Y)—on a 200 by 200 grid")
st.markdown("### -Kasios dataset has 15 rows with these features:")
st.markdown("* ID")
st.markdown("* Location (X and Y)")
st.markdown("### -Files (mp3) of bird recordings matched to ID for both Mistford College dataset and the Kasios dataset")

st.header("**Examples, Visual Representations**")

st.markdown("### Rose-Crested Blue Pipit Song--Wave Plot is the sound amplitude (loudness) over time in dB")
st.image("BPSong.png")
st.markdown("### Rose-Crested Blue Pipit Song--Mel Spectrogram visualizes the various sound frequencies and loudness in dB that make up what we hear.")
st.image("BPSpectrogram.png")

st.header("**Visual Comparison, Samples, Kasios files to Known RCBP Call and Song**")
st.image("KasiosRec.png")

st.header("*Data Processing*")

st.header("**Data Processing of the Recordings**")
st.markdown("### -Librosa used for audio analysis")
st.markdown("### -Convert MP3 files to WAV files")
st.markdown("### -Verified bird calls and Kasios provided test recordings – All files were:")
st.markdown("* Normalized over time")
st.markdown("* Stripped of intervals of silence")
st.markdown("* Split into 2 second time chunks")
st.markdown("### -Primary Features Explored")
st.markdown("* Mel-frequency cepstral coefficients (MFCC)-- a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.")
st.markdown("* Chroma Short Time Fourier Transform (STFT)--Transform to determine the sinusoidal frequency and phase content of a short segment of sound with a pitch profile.")
st.markdown("### -Vectors created by above processes are summarized by mean and standard deviation and put into a two-dimensional array (2 x 50), 56,272 records in the verified samples and 476 in the Kasios recordings.")
st.header("#")
st.header("**Data Processing of Bird Data**")
st.markdown("* Ensured consistency (capitalizations, etc.) in all columns")
st.markdown("* Changed date to datetime object")
st.markdown("* Grouped times to morning, afternoon, evening, night")
st.markdown("* Created Season and Year and then Season/Year columns")
st.markdown("* Added colors for each of 19 bird types")
st.header("#")
st.header("#")

st.header("*Data Visualization*")
st.header("**All Birds in Mistford College Dataset**")

# Read in the data
birds = pd.read_csv("AllBirdsv4.csv")

# changing capitalization of desired bird
birds.replace({"Rose-crested Blue Pipit": "Rose-Crested Blue Pipit"}, inplace=True)
birdcolors = {"Orange Pine Plover": "orange", 
              "Green-tipped Scarlet Pipit": "green", 
              "Blue-collared Zipper": "deepskyblue",
             "Vermillion Trillian": "crimson",
             "Purple Tooting Tout": "purple",
             "Rose-Crested Blue Pipit": "blue",
             "Carries Champagne Pipit": "tan",
             "Eastern Corn Skeet": "chocolate",
             "Pinkfinch": "deeppink",
             "Queenscoat": "darkviolet",
             "Lesser Birchbeere": "chartreuse",
             "Bombadil": "sandybrown",
             "Broad-winged Jojo": "turquoise",
             "Ordinary Snape": "slategray",
             "Scrawny Jay": "olivedrab",
             "Darkwing Sparrow": "darkseagreen",
             "Canadian Cootamum": "tomato",
             "Bent-beak Riffraff": "burlywood",
             "Qax": "coral"}
# function to deal with time
def floatconvert(x):
    try:
        x=float(x)
        if x>24:
            x=x/100
        return x
    except ValueError:
        return np.nan

# function to deal with date
def datestring(x):
    if x == "0000-00-00":
        return np.nan
    elif re.search("-", x) is not None:
        templist = x.split("-")
        if templist[-2]=="00":
            return np.nan
        templist[-1]="01"
        return "-".join(templist)
    else:
        return x
birds["pm"]=np.where(birds.Time.str.contains("pm"), 1, 0)        
birds["Time"]=birds.Time.str.replace("AM", "").str.replace(":", ".").str.replace(";",".").str.replace("am", "").str.replace("pm", "").str.strip().apply(floatconvert)
birds["Time"]=birds.apply(lambda x: x["Time"]+12 if x["pm"]==1 and x["Time"]<12 else x["Time"], axis=1)
birds["Time"]=pd.cut(birds.Time, bins=[0,6,12,18,24], labels=["Night", "Morning", "Afternoon", "Evening"], right=False)
birds.loc[birds["File ID"]==98668, "Time"]="Afternoon"
birds.loc[birds["File ID"]==98204, "Time"]="Evening"
birds.drop("pm", axis=1, inplace = True)

# cleaning the Date column
birds["Date"]= birds.Date.str.strip().apply(datestring)
birds["Date"]=pd.to_datetime(birds["Date"])

# creating a season column
birds["Season"]=birds.Date.dt.month%12//3+1
birds["Season"]=birds["Season"].replace({1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"})

# creating a year column
birds["Year"]=birds.Date.dt.year
birds["5_Year"]=pd.cut(birds.Year, bins=np.arange(1980,2021,5), 
       labels=["1981-1985", "1986-1990", "1991-1995", "1996-2000", 
               "2001-2005", "2006-2010", "2011-2015", "2016-2020"])

# create a column for season and year
birds["Season_Year"]=np.where(birds["Date"].isnull(), np.nan, 
                              birds["Season"]+ " " + birds["Year"].astype(str))
birds["Season_Year"]=birds["Season_Year"].str.split(".", expand=True).iloc[:,0]
# category order for Season_Year
rawSYlabels=[s + " " + y for y in sorted(birds["Year"].dropna().unique().astype(int).astype(str)) for s in ["Winter", "Spring", "Summer", 
                              "Fall"]]

SYlabels=[i for i in rawSYlabels if i in birds["Season_Year"].unique()]
birds["Season_Year"] = pd.Categorical(birds["Season_Year"], categories = SYlabels, ordered=True)

# cleaning the Vocalization column
birds["Vocalization_type"]=birds.Vocalization_type.str.lower().str.strip().str.replace("?", "unknown", regex=True)

# adjusting the y variable for location
birds["Y"]=birds.Y.str.replace("?", "", regex=True).astype(int)
birds["Yadj"]=200-birds["Y"]

# adding colors for birds
birds["Bird Color"]=birds["English_name"].replace(birdcolors)

# The test set
testbirds = pd.read_csv("Test Birds Location.csv")
testbirds["yadj"]= 200-testbirds[" Y"]

# Overall Bird Map of all Years Combined
birdmap = "Lekagul Roadways 2018.bmp"

img = Image.open(birdmap)

fig = px.imshow(img, binary_format="jpeg", binary_compression_level=0, binary_string=True, width=800, height=800)

for birdname, birdcolor in birdcolors.items():
    df=birds.loc[birds["English_name"]==birdname].copy()
    fig.add_trace(go.Scatter(x=df["X"], y=df["Yadj"], 
                         mode="markers", marker=dict(size=8, color=df["Bird Color"]), name = birdname))
    
fig.add_trace(go.Scatter(x=testbirds[" X"], y=testbirds["yadj"], mode="markers+text", 
                         text=testbirds["ID"], marker=dict(size=15, color="yellow"), 
                         textfont=dict(size=15, color="black"), showlegend=False))

fig.add_trace(go.Scatter(x=[148], y=[200-159], mode="markers", name = "Dumping Site", 
                          marker = dict(size=20, color="red", symbol="x")))

fig.update_yaxes(ticktext=list(range(200, 0, -50)), tickvals=list(range(0, 200, 50)))
fig.update_xaxes(ticktext=list(range(0, 200, 50)), tickvals=list(range(0, 200, 50)))
st.plotly_chart(fig)

#st.header("**All Birds Per Season Per Year**")
# Looking at All Birds each Season (of each Year)

dummy_data_list = []
for bird in birds['English_name'].unique():
    for year in birds.loc[~birds['Season_Year'].isnull()].Season_Year.unique():
        dummy_data_list.append([-1,-1,bird,year])

dummy_data_sy = pd.concat([pd.DataFrame(dummy_data_list,columns = ["X","Y","English_name","Season_Year"]),
                                    birds.loc[~birds["Season_Year"].isnull()]]).reset_index(drop=True)

fig2 = px.scatter(dummy_data_sy, x = "X",y = "Y",color = "English_name",color_discrete_map = birdcolors,
           animation_frame = "Season_Year", category_orders={'Season_Year': birds.Season_Year.cat.categories}, 
                 range_x = [0,200], range_y = [0,200],
           labels = {"X":"","Y":"", "English_name":"Bird Type"}, title="All Birds per Season per Year")
fig2.add_layout_image(
        dict(source=img,
            x=0,
            sizex=200,
            y=200,
            sizey=200,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="contain"
        ))
fig2.add_trace(go.Scatter(x=testbirds[" X"], y=testbirds[" Y"], mode="markers+text", 
                         text=testbirds["ID"], marker=dict(size=15, color="yellow"), 
                         textfont=dict(size=15, color="black"), showlegend=False))

fig2.add_trace(go.Scatter(x=[148], y=[159], mode="markers", name = "Dumping Site", 
                          marker = dict(size=20, color="red", symbol="x")))

fig2.update_yaxes(range=[0, 200], scaleanchor="x", scaleratio=1, showticklabels=False,
                showgrid=False, showline=False)
fig2.update_xaxes(range=[0, 200], showgrid=False, showline=False, showticklabels=False)

fig2.update_layout(height=800, width=800, title_font_size=24, title_text='<b> All Birds per Season per Year</b>')

fig2.layout.plot_bgcolor = "blue"
st.plotly_chart(fig2)

#st.header("**All Rose-Crested Blue Pipits Per Season Per Year**")
# Looking at Rose-Crested Blue Pipits each Season (of each Year)

dummy_data_list = []
for bird in birds['English_name'].unique():
    for year in birds.loc[~birds['Season_Year'].isnull()].Season_Year.unique():
        dummy_data_list.append([-200,-200,bird,year])

dummy_data_sy = pd.concat([pd.DataFrame(dummy_data_list,columns = ["X","Y","English_name","Season_Year"]),
                                    birds.loc[~birds["Season_Year"].isnull()]]).reset_index(drop=True)

fig3 = px.scatter(dummy_data_sy.loc[dummy_data_sy.English_name=="Rose-Crested Blue Pipit"], x = "X",y = "Y",color = "English_name",color_discrete_map = birdcolors,
           animation_frame = "Season_Year", category_orders={'Season_Year': birds.Season_Year.cat.categories}, 
                 range_x = [0,200], range_y = [0,200],
           labels = {"X":"","Y":"", "English_name":"Bird Type"}, title="All Rose-Crested Blue Pipits per Season per Year")
fig3.add_layout_image(
        dict(source=img,
            x=0,
            sizex=200,
            y=200,
            sizey=200,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="contain"
        ))
fig3.add_trace(go.Scatter(x=testbirds[" X"], y=testbirds[" Y"], mode="markers+text", 
                         text=testbirds["ID"], marker=dict(size=15, color="yellow"), 
                         textfont=dict(size=15, color="black"), showlegend=False))

fig3.add_trace(go.Scatter(x=[148], y=[159], mode="markers", name = "Dumping Site", 
                          marker = dict(size=20, color="red", symbol="x")))

fig3.update_yaxes(range=[0, 200], scaleanchor="x", scaleratio=1, showticklabels=False,
                showgrid=False, showline=False)
fig3.update_xaxes(range=[0, 200], showgrid=False, showline=False, showticklabels=False)

fig3.update_layout(height=800, width=800, title_font_size=24, title_text='<b>All Rose-Crested Blue Pipits per Season per Year</b>')

fig3.layout.plot_bgcolor = "blue"
st.plotly_chart(fig3)

st.header("*Goal 1 Results*")
st.header("**Goal 1:  Do you detect any trends or anomalies in the patterns?**")
st.markdown("### -Rose-Crested Blue Pipits used to inhabit the area around the dumping site but have changed location.")
st.markdown("* Nearest last sighting of Rose-Crested Blue Pipit in Spring 2017")
st.markdown("### -As of 2016, most birds have left the area around the dumping ground.")
st.markdown("* Darkwing Sparrow—Spring 2018")
st.markdown("* Broad-winged Jojo—Spring 2016 and Summer 2017")
st.markdown("* Eastern Corn Skeet—Winter 2017")
st.markdown("* Carries Champagne Pipit—Fall 2017")
st.markdown("### -None of the Kasios test bird recordings are near dumping site.")

st.header("*Machine Learning Models*")
st.header("**Models Used to Analyze Kasios’ Test Data**")
st.markdown("### -Binary Classification Models—full clips")
st.markdown("* One Vs. Rest Binary Classifier")
st.markdown("* MLP Binary Classifier with SMOTE for Imbalanced Data")
st.markdown("* Decision Tree")
st.markdown("* Random Forest")
st.markdown("### -Multi-Classification with 2 second chunks (19 bird classes)")
st.markdown("* MLP Classifier")
st.markdown("* Random Forest")
st.markdown("* Long Short-Term Memory Recurrent Neural Network LSTM-RNN")
st.markdown("#")

st.header("**Binary Results:  MFCC with OneVsRest and MLP**")
st.markdown("* Imbalanced Data—used SMOTE and adjusted both over sampling the RCBP and under sampling all other birds")
st.markdown("* MFCC dataset with means and medians on entire clips")
st.markdown("* Dataset with 40 versus 100 columns/features")
st.markdown("* Limited dataset to Quality levels of A, B, and C")
st.markdown("* StandardScaler better than MinMax Scaler")
st.markdown("* Best model included two layers [1024, 16] & sgd as solver")
st.markdown("* Every model used on test data to show none of the birds in the test data was a Rose-Crested Blue Pipit")
st.markdown("#")

st.header("**Multiclass Results  -  Random Forest**")
st.markdown("* A, B, C, D quality recordings, 1000 estimator trees, Gini, on 20% Validation Set")
st.image("RFClassReport.png")
st.image("RFConfMatrix.png")
st.markdown("* F1 score of 0.91 to predict the Rose-crested Blue Pipit")
st.markdown("* Predicted that 4 of the 15 Kasios files are Blue Pipits")
st.markdown("#")

st.header("**Final Model Results  -  LSTM RNN**")
st.markdown("* optimizer='adam',loss='SparseCategoricalCrossentropy', metrics=['acc’], 25 epochs, 20% validation set")
st.image("RNNClassReport.png")
st.image("RNNTrainVal.png")
st.markdown("* F1 score of 0.94 on to predict the Rose-crested Blue Pipit")
st.markdown("* Predicted that 3 of the 15 Kasios files are Blue Pipits")
st.markdown("#")

st.header("*Goal 2 Results*")
st.header("**Goal 2:  What is the role of visualization in your analysis of the Kasios bird calls?**")
st.markdown("* Seeing gaps on the visualizations led to the elimination of silence.")
st.markdown("* The maps showed where different types of birds congregate and illustrated the movement of those centers after dumping began.")
st.markdown("* The test birds are no where near the dumping site.")
st.markdown("* The results from the classification indicate that sites 2 and 9 (and maybe 13 and 14) are most likely Rose-Crested Blue Pipits, which show even further migration from the dumping site.")
st.markdown("#")

st.header("**Goal 2: Does this set support the claim of Pipits being found across the Preserve?**")
st.markdown("* The binary classification MLP showed 62% of the predicted Rose-Crested Blue Pipits in the Mistford College data are actually Rose-Crested Blue Pipits, while 97% of the other birds are predicted correctly, but predicted none of the Kasios birds were Rose-Crested Blue Pipits.")
st.markdown("* The Random Forest model predicted the Rose-Crested Blue Pipit 91% of the time, and 4 of the Kasios’ birds were RCBP.")
st.markdown("* The best model (LSTM-RNN) accurately predicted Rose-Crested Blue Pipits 94% of the time and correctly identifying other bird species better than 88% of the time.")
st.markdown("* In summary, at most 3-4 of the bird recordings from the Kasios test data are Rose-Crested Blue Pipits based on the analysis.")
st.markdown("#")

st.header("*Goal 3 Results*")
st.header("**Goal 3: Formulate a hypotheses concerning the state of the Rose Crested Blue Pipit. What are your primary pieces of evidence to support your assertion?**")
st.markdown("*Hypothesis:*  The Rose-Crested Blue Pipit population is decreasing as a result of migration away from a dumping site and possibly polluted air.")
st.markdown("* The map shows Rose-Crested Blue Pipits and other birds have moved away from the dumping site and inhabit other areas of the Preserve.")
st.markdown("* Data analysis of sound recordings predicts less than 5 of the Kasios provided recordings are of the Blue Pipit.  Kasios is either unaware or trying to deceive with the test data of 15 bird recordings.")
st.markdown("* None of the Kasios provided recordings were made near the original dumping site.")

st.header("**Goal 3 (Actionable Insights):  What next steps should be taken in the investigation to either support or refute the Kasios claim that the Pipits are actually thriving across the Boonsong Lekagul Wildlife Preserve?**")
st.markdown("* Continue to record all bird locations in the preserve, map future changes in the populations and locations, and investigate if more contamination has occurred in other areas of the preserve.")
st.markdown("* Analyze pollutants in the air and in the water in representative locations in the preserve.  Determine if pollutants still exist.")
st.markdown("* Pursue methods to clean the preserve, if pollutants do exist.")
st.markdown("* Determine if satellite imagery of the preserve over time exists.  Changes in air quality and ground vegetation may also be affecting the wildlife.")
st.markdown("* Explore the disruption to other birds species (competition for territory) caused by the migration of the Rose-Crested Blue Pipit.")
#st.markdown("#")

st.header("*Key Insights and Conclusion*")
st.header("**Key Insights and Data Science Conclusions**")
st.markdown("* Librosa is an extensive package of sound analysis tools, that processes data conducive to time series analysis, and takes a great deal of time to master with a steep learning curve.")
st.markdown("* The machine learning and model tuning techniques explored required extreme hardware resources.  Kernels blew up regularly.")  
st.markdown("* Neural networks proved to be effective in sound analysis.")

st.header("*Questions*")
st.markdown("What questions can we answer?")


 
