#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:12:41 2022

@author: charlie
"""
import numpy as np
import pandas as pd
import geopandas
import folium
import json
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns; sns.set(style = 'ticks', color_codes=True)




data = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/cleanData/cleanedDataImp.csv")
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

modData = data.copy()
modData['countryMapMatch'] = modData['country']
modData.replace({"Ivory Coast":"Côte d'Ivoire", 
                 "Bosnia and Herzegovina":"Bosnia and Herz.",
                 "Dominican Republic":"Dominican Rep.",
                 "United States":"United States of America"}, inplace = True)
table = world.merge(modData, how="left", left_on=['name'], right_on=['countryMapMatch'], indicator = True)

table2019 = table[table['year'] == 2019]

#%%

import io
from PIL import Image

## Happiness Score mapped

#ts = np.linspace(table2019['happinessScore'].min(), table2019['happinessScore'].max(), 15, dtype = float).tolist()

happMap = folium.Map(location=[0, 0], 
                    zoom_start=1,
                    tiles = None)

folium.Choropleth(geo_data=world,
            data=table2019,
            columns=['countryMapMatch','happinessScore'],
            #bins = ts,
            key_on='feature.properties.name',
            fill_color = 'Blues',
            fill_opacity = 1,
            nan_fill_color='#989B9A',
            line_opacity=0.2,
            line_color = 'White',
            legend_name='Happiness Score'                
).add_to(happMap)

# For saving as png - currently throwing an error with geckodriver
# img_data = happMap._to_png(5)
# img = Image.open(io.BytesIO(img_data))
# img.save("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/results/plots/happinessMap.png")
happMap.save('/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/results/plots/happinessMap.html')

# %%

## Ocean Health Index mapped


ohiMap = folium.Map(location=[0, 0], 
                    zoom_start=1,
                    tiles = None)

folium.Choropleth(geo_data=world,
            data=table2019,
            columns=['countryMapMatch','index'],
            #bins = ts,
            key_on='feature.properties.name',
            fill_color = 'Blues',
            fill_opacity = 1,
            nan_fill_color='#989B9A',
            line_opacity=0.2,
            line_color = 'White',
            legend_name='Ocean Health Index Score'                
).add_to(ohiMap)

ohiMap.save('/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/results/plots/ohiMap.html')
# %%

geo_json_data = json.load(open('/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData/custom.geo.json'))
table2019Dict = table2019.set_index('countryMapMatch')['happinessScore']

# %%

fig, ax = plt.subplots(figsize = (12, 10))

table2019.plot(column = 'happinessScore', cmap = 'Reds', linewidth = 1, ax = ax, edgecolor = '0.3', alpha = 0.8)
table2019['lbl_pts'] = table2019['geometry'].apply(lambda x: x.representative_point().coords[0])

sm = plt.cm.ScalarMappable(cmap = 'Blues', norm = plt.Normalize(vmin = min(table2019['happinessScore']), vmax = max(table2019['happinessScore'])))

ax.axis('off')
cbar = fig.colorbar(sm)

plt.show()

# %%

fig = px.choropleth_mapbox(table,
                           geojson = table,
                           featureidkey = 'properties.name',
                           locations = 'country',
                           color = 'happinessScore',
                           hover_name = 'country',
                           hover_data = ['happinessScore'],
                           color_continuous_scale = 'Blues',
                           animation_frame = 'year',
                           mapbox_style = 'carto-position',
                           )

fig.show()



# %%

import branca

colorscale = branca.colormap.linear.YlGnBu_09.scale(0, 30)

employed_series = df.set_index("countryMapMatch")["Unemployment_rate_2011"]


def style_function(feature):
    employed = employed_series.get(int(feature["id"][-5:]), None)
    return {
        "fillOpacity": 0.5,
        "weight": 0,
        "fillColor": "#black" if employed is None else colorscale(employed),
    }


m = folium.Map(location=[48, -102], tiles="cartodbpositron", zoom_start=3)

folium.TopoJson(
    json.loads(requests.get(county_geo).text),
    "objects.us_counties_20m",
    style_function=style_function,
).add_to(m)


m





# %%


def my_color_function(feature):
    """Maps low values to green and hugh values to red."""
    if table2019Dict[feature['id']] > 6:
        return '#ff0000'
    else:
        return '#008000'
# %%

m = folium.Map([43,-100], tiles='cartodbpositron', zoom_start=4)

folium.GeoJson(
    table2019,
    style_function=lambda feature: {
        'fillColor': my_color_function(feature),
        'color' : 'black',
        'weight' : 2
        }
    ).add_to(m)

m


#%%

bins =  (table2019['score'].quantile((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))).tolist()
m = folium.Map([43,-100], tiles='cartodbpositron', zoom_start=4)
folium.Choropleth(
    geo_data=table2019,
    name='choropleth',
    data=table,
    columns=['countryMapMatch', 'score'],
    key_on='feature.properties.name',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Happiness Score',
    reset = True
).add_to(m)
m.save('happiness.html')


# table3 = pd.DataFrame(table3.drop(columns='geometry'))
# nand3 = table3[table3.isna().any(axis=1)]
# nand3 = nand3[nand3['_merge'] == 'left_only']

# %%
bins =  (table2019['score'].quantile((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))).tolist()
m = folium.Map([43,-100], tiles='cartodbpositron', zoom_start=4)
folium.Choropleth(
    geo_data=table2019,
    name='choropleth',
    data=table,
    columns=['countryMapMatch', 'index'],
    key_on='feature.properties.name',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Ocean Health',
    reset = True
).add_to(m)
m.save('ohi.html')
