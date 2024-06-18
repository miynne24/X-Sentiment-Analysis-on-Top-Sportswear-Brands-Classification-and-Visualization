<<<<<<< HEAD
import requests
from flask import Flask, render_template, jsonify
from flask import request
import pandas as pd
import json
from textblob import TextBlob
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from langdetect import detect, LangDetectException
import re
from nltk.corpus import stopwords

main = Flask(__name__)


@main.route("/")
def home():
    return render_template("homepage.html")


@main.route("/dashboard")
def dashboard():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')

    # Convert 'created_at' to datetime with specified format and group by month
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y').dt.to_period('M')

    # Prepare sentiment data
    sentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    sentiment_data = [int(sentiment_counts[1]), int(sentiment_counts[0]), int(sentiment_counts[-1])]

    # Prepare grouped data for aspect/topic counts by brand
    grouped_data = df.groupby(['Brands', 'Topic Label']).size().unstack(fill_value=0)
    brands_topic = grouped_data.index.tolist()
    topics = grouped_data.columns.tolist()
    topic_data = grouped_data.values.tolist()

    # Grouped data for brands by categories
    brand_data = df.groupby(['Brands', 'Category']).size().unstack(fill_value=0)
    brands = brand_data.index.tolist()
    categories = brand_data.columns.tolist()
    brand_values = brand_data.values.tolist()

    # Prepare data for total mentioned aspects
    aspect_counts = df['Topic Label'].value_counts()
    aspect_counts = aspect_counts.sort_index()#To sort the legend in alphabetical order
    aspects = aspect_counts.index.tolist()
    aspect_data = aspect_counts.values.tolist()

    # Prepare data for total mentioned categories
    category_counts = df['Category'].value_counts()
    category_counts = category_counts.sort_index()#To sort the legend in alphabetical order
    category_names = category_counts.index.tolist()
    category_data = category_counts.values.tolist()

    # Prepare data for total mentioned brands
    brand_counts = df['Brands'].value_counts()
    brand_counts = brand_counts.sort_index()#To sort the legend in alphabetical order
    brand_names = brand_counts.index.tolist()
    brand_data = brand_counts.values.tolist()

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0).sort_index()
    months = monthly_sentiment.index.astype(str).tolist()
    positive_counts = monthly_sentiment.get(1, []).tolist()
    neutral_counts = monthly_sentiment.get(0, []).tolist()
    negative_counts = monthly_sentiment.get(-1, []).tolist()

    # Pass all data to the template
    return render_template("dashboard.html", sentiment_data=sentiment_data, brands_topic=brands_topic, topics=topics,
                           topic_data=topic_data, categories=categories, brands=brands,
                           brand_values=brand_values,aspects=aspects, aspect_data=aspect_data,
                           brand_names=brand_names, brand_data=brand_data, months=months,
                           positive=positive_counts, neutral=neutral_counts, negative=negative_counts,
                           category_names=category_names, category_data=category_data)


@main.route("/adidas")
def adidas():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Filter for Adidas brand
    df = df[df['Brands'] == 'Adidas']
    # Prepare sentiment data
    adidassentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    adidassentiment_data = [int(adidassentiment_counts[1]), int(adidassentiment_counts[0]), int(adidassentiment_counts[-1])]

    # Filter for Adidas brand
    df = df[df['Brands'] == 'Adidas']
    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data = monthly_sentiment.reset_index().to_dict(orient='records')

    # Format data correctly
    new_data = []
    for record in data:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),  # Convert key to string and ensure integers
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data.append(new_record)

    adidas_df = df[df['Brands'] == 'Adidas']
    grouped_sentiments = adidas_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    grouped_sentiments = grouped_sentiments.reindex(columns=[-1, 0, 1], fill_value=0)

    topics = grouped_sentiments.index.tolist()
    negative = grouped_sentiments[-1].tolist()
    neutral = grouped_sentiments[0].tolist()
    positive = grouped_sentiments[1].tolist()

    category_sentiments = adidas_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = category_sentiments.index.tolist()
    catnegative = category_sentiments[-1].tolist()
    catneutral = category_sentiments[0].tolist()
    catpositive = category_sentiments[1].tolist()

    # Prepare adidas aspect data for pie chart
    aspect_counts = adidas_df['Topic Label'].value_counts()
    adidasaspects = aspect_counts.index.tolist()
    adidasaspect_data = aspect_counts.values.tolist()

    # Prepare category data for pie chart
    category_counts = adidas_df['Category'].value_counts()
    category_counts = category_counts.sort_index()
    adidascategories = category_counts.index.tolist()
    adidascategory_data = category_counts.values.tolist()

    return render_template("adidas.html", adidassentiment_data=adidassentiment_data, data=new_data,topics=topics,
                           negative=negative, neutral=neutral, positive=positive, categories=categories, catnegative=catnegative, catneutral=catneutral,
                           catpositive=catpositive, aspects=adidasaspects, aspect_data=adidasaspect_data, adidascategories=adidascategories, adidascategory_data=adidascategory_data)


@main.route("/nike")
def nike():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Filter for Nike brand
    df = df[df['Brands'] == 'Nike']
    # Prepare sentiment data
    nikesentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    nikesentiment_data = [int(nikesentiment_counts[1]), int(nikesentiment_counts[0]), int(nikesentiment_counts[-1])]

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data = monthly_sentiment.reset_index().to_dict(orient='records')

    # Format data correctly
    new_data = []
    for record in data:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data.append(new_record)

    nike_df = df[df['Brands'] == 'Nike']
    grouped_sentiments = nike_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    grouped_sentiments = grouped_sentiments.reindex(columns=[-1, 0, 1], fill_value=0)

    topics = grouped_sentiments.index.tolist()
    negative = grouped_sentiments[-1].tolist()
    neutral = grouped_sentiments[0].tolist()
    positive = grouped_sentiments[1].tolist()

    category_sentiments = nike_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = category_sentiments.index.tolist()
    catnegative = category_sentiments[-1].tolist()
    catneutral = category_sentiments[0].tolist()
    catpositive = category_sentiments[1].tolist()

    # Prepare nike aspect data for pie chart
    aspect_counts = nike_df['Topic Label'].value_counts()
    nikeaspects = aspect_counts.index.tolist()
    nikeaspect_data = aspect_counts.values.tolist()

    # Prepare category data for pie chart
    category_counts = nike_df['Category'].value_counts()
    category_counts = category_counts.sort_index()
    nikecategories = category_counts.index.tolist()
    nikecategory_data = category_counts.values.tolist()

    return render_template("nike.html", nikesentiment_data=nikesentiment_data, data=new_data, topics=topics,
                           negative=negative, neutral=neutral, positive=positive, categories=categories,
                           catnegative=catnegative, catneutral=catneutral,
                           catpositive=catpositive, aspects=nikeaspects, aspect_data=nikeaspect_data,
                           nikecategories=nikecategories, nikecategory_data=nikecategory_data)

@main.route("/puma")
def puma():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Filter for Puma brand
    df = df[df['Brands'] == 'Puma']
    # Prepare sentiment data
    pumasentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    pumasentiment_data = [int(pumasentiment_counts[1]), int(pumasentiment_counts[0]), int(pumasentiment_counts[-1])]

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data = monthly_sentiment.reset_index().to_dict(orient='records')

    # Format data correctly
    new_data = []
    for record in data:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data.append(new_record)

    puma_df = df[df['Brands'] == 'Puma']
    grouped_sentiments = puma_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    grouped_sentiments = grouped_sentiments.reindex(columns=[-1, 0, 1], fill_value=0)

    topics = grouped_sentiments.index.tolist()
    negative = grouped_sentiments[-1].tolist()
    neutral = grouped_sentiments[0].tolist()
    positive = grouped_sentiments[1].tolist()

    category_sentiments = puma_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = category_sentiments.index.tolist()
    catnegative = category_sentiments[-1].tolist()
    catneutral = category_sentiments[0].tolist()
    catpositive = category_sentiments[1].tolist()

    # Prepare puma aspect data for pie chart
    aspect_counts = puma_df['Topic Label'].value_counts()
    pumaaspects = aspect_counts.index.tolist()
    pumaaspect_data = aspect_counts.values.tolist()

    # Prepare category data for pie chart
    category_counts = puma_df['Category'].value_counts()
    category_counts = category_counts.sort_index()
    pumacategories = category_counts.index.tolist()
    pumacategory_data = category_counts.values.tolist()



    return render_template("puma.html", pumasentiment_data=pumasentiment_data, data=new_data, topics=topics,
                           negative=negative, neutral=neutral, positive=positive, categories=categories,
                           catnegative=catnegative, catneutral=catneutral,
                           catpositive=catpositive, aspects=pumaaspects, aspect_data=pumaaspect_data,
                           pumacategories=pumacategories, pumacategory_data=pumacategory_data)

@main.route("/catanalysis")
def catanalysis():
    global new_record
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Ensure columns are correctly typed
    df['Sentiment'] = df['Sentiment'].astype(int)
    df['Brands'] = df['Brands'].astype(str)
    df['Category'] = df['Category'].astype(str)


    # Group and pivot the data
    sentiment_pivot = df.groupby(['Brands', 'Category', 'Sentiment']).size().unstack(fill_value=0).reset_index()

    # Convert data to a structured format for JSON serialization
    categories = list(df['Category'].unique())
    brands = list(df['Brands'].unique())

    # Create a mapping of brand-category to sentiment values
    sentiment_data = {}
    for index, row in sentiment_pivot.iterrows():
        brand = row['Brands']
        category = row['Category']
        key = f'{brand}-{category}'
        sentiment_data[key] = {
            '-1': row[-1] if -1 in row else 0,
            '0': row[0] if 0 in row else 0,
            '1': row[1] if 1 in row else 0
        }

    data_dict = {
        'categories': categories,
        'brands': brands,
        'sentiment_data': sentiment_data
    }

    # Line Chart Footwear
    df_footwear = df[df['Category'] == 'Footwear']
    footwearsentiment_counts = df_footwear['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    footwearsentiment_data = [int(footwearsentiment_counts[1]), int(footwearsentiment_counts[0]),
                              int(footwearsentiment_counts[-1])]
    monthly_sentiment_footwear = df_footwear.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data_footwear = monthly_sentiment_footwear.reset_index().to_dict(orient='records')
    new_data_footwear = []
    for record in data_footwear:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data_footwear.append(new_record)

     # Line Chart Clothing
    df_clothing = df[df['Category'] == 'Clothing']
    clothingsentiment_counts = df_clothing['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    clothingsentiment_data = [int(clothingsentiment_counts[1]), int(clothingsentiment_counts[0]), int(clothingsentiment_counts[-1])]

    monthly_sentiment_clothing = df_clothing.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data_clothing = monthly_sentiment_clothing.reset_index().to_dict(orient='records')
    new_data_clothing = []
    for record in data_clothing:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data_clothing.append(new_record)

    # Line Chart Other
    df_other = df[df['Category'] == 'Other']
    othersentiment_counts = df_other['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    othersentiment_data = [int(othersentiment_counts[1]), int(othersentiment_counts[0]), int(othersentiment_counts[-1])]

    monthly_sentiment_other = df_other.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data_other = monthly_sentiment_other.reset_index().to_dict(orient='records')
    new_data_other = []
    for record in data_other:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data_other.append(new_record)

    categories_sentiment = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = categories_sentiment.index.tolist()
    positive = categories_sentiment.get(1, []).fillna(0).astype(int).tolist()
    neutral = categories_sentiment.get(0, []).fillna(0).astype(int).tolist()
    negative = categories_sentiment.get(-1, []).fillna(0).astype(int).tolist()

    categories_aspects = df.groupby(['Category', 'Topic Label']).size().unstack(fill_value=0)
    aspects = categories_aspects.columns.tolist()
    aspect_data = {aspect: categories_aspects[aspect].fillna(0).astype(int).tolist() for aspect in aspects}

    # Preparing data for brands by categories
    categories_brands = df.groupby(['Category', 'Brands']).size().unstack(fill_value=0)
    brands = categories_brands.columns.tolist()
    brand_data = {brand: categories_brands[brand].fillna(0).astype(int).tolist() for brand in brands}

    # Serialize data for JavaScript
    categories_json = json.dumps(categories)
    aspects_json = json.dumps(aspects)
    aspect_data_json = json.dumps(aspect_data)

    categories = ['Clothing','Footwear','Other']
    data2_mapping = {}

    for category in categories:
        filtered_data = df[df['Category'] == category]
        # Ensure the 'Sentiment' column is treated as categorical for proper pivoting
        filtered_data['Sentiment'] = filtered_data['Sentiment'].astype(str)
        pivot_table = filtered_data.pivot_table(index='Brands', columns='Sentiment', values='created_at',
                                                aggfunc='count', fill_value=0)
        data2_mapping[category] = {
            'brands': pivot_table.index.tolist(),
            'data': pivot_table.to_dict('list')
        }

    return render_template("catanalysis.html",
                           footwearsentiment_data=footwearsentiment_data,
                           footwear_data=new_data_footwear,
                           clothing_sentiment_data=clothingsentiment_data,
                           clothing_data=new_data_clothing,
                           other_sentiment_data=othersentiment_data,
                           other_data=new_data_other,
                           categories=categories_json,
                           positive=json.dumps(positive),
                           neutral=json.dumps(neutral),
                           negative=json.dumps(negative),
                           aspects=aspects_json,
                           aspect_data=aspect_data_json,
                           brands=json.dumps(brands),
                           brand_data=json.dumps(brand_data),
                            data2_mapping = json.dumps(data2_mapping)
                           )


@main.route("/aspanalysis")
def aspanalysis():
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Replace "Discount and Promotion" with "Discount" in the 'Topic Label' column
    df['Topic Label'] = df['Topic Label'].replace('Discount and Promotion', 'Discount')

    aspects_sentiment = df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    aspects = aspects_sentiment.index.tolist()
    positive = aspects_sentiment.get(1, []).fillna(0).astype(int).tolist()
    neutral = aspects_sentiment.get(0, []).fillna(0).astype(int).tolist()
    negative = aspects_sentiment.get(-1, []).fillna(0).astype(int).tolist()
    aspects_json = json.dumps(aspects)

    # Function to process topic data
    def process_topic_data(topic_label):
        topic_df = df[df['Topic Label'] == topic_label]

        # PIE CHART FOR BRANDS
        topicB_counts = topic_df['Brands'].value_counts()
        topicBaspects = topicB_counts.index.tolist()
        topicBaspect_data = topicB_counts.values.tolist()

        # PIE CHART FOR SENTIMENT
        sentiment_mapping = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
        topic_df['Sentiment'] = topic_df['Sentiment'].map(sentiment_mapping)
        topic_counts = topic_df['Sentiment'].value_counts()
        topicSaspects = topic_counts.index.tolist()
        topicSaspect_data = topic_counts.values.tolist()

        # LINE CHART
        df_topic = df[df['Topic Label'] == topic_label]
        topicsentiment_counts = df_topic['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
        topicsentiment_data = [int(topicsentiment_counts[1]), int(topicsentiment_counts[0]),
                               int(topicsentiment_counts[-1])]
        monthly_sentiment_topic = df_topic.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
        data_topic = monthly_sentiment_topic.reset_index().to_dict(orient='records')
        new_data_topic = []
        for record in data_topic:
            new_record = {
                'created_at': str(record['created_at']),
                'positive': int(record.get(1, 0)),
                'neutral': int(record.get(0, 0)),
                'negative': int(record.get(-1, 0)),
            }
            new_data_topic.append(new_record)

        return topicBaspects, topicBaspect_data, topicSaspects, topicSaspect_data, topicsentiment_data, new_data_topic

    # Process data for each topic
    priceBaspects, priceB_data, priceSaspects, priceS_data, price_sentiment_data, price_data = process_topic_data(
        'Price')
    comfortBaspects, comfortB_data, comfortSaspects, comfortS_data, comfort_sentiment_data, comfort_data = process_topic_data(
        'Comfortability')
    durabilityBaspects, durabilityB_data, durabilitySaspects, durabilityS_data, durability_sentiment_data, durability_data = process_topic_data(
        'Durability')
    sizeBaspects, sizeB_data, sizeSaspects, sizeS_data, size_sentiment_data, size_data = process_topic_data(
        'Size Availability')
    discountBaspects, discountB_data, discountSaspects, discountS_data, discount_sentiment_data, discount_data = process_topic_data(
        'Discount')

    aspects = ['Price', 'Comfortability', 'Durability', 'Discount', 'Size Availability']
    data_mapping = {}

    for aspect in aspects:
        filtered_data = df[df['Topic Label'] == aspect]
        # Ensure the 'Sentiment' column is treated as categorical for proper pivoting
        filtered_data['Sentiment'] = filtered_data['Sentiment'].astype(str)
        pivot_table = filtered_data.pivot_table(index='Brands', columns='Sentiment', values='created_at',
                                                aggfunc='count', fill_value=0)
        data_mapping[aspect] = {
            'brands': pivot_table.index.tolist(),
            'data': pivot_table.to_dict('list')
        }


    return render_template("aspanalysis.html",
                           aspects=aspects_json, positive=json.dumps(positive), neutral=json.dumps(neutral),
                           negative=json.dumps(negative),
                           priceBaspects=priceBaspects, priceB_data=priceB_data, priceSaspects=priceSaspects,
                           priceS_data=priceS_data, price_sentiment_data=price_sentiment_data, price_data=price_data,
                           comfortBaspects=comfortBaspects, comfortB_data=comfortB_data,
                           comfortSaspects=comfortSaspects, comfortS_data=comfortS_data,
                           comfort_sentiment_data=comfort_sentiment_data, comfort_data=comfort_data,
                           durabilityBaspects=durabilityBaspects, durabilityB_data=durabilityB_data,
                           durabilitySaspects=durabilitySaspects, durabilityS_data=durabilityS_data,
                           durability_sentiment_data=durability_sentiment_data, durability_data=durability_data,
                           sizeBaspects=sizeBaspects, sizeB_data=sizeB_data, sizeSaspects=sizeSaspects,
                           sizeS_data=sizeS_data, size_sentiment_data=size_sentiment_data, size_data=size_data,
                           discountBaspects=discountBaspects, discountB_data=discountB_data,
                           discountSaspects=discountSaspects, discountS_data=discountS_data,
                           discount_sentiment_data=discount_sentiment_data, discount_data=discount_data,
                           data_mapping=json.dumps(data_mapping)
                           )


@main.route("/companalysis")
def companalysis():
    return render_template("companalysis.html")


@main.route("/generate_charts", methods=["POST"])
def generate_charts():
    # Load the dataset
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Map sentiment values to labels
    df['Sentiment'] = df['Sentiment'].map({1: 'Positive', 0: 'Neutral', -1: 'Negative'})

    # Ensure 'created_at' column exists and handle invalid/missing dates
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
        df = df.dropna(subset=['created_at'])  # Drop rows where 'created_at' could not be parsed
        df['created_at'] = df['created_at'].dt.to_period('M')
    else:
        print("Error: 'created_at' column is missing in the dataset.")
        return jsonify({'error': "'created_at' column is missing in the dataset."})


    brand1 = request.json.get("brand1")
    brand2 = request.json.get("brand2")

    def get_sentiment_distribution(brand):
        brand_df = df[df['Brands'] == brand]
        sentiment_counts = brand_df['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
        return sentiment_counts.to_dict()

    def get_aspect_sentiment(brand):
        brand_df = df[df['Brands'] == brand]
        aspect_sentiment = brand_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
        return aspect_sentiment.to_dict(orient='index')

    def get_category_sentiment(brand):
        brand_df = df[df['Brands'] == brand]
        category_sentiment = brand_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
        return category_sentiment.to_dict(orient='index')

    def get_monthly_sentiment_count(brand, sentiment):
        brand_df = df[(df['Brands'] == brand) & (df['Sentiment'] == sentiment)]
        monthly_sentiment = brand_df.groupby('created_at').size()
        monthly_sentiment.index = monthly_sentiment.index.astype(str)  # Convert PeriodIndex to string
        return monthly_sentiment.to_dict()

    def get_all_monthly_sentiments(brand):
        positive_counts = get_monthly_sentiment_count(brand, 'Positive')
        neutral_counts = get_monthly_sentiment_count(brand, 'Neutral')
        negative_counts = get_monthly_sentiment_count(brand, 'Negative')

        total_counts = {}
        for month in set(positive_counts.keys()).union(neutral_counts.keys()).union(negative_counts.keys()):
            total_counts[month] = positive_counts.get(month, 0) + neutral_counts.get(month, 0) + negative_counts.get(
                month, 0)

        return {
            'total': total_counts,
            'positive': positive_counts,
            'neutral': neutral_counts,
            'negative': negative_counts
        }

    pie_data = {
        "brand1": get_sentiment_distribution(brand1),
        "brand2": get_sentiment_distribution(brand2)
    }

    bar_data = {
        "brand1": get_aspect_sentiment(brand1),
        "brand2": get_aspect_sentiment(brand2)
    }

    barcategory_data = {
        "brand1": get_category_sentiment(brand1),
        "brand2": get_category_sentiment(brand2)
    }

    monthly_data = {
        "brand1": get_all_monthly_sentiments(brand1),
        "brand2": get_all_monthly_sentiments(brand2)
    }

    response_data = {
        'pie': pie_data,
        'bar': bar_data,
        'barcategory': barcategory_data,
        'line': monthly_data
    }

    print(response_data)  # Print the response data for debugging

    return jsonify(response_data)

@main.route("/intro")
def intro():
    return render_template("intro.html")


@main.route("/twitter")
def twitter():
    return render_template("twitter.html")

# Define stopwords in Bahasa Malaysia and Bahasa Indonesia
malay_indonesian_stopwords = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'adalah', 'itu', 'dengan',
    'bahwa', 'ini', 'atau', 'oleh', 'sebagai', 'mereka', 'kami', 'kita', 'saya',
    'aku', 'anda', 'dia', 'kamu', 'apa', 'mengapa', 'bagaimana', 'bilakah',
    'dimana', 'apakah', 'mengapa', 'bagaimana', 'bilamana', 'dengan', 'ketika',
    'jika', 'jikalau', 'kenapa', 'kapan', 'namun', 'tetapi', 'adalah', 'ialah',
    'itu', 'ini', 'bagi', 'dalam', 'antara', 'tersebut', 'pada', 'dengan',
    'tanpa', 'tanpa', 'sehingga', 'karena', 'tentang', 'seperti', 'setelah',
    'sebelum', 'sekitar', 'walaupun', 'meskipun', 'bahkan', 'juga', 'lagi',
    'lebih', 'harus', 'bisa', 'dapat', 'akan', 'selalu', 'segera', 'setiap',
    'selama', 'seperti', 'demikian', 'kemudian', 'lain', 'dalam', 'seluruh',
    'semua', 'sering', 'kadang', 'pernah', 'mungkin', 'hampir', 'hanya',
    'berbagai', 'lainnya', 'pula', 'sudah', 'belum', 'tidak', 'bukan', 'tanpa'
])

# Define the lists of custom sentiment words
positive_words = set([
    'good', 'happy', 'awesome', 'amazing', 'joy', 'love', 'excellent', 'like', 'durable', 'comfy',
    'bagus', 'gembira', 'hebat', 'menakjubkan', 'seronok', 'suka', 'cemerlang', 'selesa', 'tahan lama', 'berkualiti',
    'mantap', 'berstamina', 'fleksibel', 'bergaya', 'senang', 'praktikal', 'inovatif','murah','promosi','promo','diskaun','discount'
])
neutral_words = set([
    'okay', 'fine', 'normal', 'common', 'standard',
    'ok', 'biasa', 'umum', 'standard', 'rata-rata', 'biasa-biasa', 'sering'
])
negative_words = set([
    'bad', 'sad', 'terrible', 'awful', 'hate', 'dislike', 'poor',
    'buruk', 'sedih', 'mahal', 'mengerikan', 'menjijikkan', 'benci', 'tidak suka', 'lemah', 'cepat rosak', 'tidak tahan lama',
    'keras', 'menyakitkan', 'kaku', 'tidak fleksibel','tak tahan lama', 'takselesa', 'taksuka','rosak'
])
# Utility function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    # Expand common contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "i'm": "i am", "he's": "he is", "she's": "she is", "it's": "it is",
        "that's": "that is", "there's": "there is", "what's": "what is"
    }
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    text = pattern.sub(lambda x: contractions[x.group()], text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english')).union(malay_indonesian_stopwords)  # Combine stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def custom_sentiment_analysis(text):
    words = text.split()
    pos_count = sum(1 for word in words if word in positive_words)
    neu_count = sum(1 for word in words if word in neutral_words)
    neg_count = sum(1 for word in words if word in negative_words)

    # Calculate polarity
    if pos_count + neg_count == 0:
        polarity_score = 0
    else:
        polarity_score = (pos_count - neg_count) / (pos_count + neg_count)
    return polarity_score, pos_count, neu_count, neg_count

@main.route('/analyzer', methods=['GET', 'POST'])
def analyzer():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)

        # Perform custom sentiment analysis
        custom_polarity, pos_count, neu_count, neg_count = custom_sentiment_analysis(processed_text)

        # Use TextBlob for additional sentiment analysis
        blob = TextBlob(processed_text)
        blob_polarity = blob.sentiment.polarity

        # Combine results
        final_polarity = (custom_polarity + blob_polarity) / 2

        # Determine sentiment category
        if final_polarity > 0:
            sentiment_category = 'Positive'
        elif final_polarity < 0:
            sentiment_category = 'Negative'
        else:
            sentiment_category = 'Neutral'

        return jsonify({
            'sentiment': sentiment_category,
            'polarity': round(final_polarity, 2),
            'details': {
                'positive': pos_count,
                'neutral': neu_count,
                'negative': neg_count,
                'textblob_polarity': round(blob_polarity, 2),
                'custom_polarity': round(custom_polarity, 2)
            }
        })
    else:
        return render_template("analyzer.html")


if __name__ == '__main__':
    main.run(debug=True)
=======
import requests
from flask import Flask, render_template, jsonify
from flask import request
import pandas as pd
import json
from textblob import TextBlob
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from langdetect import detect, LangDetectException
from googletrans import Translator

main = Flask(__name__)


@main.route("/")
def home():
    return render_template("homepage.html")


@main.route("/dashboard")
def dashboard():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')

    # Convert 'created_at' to datetime with specified format and group by month
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y').dt.to_period('M')

    # Prepare sentiment data
    sentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    sentiment_data = [int(sentiment_counts[1]), int(sentiment_counts[0]), int(sentiment_counts[-1])]

    # Prepare grouped data for aspect/topic counts by brand
    grouped_data = df.groupby(['Brands', 'Topic Label']).size().unstack(fill_value=0)
    brands_topic = grouped_data.index.tolist()
    topics = grouped_data.columns.tolist()
    topic_data = grouped_data.values.tolist()

    # Grouped data for brands by categories
    brand_data = df.groupby(['Brands', 'Category']).size().unstack(fill_value=0)
    brands = brand_data.index.tolist()
    categories = brand_data.columns.tolist()
    brand_values = brand_data.values.tolist()

    # Prepare data for total mentioned aspects
    aspect_counts = df['Topic Label'].value_counts()
    aspect_counts = aspect_counts.sort_index()#To sort the legend in alphabetical order
    aspects = aspect_counts.index.tolist()
    aspect_data = aspect_counts.values.tolist()

    # Prepare data for total mentioned categories
    category_counts = df['Category'].value_counts()
    category_counts = category_counts.sort_index()#To sort the legend in alphabetical order
    category_names = category_counts.index.tolist()
    category_data = category_counts.values.tolist()

    # Prepare data for total mentioned brands
    brand_counts = df['Brands'].value_counts()
    brand_counts = brand_counts.sort_index()#To sort the legend in alphabetical order
    brand_names = brand_counts.index.tolist()
    brand_data = brand_counts.values.tolist()

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0).sort_index()
    months = monthly_sentiment.index.astype(str).tolist()
    positive_counts = monthly_sentiment.get(1, []).tolist()
    neutral_counts = monthly_sentiment.get(0, []).tolist()
    negative_counts = monthly_sentiment.get(-1, []).tolist()

    # Pass all data to the template
    return render_template("dashboard.html", sentiment_data=sentiment_data, brands_topic=brands_topic, topics=topics,
                           topic_data=topic_data, categories=categories, brands=brands,
                           brand_values=brand_values,aspects=aspects, aspect_data=aspect_data,
                           brand_names=brand_names, brand_data=brand_data, months=months,
                           positive=positive_counts, neutral=neutral_counts, negative=negative_counts,
                           category_names=category_names, category_data=category_data)


@main.route("/adidas")
def adidas():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Filter for Adidas brand
    df = df[df['Brands'] == 'Adidas']
    # Prepare sentiment data
    adidassentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    adidassentiment_data = [int(adidassentiment_counts[1]), int(adidassentiment_counts[0]), int(adidassentiment_counts[-1])]

    # Filter for Adidas brand
    df = df[df['Brands'] == 'Adidas']
    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data = monthly_sentiment.reset_index().to_dict(orient='records')

    # Format data correctly
    new_data = []
    for record in data:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),  # Convert key to string and ensure integers
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data.append(new_record)

    adidas_df = df[df['Brands'] == 'Adidas']
    grouped_sentiments = adidas_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    grouped_sentiments = grouped_sentiments.reindex(columns=[-1, 0, 1], fill_value=0)

    topics = grouped_sentiments.index.tolist()
    negative = grouped_sentiments[-1].tolist()
    neutral = grouped_sentiments[0].tolist()
    positive = grouped_sentiments[1].tolist()

    category_sentiments = adidas_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = category_sentiments.index.tolist()
    catnegative = category_sentiments[-1].tolist()
    catneutral = category_sentiments[0].tolist()
    catpositive = category_sentiments[1].tolist()

    # Prepare adidas aspect data for pie chart
    aspect_counts = adidas_df['Topic Label'].value_counts()
    adidasaspects = aspect_counts.index.tolist()
    adidasaspect_data = aspect_counts.values.tolist()

    # Prepare category data for pie chart
    category_counts = adidas_df['Category'].value_counts()
    category_counts = category_counts.sort_index()
    adidascategories = category_counts.index.tolist()
    adidascategory_data = category_counts.values.tolist()

    return render_template("adidas.html", adidassentiment_data=adidassentiment_data, data=new_data,topics=topics,
                           negative=negative, neutral=neutral, positive=positive, categories=categories, catnegative=catnegative, catneutral=catneutral,
                           catpositive=catpositive, aspects=adidasaspects, aspect_data=adidasaspect_data, adidascategories=adidascategories, adidascategory_data=adidascategory_data)


@main.route("/nike")
def nike():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Filter for Nike brand
    df = df[df['Brands'] == 'Nike']
    # Prepare sentiment data
    nikesentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    nikesentiment_data = [int(nikesentiment_counts[1]), int(nikesentiment_counts[0]), int(nikesentiment_counts[-1])]

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data = monthly_sentiment.reset_index().to_dict(orient='records')

    # Format data correctly
    new_data = []
    for record in data:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data.append(new_record)

    nike_df = df[df['Brands'] == 'Nike']
    grouped_sentiments = nike_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    grouped_sentiments = grouped_sentiments.reindex(columns=[-1, 0, 1], fill_value=0)

    topics = grouped_sentiments.index.tolist()
    negative = grouped_sentiments[-1].tolist()
    neutral = grouped_sentiments[0].tolist()
    positive = grouped_sentiments[1].tolist()

    category_sentiments = nike_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = category_sentiments.index.tolist()
    catnegative = category_sentiments[-1].tolist()
    catneutral = category_sentiments[0].tolist()
    catpositive = category_sentiments[1].tolist()

    # Prepare nike aspect data for pie chart
    aspect_counts = nike_df['Topic Label'].value_counts()
    nikeaspects = aspect_counts.index.tolist()
    nikeaspect_data = aspect_counts.values.tolist()

    # Prepare category data for pie chart
    category_counts = nike_df['Category'].value_counts()
    category_counts = category_counts.sort_index()
    nikecategories = category_counts.index.tolist()
    nikecategory_data = category_counts.values.tolist()

    return render_template("nike.html", nikesentiment_data=nikesentiment_data, data=new_data, topics=topics,
                           negative=negative, neutral=neutral, positive=positive, categories=categories,
                           catnegative=catnegative, catneutral=catneutral,
                           catpositive=catpositive, aspects=nikeaspects, aspect_data=nikeaspect_data,
                           nikecategories=nikecategories, nikecategory_data=nikecategory_data)

@main.route("/puma")
def puma():
    # Load data
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Filter for Puma brand
    df = df[df['Brands'] == 'Puma']
    # Prepare sentiment data
    pumasentiment_counts = df['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    pumasentiment_data = [int(pumasentiment_counts[1]), int(pumasentiment_counts[0]), int(pumasentiment_counts[-1])]

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Group by month and sentiment
    monthly_sentiment = df.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data = monthly_sentiment.reset_index().to_dict(orient='records')

    # Format data correctly
    new_data = []
    for record in data:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data.append(new_record)

    puma_df = df[df['Brands'] == 'Puma']
    grouped_sentiments = puma_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    grouped_sentiments = grouped_sentiments.reindex(columns=[-1, 0, 1], fill_value=0)

    topics = grouped_sentiments.index.tolist()
    negative = grouped_sentiments[-1].tolist()
    neutral = grouped_sentiments[0].tolist()
    positive = grouped_sentiments[1].tolist()

    category_sentiments = puma_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = category_sentiments.index.tolist()
    catnegative = category_sentiments[-1].tolist()
    catneutral = category_sentiments[0].tolist()
    catpositive = category_sentiments[1].tolist()

    # Prepare puma aspect data for pie chart
    aspect_counts = puma_df['Topic Label'].value_counts()
    pumaaspects = aspect_counts.index.tolist()
    pumaaspect_data = aspect_counts.values.tolist()

    # Prepare category data for pie chart
    category_counts = puma_df['Category'].value_counts()
    category_counts = category_counts.sort_index()
    pumacategories = category_counts.index.tolist()
    pumacategory_data = category_counts.values.tolist()



    return render_template("puma.html", pumasentiment_data=pumasentiment_data, data=new_data, topics=topics,
                           negative=negative, neutral=neutral, positive=positive, categories=categories,
                           catnegative=catnegative, catneutral=catneutral,
                           catpositive=catpositive, aspects=pumaaspects, aspect_data=pumaaspect_data,
                           pumacategories=pumacategories, pumacategory_data=pumacategory_data)

@main.route("/catanalysis")
def catanalysis():
    global new_record
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Line Chart Footwear
    df_footwear = df[df['Category'] == 'Footwear']
    footwearsentiment_counts = df_footwear['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    footwearsentiment_data = [int(footwearsentiment_counts[1]), int(footwearsentiment_counts[0]),
                              int(footwearsentiment_counts[-1])]
    monthly_sentiment_footwear = df_footwear.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data_footwear = monthly_sentiment_footwear.reset_index().to_dict(orient='records')
    new_data_footwear = []
    for record in data_footwear:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data_footwear.append(new_record)

     # Line Chart Clothing
    df_clothing = df[df['Category'] == 'Clothing']
    clothingsentiment_counts = df_clothing['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    clothingsentiment_data = [int(clothingsentiment_counts[1]), int(clothingsentiment_counts[0]), int(clothingsentiment_counts[-1])]

    monthly_sentiment_clothing = df_clothing.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data_clothing = monthly_sentiment_clothing.reset_index().to_dict(orient='records')
    new_data_clothing = []
    for record in data_clothing:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data_clothing.append(new_record)

    # Line Chart Other
    df_other = df[df['Category'] == 'Other']
    othersentiment_counts = df_other['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
    othersentiment_data = [int(othersentiment_counts[1]), int(othersentiment_counts[0]), int(othersentiment_counts[-1])]

    monthly_sentiment_other = df_other.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
    data_other = monthly_sentiment_other.reset_index().to_dict(orient='records')
    new_data_other = []
    for record in data_other:
        new_record = {
            'created_at': str(record['created_at']),
            'positive': int(record.get(1, 0)),
            'neutral': int(record.get(0, 0)),
            'negative': int(record.get(-1, 0)),
        }
        new_data_other.append(new_record)

    categories_sentiment = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    categories = categories_sentiment.index.tolist()
    positive = categories_sentiment.get(1, []).fillna(0).astype(int).tolist()
    neutral = categories_sentiment.get(0, []).fillna(0).astype(int).tolist()
    negative = categories_sentiment.get(-1, []).fillna(0).astype(int).tolist()

    categories_aspects = df.groupby(['Category', 'Topic Label']).size().unstack(fill_value=0)
    aspects = categories_aspects.columns.tolist()
    aspect_data = {aspect: categories_aspects[aspect].fillna(0).astype(int).tolist() for aspect in aspects}

    # Preparing data for brands by categories
    categories_brands = df.groupby(['Category', 'Brands']).size().unstack(fill_value=0)
    brands = categories_brands.columns.tolist()
    brand_data = {brand: categories_brands[brand].fillna(0).astype(int).tolist() for brand in brands}

    # Serialize data for JavaScript
    categories_json = json.dumps(categories)
    aspects_json = json.dumps(aspects)
    aspect_data_json = json.dumps(aspect_data)


    return render_template("catanalysis.html",
                           footwearsentiment_data=footwearsentiment_data,
                           footwear_data=new_data_footwear,
                           clothing_sentiment_data=clothingsentiment_data,
                           clothing_data=new_data_clothing,
                           other_sentiment_data=othersentiment_data,
                           other_data=new_data_other,
                           categories=categories_json,
                           positive=json.dumps(positive),
                           neutral=json.dumps(neutral),
                           negative=json.dumps(negative),
                           aspects=aspects_json,
                           aspect_data=aspect_data_json,
                           brands=json.dumps(brands),
                           brand_data=json.dumps(brand_data)
                           )


@main.route("/aspanalysis")
def aspanalysis():
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')

    # Convert 'created_at' to datetime with specified format
    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    # Convert datetime to Period (month) for grouping, assuming timezone is not critical
    df['created_at'] = df['created_at'].dt.to_period('M')

    # Replace "Discount and Promotion" with "Discount" in the 'Topic Label' column
    df['Topic Label'] = df['Topic Label'].replace('Discount and Promotion', 'Discount')

    aspects_sentiment = df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
    aspects = aspects_sentiment.index.tolist()
    positive = aspects_sentiment.get(1, []).fillna(0).astype(int).tolist()
    neutral = aspects_sentiment.get(0, []).fillna(0).astype(int).tolist()
    negative = aspects_sentiment.get(-1, []).fillna(0).astype(int).tolist()
    aspects_json = json.dumps(aspects)

    # Function to process topic data
    def process_topic_data(topic_label):
        topic_df = df[df['Topic Label'] == topic_label]

        # PIE CHART FOR BRANDS
        topicB_counts = topic_df['Brands'].value_counts()
        topicBaspects = topicB_counts.index.tolist()
        topicBaspect_data = topicB_counts.values.tolist()

        # PIE CHART FOR SENTIMENT
        sentiment_mapping = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
        topic_df['Sentiment'] = topic_df['Sentiment'].map(sentiment_mapping)
        topic_counts = topic_df['Sentiment'].value_counts()
        topicSaspects = topic_counts.index.tolist()
        topicSaspect_data = topic_counts.values.tolist()

        # LINE CHART
        df_topic = df[df['Topic Label'] == topic_label]
        topicsentiment_counts = df_topic['Sentiment'].value_counts().reindex([1, 0, -1], fill_value=0)
        topicsentiment_data = [int(topicsentiment_counts[1]), int(topicsentiment_counts[0]),
                               int(topicsentiment_counts[-1])]
        monthly_sentiment_topic = df_topic.groupby(['created_at', 'Sentiment']).size().unstack(fill_value=0)
        data_topic = monthly_sentiment_topic.reset_index().to_dict(orient='records')
        new_data_topic = []
        for record in data_topic:
            new_record = {
                'created_at': str(record['created_at']),
                'positive': int(record.get(1, 0)),
                'neutral': int(record.get(0, 0)),
                'negative': int(record.get(-1, 0)),
            }
            new_data_topic.append(new_record)

        return topicBaspects, topicBaspect_data, topicSaspects, topicSaspect_data, topicsentiment_data, new_data_topic

    # Process data for each topic
    priceBaspects, priceB_data, priceSaspects, priceS_data, price_sentiment_data, price_data = process_topic_data(
        'Price')
    comfortBaspects, comfortB_data, comfortSaspects, comfortS_data, comfort_sentiment_data, comfort_data = process_topic_data(
        'Comfortability')
    durabilityBaspects, durabilityB_data, durabilitySaspects, durabilityS_data, durability_sentiment_data, durability_data = process_topic_data(
        'Durability')
    sizeBaspects, sizeB_data, sizeSaspects, sizeS_data, size_sentiment_data, size_data = process_topic_data(
        'Size Availability')
    discountBaspects, discountB_data, discountSaspects, discountS_data, discount_sentiment_data, discount_data = process_topic_data(
        'Discount')

    return render_template("aspanalysis.html",
                           aspects=aspects_json, positive=json.dumps(positive), neutral=json.dumps(neutral),
                           negative=json.dumps(negative),
                           priceBaspects=priceBaspects, priceB_data=priceB_data, priceSaspects=priceSaspects,
                           priceS_data=priceS_data, price_sentiment_data=price_sentiment_data, price_data=price_data,
                           comfortBaspects=comfortBaspects, comfortB_data=comfortB_data,
                           comfortSaspects=comfortSaspects, comfortS_data=comfortS_data,
                           comfort_sentiment_data=comfort_sentiment_data, comfort_data=comfort_data,
                           durabilityBaspects=durabilityBaspects, durabilityB_data=durabilityB_data,
                           durabilitySaspects=durabilitySaspects, durabilityS_data=durabilityS_data,
                           durability_sentiment_data=durability_sentiment_data, durability_data=durability_data,
                           sizeBaspects=sizeBaspects, sizeB_data=sizeB_data, sizeSaspects=sizeSaspects,
                           sizeS_data=sizeS_data, size_sentiment_data=size_sentiment_data, size_data=size_data,
                           discountBaspects=discountBaspects, discountB_data=discountB_data,
                           discountSaspects=discountSaspects, discountS_data=discountS_data,
                           discount_sentiment_data=discount_sentiment_data, discount_data=discount_data,
                           )


@main.route("/companalysis")
def companalysis():
    return render_template("companalysis.html")


@main.route("/generate_charts", methods=["POST"])
def generate_charts():
    # Load the dataset
    df = pd.read_csv('static/Combined_Brands_Dataset_NB.csv')
    # Map sentiment values to labels
    df['Sentiment'] = df['Sentiment'].map({1: 'Positive', 0: 'Neutral', -1: 'Negative'})

    # Ensure 'created_at' column exists and handle invalid/missing dates
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
        df = df.dropna(subset=['created_at'])  # Drop rows where 'created_at' could not be parsed
        df['created_at'] = df['created_at'].dt.to_period('M')
    else:
        print("Error: 'created_at' column is missing in the dataset.")
        return jsonify({'error': "'created_at' column is missing in the dataset."})


    brand1 = request.json.get("brand1")
    brand2 = request.json.get("brand2")

    def get_sentiment_distribution(brand):
        brand_df = df[df['Brands'] == brand]
        sentiment_counts = brand_df['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
        return sentiment_counts.to_dict()

    def get_aspect_sentiment(brand):
        brand_df = df[df['Brands'] == brand]
        aspect_sentiment = brand_df.groupby(['Topic Label', 'Sentiment']).size().unstack(fill_value=0)
        return aspect_sentiment.to_dict(orient='index')

    def get_category_sentiment(brand):
        brand_df = df[df['Brands'] == brand]
        category_sentiment = brand_df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
        return category_sentiment.to_dict(orient='index')

    def get_monthly_sentiment_count(brand, sentiment):
        brand_df = df[(df['Brands'] == brand) & (df['Sentiment'] == sentiment)]
        monthly_sentiment = brand_df.groupby('created_at').size()
        monthly_sentiment.index = monthly_sentiment.index.astype(str)  # Convert PeriodIndex to string
        return monthly_sentiment.to_dict()

    def get_all_monthly_sentiments(brand):
        positive_counts = get_monthly_sentiment_count(brand, 'Positive')
        neutral_counts = get_monthly_sentiment_count(brand, 'Neutral')
        negative_counts = get_monthly_sentiment_count(brand, 'Negative')

        total_counts = {}
        for month in set(positive_counts.keys()).union(neutral_counts.keys()).union(negative_counts.keys()):
            total_counts[month] = positive_counts.get(month, 0) + neutral_counts.get(month, 0) + negative_counts.get(
                month, 0)

        return {
            'total': total_counts,
            'positive': positive_counts,
            'neutral': neutral_counts,
            'negative': negative_counts
        }

    pie_data = {
        "brand1": get_sentiment_distribution(brand1),
        "brand2": get_sentiment_distribution(brand2)
    }

    bar_data = {
        "brand1": get_aspect_sentiment(brand1),
        "brand2": get_aspect_sentiment(brand2)
    }

    barcategory_data = {
        "brand1": get_category_sentiment(brand1),
        "brand2": get_category_sentiment(brand2)
    }

    monthly_data = {
        "brand1": get_all_monthly_sentiments(brand1),
        "brand2": get_all_monthly_sentiments(brand2)
    }

    response_data = {
        'pie': pie_data,
        'bar': bar_data,
        'barcategory': barcategory_data,
        'line': monthly_data
    }

    print(response_data)  # Print the response data for debugging

    return jsonify(response_data)

@main.route("/intro")
def intro():
    return render_template("intro.html")


@main.route("/twitter")
def twitter():
    return render_template("twitter.html")




@main.route('/analyzer', methods=['GET', 'POST'])
def analyzer():
    if request.method == 'POST':
        text = request.form['text']
        translator = Translator()

        # Detect the language of the text
        try:
            language = detect(text)
        except LangDetectException:
            return jsonify({'error': 'Language detection failed'}), 400

        # Translate to English if the text is in Indonesian
        if language == 'id':
            translated = translator.translate(text, src='id', dest='en')
            text = translated.text

        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # -1 to 1 where -1 is negative and 1 is positive
        if sentiment > 0:
            sentiment_category = 'Positive'
        elif sentiment < 0:
            sentiment_category = 'Negative'
        else:
            sentiment_category = 'Neutral'

        return jsonify({
            'sentiment': sentiment_category,
            'polarity': round(sentiment, 2)
        })
    else:
        return render_template("analyzer.html")



if __name__ == '__main__':
    main.run(debug=True)
>>>>>>> origin/main
