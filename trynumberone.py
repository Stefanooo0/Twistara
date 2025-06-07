import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import datetime
import requests # For OpenRouter API calls
import google.generativeai as genai # For Gemini API calls

# --- Configuration ---
# Set page config for wider layout
st.set_page_config(layout="wide", page_title="Media Intelligence Dashboard")

# --- Constants ---
REQUIRED_COLUMNS = [
    'Date', 'Platform', 'Sentiment', 'Location', 'Engagements',
    'Media Type', 'Influencer Brand', 'Post Type'
]

# OpenRouter Models (subset for example)
OPENROUTER_MODELS = {
    'OpenAI GPT-3.5 Turbo': 'openai/gpt-3.5-turbo',
    'OpenAI GPT-4o': 'openai/gpt-4o',
    'Google Gemini Pro': 'google/gemini-pro',
    'Anthropic Claude 3 Opus': 'anthropic/claude-3-opus',
    'Mistral 7B Instruct': 'mistralai/mistral-7b-instruct',
}

# API Keys (user will input them)
# For Gemini, the API key is usually handled by `genai.configure(api_key=...)`
# For OpenRouter, it's passed in the header

# --- Helper Functions ---

def parse_csv(csv_text):
    """
    Parses CSV text into a pandas DataFrame.
    """
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def clean_data(df):
    """
    Cleans and normalizes the DataFrame.
    - Converts 'Date' to datetime.
    - Fills missing 'Engagements' with 0.
    - Normalizes column names.
    - Filters out rows with invalid dates.
    """
    normalized_df = df.copy()

    # Normalize column names
    normalized_df.columns = [col.lower().replace(' ', '').replace('_', '') for col in normalized_df.columns]

    # Check for required columns after normalization
    missing_cols = [col.lower().replace(' ', '').replace('_', '') for col in REQUIRED_COLUMNS if col.lower().replace(' ', '').replace('_', '') not in normalized_df.columns]
    if missing_cols:
        st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}. Please ensure your CSV has: {', '.join(REQUIRED_COLUMNS)}")
        return pd.DataFrame() # Return empty DataFrame on error

    # Convert 'date' to datetime
    if 'date' in normalized_df.columns:
        normalized_df['date'] = pd.to_datetime(normalized_df['date'], errors='coerce')
        normalized_df = normalized_df.dropna(subset=['date']) # Drop rows where date conversion failed
    else:
        st.warning("'Date' column not found or normalized. Date-based charts and filters will not work.")
        normalized_df['date'] = pd.NaT # Add a NaT column if missing

    # Fill missing 'engagements' with 0 and convert to int
    if 'engagements' in normalized_df.columns:
        normalized_df['engagements'] = pd.to_numeric(normalized_df['engagements'], errors='coerce').fillna(0).astype(int)
    else:
        st.warning("'Engagements' column not found or normalized. Engagement-based charts might be inaccurate.")
        normalized_df['engagements'] = 0 # Add a zero column if missing

    # Ensure other required fields exist, filling with 'Unknown'
    for col in ['platform', 'sentiment', 'location', 'mediatype', 'influencerbrand', 'posttype']:
        if col not in normalized_df.columns:
            normalized_df[col] = 'Unknown'
        normalized_df[col] = normalized_df[col].fillna('Unknown').astype(str) # Ensure string type

    return normalized_df

def generate_insights_sentiment(data_df):
    insights = []
    if not data_df.empty and 'sentiment' in data_df.columns:
        sentiment_counts = data_df['sentiment'].value_counts()
        total_sentiments = sentiment_counts.sum()
        if total_sentiments > 0:
            sorted_sentiments = sentiment_counts.sort_values(ascending=False)
            insights.append(f"The dominant sentiment is '{sorted_sentiments.index[0]}' accounting for {((sorted_sentiments.iloc[0] / total_sentiments) * 100):.2f}% of all posts.")
            if len(sorted_sentiments) > 1:
                insights.append(f"The second most common sentiment is '{sorted_sentiments.index[1]}' with {((sorted_sentiments.iloc[1] / total_sentiments) * 100):.2f}% of posts.")
            if len(sorted_sentiments) > 2:
                insights.append(f"Combined, the top two sentiments represent over {(((sorted_sentiments.iloc[0] + sorted_sentiments.iloc[1]) / total_sentiments) * 100):.2f}% of the total.")
    else:
        insights.append("No sentiment data available for analysis.")
    return insights

def generate_insights_engagement_trend(data_df):
    insights = []
    if not data_df.empty and 'date' in data_df.columns and 'engagements' in data_df.columns:
        # Aggregate engagements by date (daily total)
        daily_engagements = data_df.groupby(data_df['date'].dt.date)['engagements'].sum().sort_index()

        if not daily_engagements.empty:
            max_engagement = daily_engagements.max()
            min_engagement = daily_engagements.min()
            avg_engagement = daily_engagements.mean()
            peak_date = daily_engagements.idxmax().strftime('%Y-%m-%d')

            insights.append(f"The highest daily engagement recorded was {max_engagement:,} on {peak_date}.")
            insights.append(f"The average daily engagement across the period is approximately {avg_engagement:,.0f}.")

            if len(daily_engagements) > 1:
                first_engagement = daily_engagements.iloc[0]
                last_engagement = daily_engagements.iloc[-1]
                if last_engagement > first_engagement:
                    insights.append('There appears to be an upward trend in overall engagements over the observed period.')
                elif last_engagement < first_engagement:
                    insights.append('There appears to be a downward trend in overall engagements over the observed period.')
                else:
                    insights.append('Engagements remained relatively stable across the period, with no significant trend.')
    else:
        insights.append('No engagement data available to analyze trend.')
    return insights


def generate_insights_platform_engagement(data_df):
    insights = []
    if not data_df.empty and 'platform' in data_df.columns and 'engagements' in data_df.columns:
        platform_engagements = data_df.groupby('platform')['engagements'].sum().sort_values(ascending=False)
        if not platform_engagements.empty:
            total_engagements = platform_engagements.sum()
            insights.append(f"The platform '{platform_engagements.index[0]}' leads with {platform_engagements.iloc[0]:,} engagements, representing {((platform_engagements.iloc[0] / total_engagements) * 100):.2f}% of the total.")
            if len(platform_engagements) > 1:
                insights.append(f"The second most engaging platform is '{platform_engagements.index[1]}' with {platform_engagements.iloc[1]:,} engagements.")
            if len(platform_engagements) > 2:
                insights.append("The top three platforms collectively account for a significant portion of total engagements.")
    else:
        insights.append('No platform engagement data available.')
    return insights

def generate_insights_media_type(data_df):
    insights = []
    if not data_df.empty and 'mediatype' in data_df.columns:
        media_type_counts = data_df['mediatype'].value_counts()
        total_media_types = media_type_counts.sum()
        if total_media_types > 0:
            sorted_media_types = media_type_counts.sort_values(ascending=False)
            insights.append(f"The most prevalent media type is '{sorted_media_types.index[0]}', accounting for {((sorted_media_types.iloc[0] / total_media_types) * 100):.2f}% of all posts.")
            if len(sorted_media_types) > 1:
                insights.append(f"The second most common media type is '{sorted_media_types.index[1]}' with {((sorted_media_types.iloc[1] / total_media_types) * 100):.2f}% of posts.")
            if len(sorted_media_types) > 2:
                insights.append("Content strategy appears to focus heavily on the top few media types, which combined make up a significant majority.")
    else:
        insights.append('No media type data available for analysis.')
    return insights

def generate_insights_top_locations(data_df):
    insights = []
    if not data_df.empty and 'location' in data_df.columns:
        location_counts = data_df['location'].value_counts().head(5)
        total_posts = data_df.shape[0]
        if total_posts > 0 and not location_counts.empty:
            insights.append(f"The top location for posts is '{location_counts.index[0]}' with {location_counts.iloc[0]} posts, representing {((location_counts.iloc[0] / total_posts) * 100):.2f}% of all posts.")
            if len(location_counts) > 1:
                insights.append(f"The second most active location is '{location_counts.index[1]}' with {location_counts.iloc[1]} posts.")
            if len(location_counts) > 2:
                insights.append("The top locations indicate primary areas of content activity or audience presence.")
    else:
        insights.append('No location data available for analysis.')
    return insights

def generate_data_summary_and_recommendations(insights_list, ai_source, openrouter_api_key, openrouter_model):
    """
    Generates a concise summary of the data and campaign recommendations using an LLM.
    """
    full_insights_text = "\n".join(insights_list)
    if not full_insights_text.strip():
        return "No specific insights to summarize.", "No recommendations can be generated without data."

    st.session_state.generating_summary = True
    st.session_state.generating_recommendations = True
    st.session_state.data_summary = "Generating summary..."
    st.session_state.campaign_recommendations = "Generating recommendations..."
    st.rerun() # Rerun to update UI with loading state

    prompt = f"""Based on the following media intelligence insights, provide:
    1. A concise executive summary of the key findings.
    2. Actionable campaign recommendations to optimize future strategies, focusing on maximizing engagement and positive sentiment across platforms.
    
    Insights:
    {full_insights_text}
    
    Format your response with clear headings for 'Executive Summary' and 'Campaign Recommendations'.
    Recommendations should be a bulleted list."""

    summary_text = ""
    recommendations_text = ""
    try:
        if ai_source == 'us':
            # Gemini API Call
            if 'GEMINI_API_KEY' not in st.secrets:
                st.error("Gemini API key not found in Streamlit secrets. Please configure it.")
                st.session_state.data_summary = "Error: Gemini API key not configured."
                st.session_state.campaign_recommendations = "Error: Gemini API key not configured."
                return

            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-pro') # Using gemini-pro for more robust generation
            response = model.generate_content(prompt)
            generated_text = response.text
        elif ai_source == 'openrouter':
            # OpenRouter AI API Call
            if not openrouter_api_key:
                st.error("OpenRouter API Key is required for OpenRouter AI analysis.")
                st.session_state.data_summary = "Error: OpenRouter API Key missing."
                st.session_state.campaign_recommendations = "Error: OpenRouter API Key missing."
                return

            headers = {
                'Authorization': f'Bearer {openrouter_api_key}',
                'Content-Type': 'application/json',
                # 'HTTP-Referer': 'https://your-deployed-app-url.streamlit.app', # Replace with your deployed app URL
                'X-Title': 'Media Intelligence Dashboard (Streamlit)'
            }
            data = {
                'model': openrouter_model,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False
            }
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status() # Raise an exception for HTTP errors
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
        else:
            st.error("Invalid AI source selected.")
            generated_text = "Error: Invalid AI source."

        # Parse the generated text into summary and recommendations
        summary_section_start = generated_text.find("Executive Summary")
        recommendations_section_start = generated_text.find("Campaign Recommendations")

        if summary_section_start != -1 and recommendations_section_start != -1:
            summary_text = generated_text[summary_section_start:recommendations_section_start].replace("Executive Summary", "").strip()
            recommendations_text = generated_text[recommendations_section_start:].replace("Campaign Recommendations", "").strip()
        else:
            summary_text = "Could not parse summary from AI response. Full response:\n" + generated_text
            recommendations_text = "Could not parse recommendations from AI response. Full response:\n" + generated_text

    except Exception as e:
        summary_text = f"Error generating analysis: {e}"
        recommendations_text = f"Error generating recommendations: {e}"

    st.session_state.data_summary = summary_text
    st.session_state.campaign_recommendations = recommendations_text
    st.session_state.generating_summary = False
    st.session_state.generating_recommendations = False
    st.rerun() # Rerun to update UI with final results

# --- Streamlit UI ---

st.title("Interactive Media Intelligence Dashboard")

# --- Section 1: CSV Upload ---
st.header("1. Upload Your CSV File")
st.markdown("""
Please upload a CSV file containing the following columns:
`Date, Platform, Sentiment, Location, Engagements, Media Type, Influencer Brand, Post Type`.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.session_state.file_name = uploaded_file.name
    st.session_state.processing = True
    st.session_state.error = ''
    st.session_state.csv_data = None
    st.session_state.cleaned_data = None
    st.session_state.filtered_data = None
    st.session_state.data_summary = ''
    st.session_state.campaign_recommendations = ''
    st.session_state.generating_summary = False
    st.session_state.generating_recommendations = False


    try:
        csv_text = uploaded_file.getvalue().decode("utf-8")
        st.session_state.csv_data = parse_csv(csv_text)
        st.session_state.cleaned_data = clean_data(st.session_state.csv_data)
        st.session_state.processing = False
        if st.session_state.cleaned_data.empty:
            st.session_state.error = 'No valid data found after cleaning. Please check your CSV file columns (Date, Engagements etc.)'
    except Exception as e:
        st.session_state.error = f'Failed to process file: {e}'
        st.session_state.processing = False

if 'file_name' in st.session_state and st.session_state.file_name:
    st.markdown(f"**File selected:** {st.session_state.file_name}")
    if st.session_state.processing:
        st.info("Processing...")
    elif st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.cleaned_data is not None and not st.session_state.cleaned_data.empty:
        st.success("File processed successfully. Data cleaned.")
    else:
        st.warning("No valid data found or file not yet uploaded.")

# --- Section 2: Data Cleaning (Implicitly handled) ---
st.header("2. Data Cleaning")
st.markdown("""
The uploaded data is automatically cleaned:
- 'Date' column converted to proper datetime objects.
- Missing 'Engagements' values are filled with 0.
- Column names are normalized for consistent processing.
""")

# --- Section: Data Filters ---
if 'cleaned_data' in st.session_state and st.session_state.cleaned_data is not None and not st.session_state.cleaned_data.empty:
    st.header("Data Filters")

    cols = st.columns(3)
    
    # Date Range Filter
    min_date = st.session_state.cleaned_data['date'].min().date() if not st.session_state.cleaned_data['date'].isnull().all() else datetime.date.today()
    max_date = st.session_state.cleaned_data['date'].max().date() if not st.session_state.cleaned_data['date'].isnull().all() else datetime.date.today()

    with cols[0]:
        filter_start_date = st.date_input("From Date:", value=min_date, min_value=min_date, max_value=max_date)
    with cols[1]:
        filter_end_date = st.date_input("To Date:", value=max_date, min_value=min_date, max_value=max_date)

    # Populate filter options
    platform_options = ['All'] + sorted(st.session_state.cleaned_data['platform'].unique().tolist())
    sentiment_options = ['All'] + sorted(st.session_state.cleaned_data['sentiment'].unique().tolist())
    media_type_options = ['All'] + sorted(st.session_state.cleaned_data['mediatype'].unique().tolist())
    location_options = ['All'] + sorted(st.session_state.cleaned_data['location'].unique().tolist())
    influencer_brand_options = ['All'] + sorted(st.session_state.cleaned_data['influencerbrand'].unique().tolist())
    post_type_options = ['All'] + sorted(st.session_state.cleaned_data['posttype'].unique().tolist())

    with cols[2]:
        filter_platform = st.selectbox("Platform:", platform_options)

    cols2 = st.columns(3)
    with cols2[0]:
        filter_sentiment = st.selectbox("Sentiment:", sentiment_options)
    with cols2[1]:
        filter_media_type = st.selectbox("Media Type:", media_type_options)
    with cols2[2]:
        filter_location = st.selectbox("Location:", location_options)

    cols3 = st.columns(2)
    with cols3[0]:
        filter_influencer_brand = st.selectbox("Influencer Brand:", influencer_brand_options)
    with cols3[1]:
        filter_post_type = st.selectbox("Post Type:", post_type_options)

    # Apply filters
    filtered_df = st.session_state.cleaned_data.copy()

    filtered_df = filtered_df[
        (filtered_df['date'].dt.date >= filter_start_date) &
        (filtered_df['date'].dt.date <= filter_end_date)
    ]
    if filter_platform != 'All':
        filtered_df = filtered_df[filtered_df['platform'] == filter_platform]
    if filter_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == filter_sentiment]
    if filter_media_type != 'All':
        filtered_df = filtered_df[filtered_df['mediatype'] == filter_media_type]
    if filter_location != 'All':
        filtered_df = filtered_df[filtered_df['location'] == filter_location]
    if filter_influencer_brand != 'All':
        filtered_df = filtered_df[filtered_df['influencerbrand'] == filter_influencer_brand]
    if filter_post_type != 'All':
        filtered_df = filtered_df[filtered_df['posttype'] == filter_post_type]

    st.session_state.filtered_data = filtered_df

    if st.button("Reset Filters"):
        # This will trigger a rerun and reset the date_input values if needed, but selectboxes usually reset if value is 'All'
        st.experimental_rerun() # Use rerun to clear all filter selections (including date inputs)

# --- Section 3 & 4: Interactive Charts and Insights ---
if 'filtered_data' in st.session_state and st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    st.header("3 & 4. Interactive Charts and Insights")

    # --- Sentiment Breakdown ---
    st.subheader("Sentiment Breakdown (Pie Chart)")
    sentiment_counts = st.session_state.filtered_data['sentiment'].value_counts()
    fig_sentiment = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                                          hole=.4, marker_colors=['#EF4444', '#FCD34D', '#10B981'])])
    fig_sentiment.update_layout(title_text='Sentiment Breakdown', font_family='Inter, sans-serif')
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_insights_sentiment(st.session_state.filtered_data):
        st.markdown(f"- {insight}")

    # --- Engagement Trend Over Time ---
    st.subheader("Engagement Trend Over Time (Line Chart)")
    daily_engagements = st.session_state.filtered_data.groupby(st.session_state.filtered_data['date'].dt.date)['engagements'].sum().sort_index()
    fig_engagement_trend = go.Figure(data=[go.Scatter(x=daily_engagements.index, y=daily_engagements.values,
                                                     mode='lines+markers', line=dict(color='#3B82F6'))])
    fig_engagement_trend.update_layout(title_text='Engagement Trend Over Time', xaxis_title='Date',
                                        yaxis_title='Total Engagements', font_family='Inter, sans-serif')
    st.plotly_chart(fig_engagement_trend, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_insights_engagement_trend(st.session_state.filtered_data):
        st.markdown(f"- {insight}")

    # --- Platform Engagements ---
    st.subheader("Platform Engagements (Bar Chart)")
    platform_engagements = st.session_state.filtered_data.groupby('platform')['engagements'].sum().sort_values(ascending=False)
    fig_platform_engagement = go.Figure(data=[go.Bar(x=platform_engagements.index, y=platform_engagements.values,
                                                    marker_color='#0EA5E9')])
    fig_platform_engagement.update_layout(title_text='Platform Engagements', xaxis_title='Platform',
                                          yaxis_title='Total Engagements', font_family='Inter, sans-serif')
    st.plotly_chart(fig_platform_engagement, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_insights_platform_engagement(st.session_state.filtered_data):
        st.markdown(f"- {insight}")

    # --- Media Type Mix ---
    st.subheader("Media Type Mix (Pie Chart)")
    media_type_counts = st.session_state.filtered_data['mediatype'].value_counts()
    fig_media_type_mix = go.Figure(data=[go.Pie(labels=media_type_counts.index, values=media_type_counts.values,
                                               hole=.4, marker_colors=['#EC4899', '#8B5CF6', '#14B8A6', '#F59E0B', '#6366F1'])])
    fig_media_type_mix.update_layout(title_text='Media Type Mix', font_family='Inter, sans-serif')
    st.plotly_chart(fig_media_type_mix, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_insights_media_type(st.session_state.filtered_data):
        st.markdown(f"- {insight}")

    # --- Top 5 Locations ---
    st.subheader("Top 5 Locations (Bar Chart)")
    location_counts = st.session_state.filtered_data['location'].value_counts().head(5)
    fig_top_locations = go.Figure(data=[go.Bar(x=location_counts.index, y=location_counts.values,
                                               marker_color='#65A30D')])
    fig_top_locations.update_layout(title_text='Top 5 Locations by Post Count', xaxis_title='Location',
                                     yaxis_title='Number of Posts', font_family='Inter, sans-serif')
    st.plotly_chart(fig_top_locations, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_insights_top_locations(st.session_state.filtered_data):
        st.markdown(f"- {insight}")

    # --- Section for Executive Summary & Recommendations Feature ---
    st.header("Generate Executive Summary & Recommendations")

    ai_source = st.radio(
        "Choose AI Source:",
        ('Analysis from Us (Gemini AI)', 'Analysis from OpenRouter AI'),
        key='ai_source_radio'
    )

    openrouter_api_key = ''
    openrouter_model = ''

    if ai_source == 'Analysis from OpenRouter AI':
        st.subheader("OpenRouter AI Settings")
        openrouter_api_key = st.text_input("OpenRouter API Key:", type="password", key='openrouter_api_key_input')
        openrouter_model = st.selectbox("Select AI Model:", list(OPENROUTER_MODELS.keys()), key='openrouter_model_select')
        openrouter_model = OPENROUTER_MODELS[openrouter_model] # Get the actual model string

    if st.button("Generate Analysis", key='generate_analysis_button',
                 disabled=st.session_state.generating_summary or st.session_state.generating_recommendations):
        # Collect all insights
        all_chart_insights = []
        all_chart_insights.extend(generate_insights_sentiment(st.session_state.filtered_data))
        all_chart_insights.extend(generate_insights_engagement_trend(st.session_state.filtered_data))
        all_chart_insights.extend(generate_insights_platform_engagement(st.session_state.filtered_data))
        all_chart_insights.extend(generate_insights_media_type(st.session_state.filtered_data))
        all_chart_insights.extend(generate_insights_top_locations(st.session_state.filtered_data))

        # Filter out default "No data" messages
        filtered_insights = [i for i in all_chart_insights if i != 'No data available for analysis.' and i != 'No engagement data available to analyze trend.' and i != 'No platform engagement data available.' and i != 'No media type data available for analysis.' and i != 'No location data available for analysis.']

        if not filtered_insights:
            st.warning("No sufficient data to generate summary and recommendations based on current filters and insights.")
            st.session_state.data_summary = "No sufficient data to generate summary."
            st.session_state.campaign_recommendations = "No recommendations generated."
        else:
            if ai_source == 'Analysis from Us (Gemini AI)':
                generate_data_summary_and_recommendations(filtered_insights, 'us', None, None)
            else: # OpenRouter
                generate_data_summary_and_recommendations(filtered_insights, 'openrouter', openrouter_api_key, openrouter_model)
    
    # --- Section 5: Data Summary ---
    st.header("5. Data Summary")
    if st.session_state.generating_summary:
        st.info("Generating summary...")
    else:
        st.markdown(st.session_state.get('data_summary', ''))

    # --- Section 6: Campaign Recommendations ---
    st.header("6. Campaign Recommendations")
    if st.session_state.generating_recommendations:
        st.info("Generating recommendations...")
    else:
        # Split and format recommendations as bullet points for better readability
        recommendations = st.session_state.get('campaign_recommendations', '')
        if recommendations:
            # Handle both Markdown bullet points and plain text lines
            lines = recommendations.split('\n')
            for line in lines:
                if line.strip().startswith('-') or line.strip().startswith('*'):
                    st.markdown(line)
                else:
                    st.write(line)
        else:
            st.write("") # Or a default message


else:
    st.info("Upload a CSV file above to see the interactive dashboard.")

# --- Branding Section ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small; color: gray;'>Powered by Gemini AI</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: small; color: gray;'>&copy; Copyright Media Intelligence Media Production</p>", unsafe_allow_html=True)
