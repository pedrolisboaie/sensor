import streamlit as st
import requests
import pandas as pd
from pandas import json_normalize
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
# Streamlit app configuration
st.set_page_config(
    page_title='Catapult 10hz data',
    page_icon='https://play-lh.googleusercontent.com/VcW9nvn7ILgmb-zof0Ez08JLTEZsRaNiCogVuw1NGwHdf4So0vl78nnU5jiwie3FI88',
    layout='wide'
)

def custom_title(title, image_url):
    st.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<img src='{image_url}' style='margin-right: 50px; width: 100px; height: 100px;'>"
        f"<h1>{title}</h1>"
        f"</div>",
        unsafe_allow_html=True
    )

image_url = "https://play-lh.googleusercontent.com/VcW9nvn7ILgmb-zof0Ez08JLTEZsRaNiCogVuw1NGwHdf4So0vl78nnU5jiwie3FI88"
custom_title("Catapult 10hz data", image_url)

def fetch_acceleration_data(token, activity_id, athlete_id):
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {token}"
    }

    acceleration_bands = [1, 2, 3]
    dfs = []  


    for band in acceleration_bands:
        url = f"https://connect-eu.catapultsports.com/api/v6/activities/{activity_id}/athletes/{athlete_id}/efforts?effort_types=acceleration&acceleration_bands={band}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()


            # Assuming the first item in the list contains the relevant data, let's try:
            try:
                acceleration_efforts = data[0]["data"]["acceleration_efforts"]
            except (TypeError, KeyError, IndexError):
                acceleration_efforts = []

            df_acceleration = pd.DataFrame(acceleration_efforts)
            df_acceleration['band'] = band  # Add a column indicating the band
            dfs.append(df_acceleration)
        else:
            st.error(f"Error fetching data for band {band}: {response.status_code}")

    # Concatenate dataframes for all bands into one
    return pd.concat(dfs, ignore_index=True)


# Fetch the sensor data
def fetch_sensor_data(token, activity_id, athlete_id):
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {token}"
    }

    url = f"https://connect-eu.catapultsports.com/api/v6/activities/{activity_id}/athletes/{athlete_id}/sensor"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df_data = json_normalize(data[0]['data'])
        metadata_cols = ['athlete_id', 'device_id', 'player_id', 'athlete_first_name', 'athlete_last_name', 'jersey', 'team_id', 'team_name', 'stream_type']
        for col in metadata_cols:
            df_data[col] = data[0][col]
        
        df_data['athlete_name'] = data[0]['athlete_first_name'] + ' ' + data[0]['athlete_last_name']
        return df_data
    else:
        st.error(f"Error: {response.status_code}")
        return None

def main():
    st.sidebar.header("Filters")
    entered_token = st.sidebar.text_input("Enter your authentication token:")
    selected_activity = None
    selected_athlete = None
    
    if entered_token:
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {entered_token}"
        }
        
        activities_url = "https://connect-eu.catapultsports.com/api/v6/activities"
        response_activities = requests.get(activities_url, headers=headers)
        
        st.sidebar.subheader("Select Activity:")
        if response_activities.status_code == 200:
            activities_data = response_activities.json()
            activity_names = [activity['name'] for activity in activities_data]
            selected_activity = st.sidebar.selectbox("", activity_names)
        else:
            st.sidebar.error(f"Error: {response_activities.status_code}")
        
        if selected_activity:
            for activity in activities_data:
                if activity['name'] == selected_activity:
                    selected_activity_id = activity['id']
                    break
            
            url = f"https://connect-eu.catapultsports.com/api/v6/activities/{selected_activity_id}/athletes"
            response = requests.get(url, headers=headers)
            st.sidebar.subheader("Select Athlete:")
            if response.status_code == 200:
                athletes_data = response.json()
                athlete_names = [f"{athlete['first_name']} {athlete['last_name']}" for athlete in athletes_data]
                selected_athlete = st.sidebar.selectbox("", athlete_names)
                for athlete in athletes_data:
                    if f"{athlete['first_name']} {athlete['last_name']}" == selected_athlete:
                        selected_athlete_id = athlete['id']
                        break
            else:
                st.sidebar.error(f"Error: {response.status_code}")

        
        if selected_athlete_id:
            df_sensor = fetch_sensor_data(entered_token, selected_activity_id, selected_athlete_id)
            df_acceleration = fetch_acceleration_data(entered_token, selected_activity_id, selected_athlete_id)
            grouped = df_sensor.groupby(df_sensor.index // 10)
            
            avg_df = grouped[['v', 'a']].mean().reset_index(drop=True)
            avg_df['v'] *= 3.6
            
            # Filter out rows where 'a' is less than 1
            avg_df = avg_df[avg_df['a'] >= 0.5]
                        # Convert 'cs' to a string format with two decimal places
            # Ensure that 'ts' is an integer
            df_sensor['ts'] = df_sensor['ts'].astype(int)
            
            # Ensure that 'cs' is a float rounded to 2 decimal places
            df_sensor['cs'] = df_sensor['cs'].round(2)
            
            # Convert 'cs' to a string, multiplied by 100 to avoid the decimal point
            # and then zero-padded to ensure two characters (e.g., 0.77 becomes '77', 0.05 becomes '05')
            df_sensor['cs'] = (df_sensor['cs'] * 100).astype(int).astype(str).str.zfill(2)
            
            # Concatenate 'ts' and 'cs' columns to form the 'timestamp' column
            df_sensor['timestamp'] = df_sensor['ts'].astype(str) + '.' + df_sensor['cs']
            
            # Convert the 'timestamp' column to float type
            df_sensor['timestamp'] = df_sensor['timestamp'].astype(float)
            
            min_v, max_v = st.sidebar.slider('Set range for v (x-axis)', 1, 36, (1, 36))
            min_a, max_a = st.sidebar.slider('Set range for a (y-axis)', 0.5, 4.5, (0.5, 4.5), step=0.5)
            filtered_avg_df = avg_df[(avg_df['v'] >= min_v) & (avg_df['v'] <= max_v) & 
                         (avg_df['a'] >= min_a) & (avg_df['a'] <= max_a)]

            
            #st.write(avg_df)
            st.subheader('A-S Profile', divider='red')
            def plot_with_lines(df: pd.DataFrame, x_range=(1, 35), y_range=(0.5, 4.5)):
    
                scatter = alt.Chart(df).mark_circle(color='orange', size=60, opacity=1).encode(
                    x=alt.X('v:Q', scale=alt.Scale(domain=list(x_range)), title='Velocity'),
                    y=alt.Y('a:Q', scale=alt.Scale(domain=list(y_range)), title='Acceleration'),
                    tooltip=['v', 'a']
                ).interactive()
                
                # Vertical lines
                vertical_lines_list = [7,14, 18, 20, 25]
                vertical_lines = alt.Chart(pd.DataFrame({'x': vertical_lines_list})).mark_rule(color='blue').encode(
                    x='x:Q'
                )
                
                # Horizontal lines
                horizontal_lines_list = [1.5, 2.5, 3.5]
                horizontal_lines = alt.Chart(pd.DataFrame({'y': horizontal_lines_list})).mark_rule(color='blue').encode(
                    y='y:Q'
                )
                
                # Combine the scatter chart with vertical and horizontal lines
                chart = scatter + vertical_lines + horizontal_lines
                
                st.altair_chart(chart, use_container_width=True)
            
            plot_with_lines(filtered_avg_df)
            
            def count_data_in_quadrants(df: pd.DataFrame, vertical_lines: list, horizontal_lines: list):
                # List to store rows before converting to dataframe
                data_list = []
            
                # Set boundary values
                v_min_bound = df['v'].min()
                a_min_bound = df['a'].min()
                v_max_bound = 36
                a_max_bound = 4.5
            
                vertical_lines = [v_min_bound] + vertical_lines + [v_max_bound]
                horizontal_lines = [a_min_bound] + horizontal_lines + [a_max_bound]
            
                # Adjust loops to include the last quadrant
                for i in range(len(vertical_lines) - 1):
                    for j in range(len(horizontal_lines) - 1):
                        count = df[(df['v'] >= vertical_lines[i]) & 
                                   (df['v'] < vertical_lines[i+1]) & 
                                   (df['a'] >= horizontal_lines[j]) & 
                                   (df['a'] < horizontal_lines[j+1])].shape[0]
            
                        # Append to the data list
                        data_list.append({
                            'v_start': vertical_lines[i],
                            'v_end': vertical_lines[i+1],
                            'a_start': horizontal_lines[j],
                            'a_end': horizontal_lines[j+1],
                            'count': count
                        })
            
                # Convert data list to dataframe
                quadrant_df = pd.DataFrame(data_list)
            
                return quadrant_df


            def format_label(label):
                # Check if the label is a range
                if '-' in label:
                    start, end = label.split('-')
                    return f"{int(float(start))}-{int(float(end))}"
                return int(float(label))
            
            def plot_heatmap(df: pd.DataFrame):
                # Reshape the dataframe for heatmap
                df_pivot = df.pivot_table(index=['a_start', 'a_end'], 
                                          columns=['v_start', 'v_end'], 
                                          values='count')
            
                # Plot heatmap
                plt.figure(figsize=(10, 6))
                sns.heatmap(df_pivot, annot=True, cmap='YlGnBu', fmt='g', cbar_kws={"label": "Count"})
                plt.xlabel("Velocity")
                plt.ylabel("Acceleration")
            
                # Adjust x-axis tick labels
                x_ticks = plt.xticks()
                x_labels = [format_label(label.get_text()) for label in x_ticks[1]]
                plt.xticks(x_ticks[0], x_labels)
            
                plt.gca().invert_yaxis()  # Reverse the y-axis for ascending acceleration values
            
                st.pyplot(plt)

            # Generate the quadrant counts
            vertical_lines_list = [7,14, 18, 20, 25]
            horizontal_lines_list = [1.5, 2.5, 3.5]
            quadrant_counts = count_data_in_quadrants(filtered_avg_df, vertical_lines_list, horizontal_lines_list)
            
            # Display the heatmap in Streamlit
            plot_heatmap(quadrant_counts)


            #st.header("Sensor Data")
            #if df_sensor is not None:
            #    st.write(df_sensor)

            st.header("Acceleration Efforts Data")
            if df_acceleration is not None:
                st.write(df_acceleration)
                
                
                # Extract relevant columns
                df_sensor_sub = df_sensor[['timestamp', 'athlete_name', 'v']]
                df_acceleration_sub = df_acceleration[['start_time', 'acceleration']]
                
            
                resulting_df = pd.merge(df_sensor_sub, df_acceleration_sub, left_on='timestamp', right_on='start_time', how='inner')
                
                resulting_df['v'] *= 3.6
            
                # Display the resulting dataframe
                st.write("Merged Data:")
                st.write(resulting_df)
                            
                
                def plot_scatter(df: pd.DataFrame):
                    chart = alt.Chart(df).mark_point().encode(
                        x=alt.X('v:Q', scale=alt.Scale(domain=[0, 37])),  # Setting the domain for x-axis
                        y=alt.Y('acceleration:Q', scale=alt.Scale(domain=[2, 6])),  # Setting the domain for y-axis
                        tooltip=['timestamp', 'athlete_name', 'v', 'acceleration']
                    ).interactive()
                
                    st.altair_chart(chart, use_container_width=True)
                
                # Assuming resulting_df is the merged dataframe you have after combining sensor and acceleration data.
                plot_scatter(resulting_df)


if __name__ == "__main__":
    main()
