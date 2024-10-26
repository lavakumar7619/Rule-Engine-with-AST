import os
import time
import requests
from fastapi import FastAPI, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Constants
API_KEY = 'YOUR_API_KEY'  # Replace with your OpenWeatherMap API key
CITIES = ['Delhi', 'Mumbai', 'Chennai', 'Bangalore', 'Kolkata', 'Hyderabad']
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
DATABASE_URL = "mongodb://localhost:27017"
DATABASE_NAME = "weather_db"

# Initialize FastAPI and MongoDB client
app = FastAPI()
client = AsyncIOMotorClient(DATABASE_URL)
db = client[DATABASE_NAME]

# Model for weather data
class WeatherData(BaseModel):
    date: datetime
    avg_temp: float
    max_temp: float
    min_temp: float
    dominant_condition: str

# Function to convert Kelvin to Celsius
def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

# Fetch weather data from OpenWeatherMap
async def fetch_weather(city):
    params = {
        'q': city,
        'appid': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    return response.json()

# Process and store daily summaries
async def process_weather_data(city):
    weather_data = await fetch_weather(city)
    if weather_data and weather_data.get('main'):
        temp = kelvin_to_celsius(weather_data['main']['temp'])
        condition = weather_data['weather'][0]['main']

        today = datetime.now().date()
        
        # Calculate daily statistics
        daily_summary = {
            'date': today,
            'avg_temp': temp,
            'max_temp': temp,
            'min_temp': temp,
            'dominant_condition': condition
        }
        
        # Update or insert daily summary in MongoDB
        existing_record = await db.weather_data.find_one({'date': today})
        if existing_record:
            daily_summary['avg_temp'] = (existing_record['avg_temp'] + temp) / 2
            daily_summary['max_temp'] = max(existing_record['max_temp'], temp)
            daily_summary['min_temp'] = min(existing_record['min_temp'], temp)
            await db.weather_data.update_one({'date': today}, {'$set': daily_summary})
        else:
            await db.weather_data.insert_one(daily_summary)

# Background task to monitor weather
async def monitor_weather():
    while True:
        for city in CITIES:
            await process_weather_data(city)
        await asyncio.sleep(300)  # Use asyncio.sleep for asynchronous delay

@app.on_event("startup")
async def startup_event():
    # Start the background task
    app.add_task(monitor_weather())

@app.get("/weather/", response_model=List[WeatherData])
async def get_weather_data():
    cursor = db.weather_data.find()
    results = await cursor.to_list(length=None)
    return results

@app.get("/alert/")
async def check_alert(threshold: float):
    # Check for any alerts based on the threshold
    alerts = []
    async for doc in db.weather_data.find():
        if doc['max_temp'] > threshold:
            alerts.append(doc)
    return alerts

# Run the application
# Use Uvicorn to run the FastAPI application
# uvicorn main:app --reload
