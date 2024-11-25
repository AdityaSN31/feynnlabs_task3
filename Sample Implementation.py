# Core ML Service Implementation

import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import json
from typing import Dict, List
import pandas as pd
from datetime import datetime

class NutritionAI:
    def __init__(self):
        self.image_size = (224, 224)
        self.model = self.load_model()
        self.nutrition_db = self.load_nutrition_database()
        
    def load_model(self) -> tf.keras.Model:
        """Load the pre-trained model for Indian food recognition"""
        base_model = tf.keras.applications.EfficientNetB4(
            weights='imagenet',
            input_shape=(224, 224, 3),
            include_top=False
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(300, activation='softmax')  # 300 Indian dishes
        ])
        
        model.load_weights('indian_food_model.h5')
        return model
    
    def load_nutrition_database(self) -> pd.DataFrame:
        """Load the Indian food nutrition database"""
        return pd.read_csv('indian_food_nutrition.csv')
    
    async def process_image(self, image: UploadFile) -> Dict:
        """Process uploaded food image and return nutrition information"""
        # Read and preprocess image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img = img.resize(self.image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Get predictions
        predictions = self.model.predict(img_array)
        dish_index = np.argmax(predictions[0])
        confidence = float(predictions[0][dish_index])
        
        # Get nutrition information
        dish_info = self.nutrition_db.iloc[dish_index]
        
        return {
            'dish_name': dish_info['name'],
            'confidence': confidence,
            'nutrition': {
                'calories': float(dish_info['calories']),
                'protein': float(dish_info['protein']),
                'carbohydrates': float(dish_info['carbohydrates']),
                'fat': float(dish_info['fat']),
                'fiber': float(dish_info['fiber'])
            }
        }

class MealPlanner:
    def __init__(self):
        self.nutrition_db = pd.read_csv('indian_food_nutrition.csv')
        
    def generate_meal_plan(
        self,
        user_preferences: Dict,
        health_goals: Dict,
        restrictions: List[str]
    ) -> Dict:
        """Generate personalized meal plan based on user preferences and goals"""
        daily_calories = self.calculate_calorie_needs(
            age=user_preferences['age'],
            weight=user_preferences['weight'],
            height=user_preferences['height'],
            activity_level=user_preferences['activity_level'],
            goal=health_goals['type']
        )
        
        # Filter foods based on restrictions
        available_foods = self.nutrition_db[
            ~self.nutrition_db['name'].isin(restrictions)
        ]
        
        # Create meal plan
        meal_plan = {
            'breakfast': self.select_meal(
                available_foods,
                daily_calories * 0.3,
                meal_type='breakfast'
            ),
            'lunch': self.select_meal(
                available_foods,
                daily_calories * 0.4,
                meal_type='lunch'
            ),
            'dinner': self.select_meal(
                available_foods,
                daily_calories * 0.3,
                meal_type='dinner'
            )
        }
        
        return meal_plan
    
    def calculate_calorie_needs(
        self,
        age: int,
        weight: float,
        height: float,
        activity_level: str,
        goal: str
    ) -> float:
        """Calculate daily calorie needs using Harris-Benedict equation"""
        # Base metabolic rate
        bmr = 10 * weight + 6.25 * height - 5 * age
        
        # Activity multiplier
        activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        
        # Goal adjustment
        goal_adjustments = {
            'weight_loss': -500,
            'maintenance': 0,
            'weight_gain': 500
        }
        
        daily_calories = (
            bmr * 
            activity_multipliers[activity_level] + 
            goal_adjustments[goal]
        )
        
        return daily_calories

class GroceryOptimizer:
    def __init__(self):
        self.grocery_db = pd.read_csv('indian_grocery_prices.csv')
        
    def optimize_grocery_list(
        self,
        meal_plan: Dict,
        budget: float,
        location: str
    ) -> Dict:
        """Generate optimized grocery list based on meal plan and budget"""
        required_ingredients = self.extract_ingredients(meal_plan)
        local_prices = self.get_local_prices(location)
        
        optimized_list = self.knapsack_optimizer(
            required_ingredients,
            local_prices,
            budget
        )
        
        return {
            'items': optimized_list,
            'total_cost': sum(item['price'] for item in optimized_list),
            'nutritional_coverage': self.calculate_coverage(
                optimized_list,
                required_ingredients
            )
        }
    
    def knapsack_optimizer(
        self,
        ingredients: List[Dict],
        prices: Dict,
        budget: float
    ) -> List[Dict]:
        """Optimize grocery list using dynamic programming"""
        n = len(ingredients)
        K = [[0 for _ in range(int(budget) + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            for w in range(int(budget) + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif prices[ingredients[i-1]['name']] <= w:
                    K[i][w] = max(
                        ingredients[i-1]['nutritional_value'] + 
                        K[i-1][w-int(prices[ingredients[i-1]['name']])],
                        K[i-1][w]
                    )
                else:
                    K[i][w] = K[i-1][w]
        
        # Backtrack to find selected items
        selected_items = []
        w = int(budget)
        for i in range(n, 0, -1):
            if K[i][w] != K[i-1][w]:
                selected_items.append({
                    'name': ingredients[i-1]['name'],
                    'quantity': ingredients[i-1]['quantity'],
                    'price': prices[ingredients[i-1]['name']]
                })
                w -= int(prices[ingredients[i-1]['name']])
        
        return selected_items

# FastAPI Application Setup
app = FastAPI()
nutrition_ai = NutritionAI()
meal_planner = MealPlanner()
grocery_optimizer = GroceryOptimizer()

@app.post("/analyze_food")
async def analyze_food(image: UploadFile = File(...)):
    """Endpoint for food image analysis"""
    return await nutrition_ai.process_image(image)

@app.post("/generate_meal_plan")
async def generate_meal_plan(
    user_preferences: Dict,
    health_goals: Dict,
    restrictions: List[str]
):
    """Endpoint for meal plan generation"""
    return meal_planner.generate_meal_plan(
        user_preferences,
        health_goals,
        restrictions
    )

@app.post("/optimize_grocery")
async def optimize_grocery(
    meal_plan: Dict,
    budget: float,
    location: str
):
    """Endpoint for grocery list optimization"""
    return grocery_optimizer.optimize_grocery_list(
        meal_plan,
        budget,
        location
    )
