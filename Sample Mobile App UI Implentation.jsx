import React, { useState, useEffect } from 'react';
import { Camera } from 'lucide-react';
import { LineChart, XAxis, YAxis, Tooltip, Line } from 'recharts';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const NutritionDashboard = () => {
  const [nutritionData, setNutritionData] = useState({
    calories: [],
    protein: [],
    carbs: [],
    fat: []
  });

  const [selectedTab, setSelectedTab] = useState('dashboard');
  
  // Sample data for demonstration
  const sampleData = [
    { name: 'Mon', calories: 2100, protein: 75, carbs: 250, fat: 70 },
    { name: 'Tue', calories: 2200, protein: 80, carbs: 260, fat: 65 },
    { name: 'Wed', calories: 1950, protein: 72, carbs: 230, fat: 68 },
    { name: 'Thu', calories: 2050, protein: 78, carbs: 245, fat: 72 },
    { name: 'Fri', calories: 2150, protein: 82, carbs: 255, fat: 69 },
  ];

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Nutrition Coach</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-4">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Daily Calories</h3>
            <p className="text-2xl font-semibold text-gray-900">2,150</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Protein</h3>
            <p className="text-2xl font-semibold text-gray-900">82g</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Carbs</h3>
            <p className="text-2xl font-semibold text-gray-900">255g</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-sm font-medium text-gray-500">Fat</h3>
            <p className="text-2xl font-semibold text-gray-900">69g</p>
          </div>
        </div>

        {/* Nutrition Chart */}
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <h2 className="text-lg font-semibold mb-4">Weekly Nutrition Trends</h2>
          <div className="h-64">
            <LineChart width={800} height={200} data={sampleData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="calories" stroke="#8884d8" />
              <Line type="monotone" dataKey="protein" stroke="#82ca9d" />
              <Line type="monotone" dataKey="carbs" stroke="#ffc658" />
              <Line type="monotone" dataKey="fat" stroke="#ff7300" />
            </LineChart>
          </div>
        </div>

        {/* Food Recognition */}
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <h2 className="text-lg font-semibold mb-4">Food Recognition</h2>
          <button className="flex items-center justify-center w-full p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-gray-400">
            <Camera className="w-8 h-8 text-gray-400" />
            <span className="ml-2 text-gray-600">Take a photo of your meal</span>
          </button>
        </div>

        {/* Meal Suggestions */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-4">Today's Meal Suggestions</h2>
          <div className="space-y-4">
            <Alert>
              <AlertTitle>Breakfast (8:00 AM)</AlertTitle>
              <AlertDescription>
                Masala Dosa with Sambar - 450 calories
              </AlertDescription>
            </Alert>
            <Alert>
              <AlertTitle>Lunch (1:00 PM)</AlertTitle>
              <AlertDescription>
                Dal, Brown Rice, Mixed Vegetables - 650 calories
              </AlertDescription>
            </Alert>
            <Alert>
              <AlertTitle>Dinner (7:00 PM)</AlertTitle>
              <AlertDescription>
                Grilled Chicken with Roti and Salad - 550 calories
              </AlertDescription>
            </Alert>
          </div>
        </div>
      </main>
    </div>
  );
};

export default NutritionDashboard;
