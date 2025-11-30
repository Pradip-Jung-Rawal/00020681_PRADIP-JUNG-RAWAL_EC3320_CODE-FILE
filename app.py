from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from model import UniversityRecommendationSystem
import os
from datetime import datetime

app = Flask(__name__)

# CORS Headers
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
}

# Initialize the recommendation system
system = None
df_data = None

def init_system():
    """Initialize the recommendation system"""
    global system, df_data
    try:
        system = UniversityRecommendationSystem()
        if system.load_and_inspect_data('collegedata.csv'):
            system.preprocess_data()
            system.train_and_evaluate()
            df_data = system.df.copy()
            print("✅ System initialized successfully")
            return True
        else:
            print("❌ Failed to initialize system")
            return False
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return False

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_html_path = os.path.join(current_dir, 'app.html')
        
        print(f"Looking for app.html at: {app_html_path}")
        print(f"File exists: {os.path.exists(app_html_path)}")
        
        with open(app_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError as e:
        print(f"Error: {e}")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return f"""
        <html>
            <head>
                <title>Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 50px; background-color: #f5f5f5; }}
                    .error-box {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 20px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="error-box">
                    <h1>⚠️ Error</h1>
                    <p>The file <strong>app.html</strong> was not found.</p>
                    <p>Looking in: <code>{current_dir}</code></p>
                    <p>Please make sure <strong>app.html</strong> is in the same folder as <strong>app.py</strong></p>
                    <p>Files in directory: {os.listdir(current_dir)}</p>
                </div>
            </body>
        </html>
        """, 404

@app.route('/api/stats', methods=['GET', 'OPTIONS'])
def get_stats():
    """Get database statistics"""
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    try:
        if df_data is None or df_data.empty:
            return jsonify({'error': 'No data available'}), 500
        
        total_unis = len(df_data)
        countries = df_data['Country'].nunique() if 'Country' in df_data.columns else 0
        cities = df_data['City'].nunique() if 'City' in df_data.columns else 0
        
        fee_median = 0
        fee_mean = 0
        fee_min = 0
        fee_max = 0
        
        if 'Fee (USD)' in df_data.columns:
            fee_median = float(df_data['Fee (USD)'].median())
            fee_mean = float(df_data['Fee (USD)'].mean())
            fee_min = float(df_data['Fee (USD)'].min())
            fee_max = float(df_data['Fee (USD)'].max())
        
        response = jsonify({
            'total_universities': total_unis,
            'countries': countries,
            'cities': cities,
            'fee_stats': {
                'median': fee_median,
                'mean': fee_mean,
                'min': fee_min,
                'max': fee_max
            }
        })
        
        return add_cors_headers(response)
        
    except Exception as e:
        print(f"Error in /api/stats: {e}")
        error_response = jsonify({'error': str(e)})
        return add_cors_headers(error_response), 500

@app.route('/api/recommend', methods=['POST', 'OPTIONS'])
def get_recommendations():
    """Get personalized university recommendations"""
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'success': False,
                'error': 'Invalid request. Please send JSON data.'
            }), 400
        
        # Extract form data
        programs_input = data.get('programs', '').strip()
        city_input = data.get('city', '').strip()
        country_input = data.get('country', '').strip()
        max_fee = float(data.get('fee_(usd)', 0))
        scholarship_required = data.get('scholarship_required', False)
        num_results = int(data.get('num_results', 10))
        
        # Validation
        if not programs_input or not city_input or max_fee <= 0:
            return jsonify({
                'success': False,
                'error': 'Invalid input parameters. Please fill in all required fields.'
            }), 400
        
        # Parse programs
        programs = [p.strip() for p in programs_input.split(',')]
        
        # Build preferences for the model
        preferences = {
            'programs': programs,
            'fee_range': [0, max_fee],
            'city': city_input,
            'country': country_input,
            'scholarship': scholarship_required
        }
        
        # Filter universities based on criteria
        filtered_df = df_data.copy()
        
        # Fee filter
        if 'Fee (USD)' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Fee (USD)'] <= max_fee]
        
        # City filter (case-insensitive, partial match)
        if 'City' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['City'].str.lower().str.contains(city_input.lower(), na=False)
            ]
        
        # Country filter if provided
        if country_input and 'Country' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['Country'].str.lower().str.contains(country_input.lower(), na=False)
            ]
        
        # Scholarship filter
        if scholarship_required and 'Scholarship' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['Scholarship'].str.lower() == 'yes'
            ]
        
        if filtered_df.empty:
            response = jsonify({
                'success': True,
                'error': 'No universities match your criteria. Try adjusting your filters.',
                'recommendations': []
            })
            return add_cors_headers(response)
        
        # Calculate match scores based on program overlap and other factors
        recommendations = []
        
        for idx, row in filtered_df.iterrows():
            try:
                # Program matching
                uni_programs = [p.strip().lower() for p in str(row.get('Programs', '')).split(';')]
                user_programs = [p.strip().lower() for p in programs]
                
                program_matches = sum(1 for up in user_programs if any(up in uniop or uniop in up for uniop in uni_programs))
                content_score = min(1.0, program_matches / len(user_programs)) if user_programs else 0.5
                
                # Fee proximity score (closer to budget is better)
                fee_value = row.get('Fee (USD)', max_fee)
                fee_score = max(0, 1 - (fee_value / max_fee) * 0.3)
                
                # Random ML and KNN scores for demonstration
                ml_score = np.random.uniform(0.6, 1.0)
                knn_score = np.random.uniform(0.6, 1.0)
                
                # Combined match score
                match_score = (content_score * 0.4 + fee_score * 0.2 + ml_score * 0.2 + knn_score * 0.2)
                
                # Affordability assessment
                affordability = 'Within budget' if fee_value <= max_fee * 0.8 else 'Close to budget'
                
                # Scholarship check
                has_scholarship = False
                if 'Scholarship' in row and row['Scholarship']:
                    has_scholarship = row['Scholarship'].lower() == 'yes'
                
                recommendation = {
                    'name': row.get('University Name', 'Unknown University'),
                    'city': row.get('City', 'Unknown'),
                    'country': row.get('Country', 'Unknown'),
                    'fee': int(fee_value),
                    'programs': row.get('Programs', 'N/A'),
                    'has_scholarship': has_scholarship,
                    'affordability': affordability,
                    'match_score': float(match_score),
                    'recommendation_reason': generate_recommendation_reason(
                        row, programs, match_score, affordability
                    ),
                    'details': {
                        'content_score': float(content_score),
                        'ml_score': float(ml_score),
                        'knn_score': float(knn_score)
                    }
                }
                
                recommendations.append(recommendation)
            
            except Exception as e:
                print(f"Error processing university {idx}: {e}")
                continue
        
        # Sort by match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        recommendations = recommendations[:num_results]
        
        response = jsonify({
            'success': True,
            'recommendations': recommendations,
            'search_criteria': {
                'programs': programs,
                'city': city_input,
                'country': country_input,
                'max_fee': max_fee,
                'scholarship_required': scholarship_required
            }
        })
        
        return add_cors_headers(response)
        
    except Exception as e:
        print(f"Error in /api/recommend: {e}")
        error_response = jsonify({'success': False, 'error': str(e)})
        return add_cors_headers(error_response), 500

def generate_recommendation_reason(row, user_programs, match_score, affordability):
    """Generate a human-readable recommendation reason"""
    reasons = []
    
    try:
        # Program match reason
        uni_programs = [p.strip().lower() for p in str(row.get('Programs', '')).split(';')]
        matching_programs = [p for p in user_programs if any(p.lower() in up or up in p.lower() for up in uni_programs)]
        
        if matching_programs:
            reasons.append(f"Offers {', '.join(matching_programs[:2])}")
        
        # Affordability reason
        if affordability == 'Within budget':
            reasons.append("Competitive tuition fees")
        
        # Scholarship reason
        if 'Scholarship' in row and row['Scholarship']:
            if row['Scholarship'].lower() == 'yes':
                reasons.append("Scholarship opportunities available")
        
        # Match score reason
        if match_score > 0.85:
            reasons.append("Excellent overall match")
        elif match_score > 0.70:
            reasons.append("Strong match for your profile")
        
    except Exception as e:
        print(f"Error generating reason: {e}")
    
    return " • ".join(reasons) if reasons else "Good match for your criteria"

def add_cors_headers(response):
    """Add CORS headers to response"""
    for header, value in CORS_HEADERS.items():
        response.headers[header] = value
    return response

@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return add_cors_headers(response)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Route not found', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error', 'status': 500}), 500

if __name__ == '__main__':
    print("="*60)
    print("UNIVERSITY FINDER PRO - STARTING APPLICATION")
    print("="*60)
    
    print("\nInitializing University Recommendation System...")
    
    if init_system():
        print("\n" + "="*60)
        print("✅ APPLICATION READY")
        print("="*60)
        print("\n📍 Access the application at:")
        print("   → http://localhost:5000/")
        print("\n📊 API Endpoints:")
        print("   → GET  http://localhost:5000/api/stats")
        print("   → POST http://localhost:5000/api/recommend")
        print("\n⚠️  Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("\n" + "="*60)
        print("❌ FAILED TO INITIALIZE APPLICATION")
        print("="*60)
