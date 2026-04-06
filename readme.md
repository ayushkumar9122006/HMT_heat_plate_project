2-D Steady state heat transfer analysis | IIT Patna

A high-performance web application designed for ME 2202: Heat and Mass Transfer.
This engine solves 2D steady-state heat conduction problems using both numerical (Successive Over-Relaxation - SOR) and analytical (Fourier Series) methods.

It features real-time cloud-synced performance tracking and complexity analysis.
Key Features
   Dual Solver Engine: Compare Numerical FDM (SOR) results with Fourier Series Analytical solutions.
   Successive Over-Relaxation (SOR): Optimized numerical solver using PyTorch for accelerated computation.
   Coordinate Precision Probe: Query exact temperature at any (x, y) coordinate with 3-decimal accuracy tracking.
   System Complexity Profiles: Scatter plots showing Iteration $O(n)$ and Runtime $O(n^3)$ distributions across all historical data.Cloud Telemetry: Integrated with Supabase for real-time data logging and performance regression.
   Modern Dashboard: Responsive UI built with React and Plotly for high-fidelity heatmaps.
   
   🛠 Tech Stack
   Frontend: React.js, Vite, Plotly.js, Axios.
   Backend: FastAPI (Python), Uvicorn.Physics/Math: PyTorch, NumPy, Scikit-Learn, Pandas.
   Database: Supabase (PostgreSQL).
   
   💻 Local Installation
   1. Clone the Repository
        Bash git clone https://github.com/SAUBHAGYA-207/thermal-analysis-engine.git
        cd thermal-analysis-engine
2. Backend Setup
        Navigate to the root directory and install dependencies:Bash# Create a virtual environment (recommended)
        python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
Create a .env file in the root directory:
for the database credentials
Run the backend server:uvicorn main:app --reload
3. Frontend Setup
    Open a new terminal and navigate to the frontend folder:
        Bash cd frontend

# Install dependencies
npm install

# Create a frontend .env file
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
Run the development server:
    Bash npm run dev
📖 How to Use 
Launch the App: Open your browser at http://localhost:5173.
Input Parameters: Set the Mesh Resolution (m) and Boundary Temperatures (T_d, T_u, T_l, T_r).

Execute: Click Execute Analysis. Watch the real-time progress bar.
Analyze:Observe the FDM vs Analytical heatmaps.
Check the Estimated vs Actual metrics boxes.
Use the Coordinate Probe to verify local accuracy at specific points.
Performance: View the Complexity Trends to see how your current run fits the $O(n)$ and $O(n^3)$ performance curves.
📊 Database Schema
    To enable the Cloud Telemetry feature, ensure your Supabase database has a table named thermal_logs with the following columns:id: int8 (Primary Key)grid_size: float8t_top, t_bottom, t_left, t_right: float8iterations: int8time_taken: float8validation_score: float8
    
🎓 tDeveloped for the Department of Mechanical Engineering, IIT Patna.
Course: ME 2202 - Heat and Mass Transfer
Supervised by: Dr. Mohd. Kaleem Khan
