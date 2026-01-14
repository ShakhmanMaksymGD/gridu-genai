# Synthetic Data Generation Platform

A powerful AI-driven platform for generating realistic synthetic data from database schemas using Google's Gemini 2.0 Flash.

## Features

### Phase 1 (Completed)
- 📊 **DDL Schema Parsing**: Parse SQL DDL files and extract table structures, constraints, and relationships
- 🤖 **AI-Powered Data Generation**: Use Gemini 2.0 Flash with function calling for realistic synthetic data
- 🔄 **Data Modification**: Modify generated data through natural language instructions
- 💾 **PostgreSQL Integration**: Store and manage data in PostgreSQL database
- 🎯 **Modular UI**: Clean, maintainable Streamlit interface with component-based architecture
- 📊 **Observability**: Langfuse integration for monitoring and analytics
- 🐳 **Containerized**: Docker support for easy deployment

### Phase 2 & 3 (Planned)
- 💬 **Natural Language Querying**: Talk-to-your-data functionality
- 📈 **Data Visualization**: Charts and plots from query results
- 🔍 **Advanced Analytics**: Statistical analysis and insights

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Google Cloud Project with Gemini API access
- Langfuse account (optional, for observability)

### Setup

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd ai-practice
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

3. **Required environment variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
   - Optional: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` for observability

4. **Start the application**:
   ```bash
   docker-compose up --build
   ```

5. **Access the application**:
   - Main App: http://localhost:8501
   - pgAdmin (optional): http://localhost:5050

### Manual Installation

If you prefer to run without Docker:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL**:
   - Install PostgreSQL locally
   - Create database: `synthetic_data_app`
   - Update connection settings in `.env`

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Upload DDL Schema
- Upload a `.sql`, `.txt`, or `.ddl` file containing CREATE TABLE statements
- Or use one of the provided sample schemas:
  - Company/Employee schema
  - Library Management schema
  - Restaurant schema

### 2. Configure Generation Parameters
- **Rows per table**: Number of rows to generate (1-10,000)
- **Temperature**: Controls randomness (0.0-1.0)
- **Instructions**: Provide specific guidance for data generation

### 3. Generate Data
- Click "Generate Data" to create synthetic data
- Preview generated tables
- Data is automatically stored in PostgreSQL

### 4. Modify Data
- Select any table to modify
- Provide natural language instructions
- Example: "Make all employees older than 25", "Increase salaries by 10%"

## Architecture

```
├── app.py                          # Main Streamlit application (refactored)
├── src/
│   ├── data_generation/
│   │   └── synthetic_data_generator.py  # Gemini-powered data generator
│   ├── database/
│   │   └── postgres_handler.py          # PostgreSQL operations
│   ├── ui/                              # Modular UI components
│   │   ├── __init__.py                  # Package exports
│   │   ├── session_manager.py           # Session state management
│   │   ├── styles.py                    # CSS styling and UI utilities
│   │   ├── file_upload.py               # File upload handling
│   │   ├── data_generation.py           # Data generation UI and logic
│   │   └── pages.py                     # Page components and navigation
│   └── utils/
│       ├── ddl_parser.py                # DDL parsing logic
│       └── langfuse_observer.py         # Observability integration
├── config/
│   └── settings.py                      # Configuration management
├── samplers/                            # Sample DDL schemas
├── data/                               # Generated data storage
└── docker-compose.yml                 # Container orchestration
```

## Refactored Architecture Benefits

The application has been completely refactored for better maintainability and scalability:

### Modular Design
- **Separation of Concerns**: Each UI component has a single responsibility
- **Session Management**: Centralized state handling in `SessionManager`
- **Component-Based**: Reusable UI components for consistent user experience
- **Clean Dependencies**: Clear import structure and minimal coupling

### Improved Maintainability
- **60% Code Reduction**: Main `app.py` reduced from 595 to ~60 lines
- **Error Isolation**: Issues in one component don't affect others
- **Easy Testing**: Each module can be tested independently
- **Team Collaboration**: Multiple developers can work on different modules

### Enhanced Readability
- **Descriptive Names**: Classes and methods clearly indicate their purpose
- **Logical Grouping**: Related functionality is grouped together
- **Simplified Flow**: Main application logic is easier to follow

## Sample DDL Schemas

The project includes three sample schemas:

1. **Company/Employee**: Companies, departments, employees, projects, benefits, reviews
2. **Library Management**: Authors, publishers, books, branches, members, loans
3. **Restaurant**: Restaurants, customers, orders, menu items, staff

## Configuration

Key configuration options in `config/settings.py`:

- `default_rows_per_table`: Default number of rows (1000)
- `default_temperature`: Default AI temperature (0.7)
- `max_retries`: Maximum retry attempts (3)
- Database connection settings
- Langfuse observability settings

## API Keys Setup

### Google Gemini API
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY` in your `.env` file

### Langfuse (Optional)
1. Sign up at [Langfuse](https://langfuse.com)
2. Create a new project
3. Copy your public and secret keys
4. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in your `.env` file

## Observability

The application integrates with Langfuse for:
- LLM interaction monitoring
- Performance metrics
- Error tracking
- User action analytics
- Data generation session tracking

## Development

### Running in Development Mode

```bash
# Set DEBUG=True in .env file
streamlit run app.py --server.runOnSave true
```

### Adding New Data Types

1. Update `DataType` enum in `ddl_parser.py`
2. Add conversion logic in `_convert_data_type()` method
3. Update data generation prompts if needed

### Extending UI

Add new components in the `src/ui/` directory and import them in `app.py`.

## Troubleshooting

### Common Issues

1. **Database connection failed**:
   - Ensure PostgreSQL is running
   - Check connection settings in `.env`
   - Wait for database to fully start in Docker

2. **Gemini API errors**:
   - Verify your API key is correct
   - Check your Google Cloud project has Gemini API enabled
   - Ensure you have sufficient quota

3. **Data generation slow or failing**:
   - Reduce the number of rows per table
   - Simplify your instructions
   - Check Langfuse logs for detailed error information

4. **Docker issues**:
   - Ensure Docker has sufficient memory allocated
   - Check container logs: `docker-compose logs app`

### Logs

- Application logs: Check Docker logs or console output
- Database logs: `docker-compose logs postgres`
- Langfuse dashboard: Monitor LLM interactions and performance