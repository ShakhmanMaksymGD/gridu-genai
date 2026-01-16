# Synthetic Data Generation & Analysis Platform

A comprehensive AI-driven platform for generating realistic synthetic data from database schemas and querying it with natural language using Google's Gemini 2.0 Flash.

## Features

### Core Features (Completed)
- 📊 **DDL Schema Parsing**: Parse SQL DDL files and extract table structures, constraints, and relationships
- 🤖 **AI-Powered Data Generation**: Use Gemini 2.0 Flash with function calling for realistic synthetic data
- 🔄 **Data Modification**: Modify generated data through natural language instructions
- 💾 **PostgreSQL Integration**: Store and manage data in PostgreSQL database
- 🎯 **Modular UI**: Clean, maintainable Streamlit interface with component-based architecture
- 📊 **Observability**: Langfuse integration for monitoring and analytics
- 🐳 **Containerized**: Docker support for easy deployment
- 💬 **Natural Language Querying**: Talk-to-your-data functionality with conversational AI interface
- 📈 **Data Visualization**: Auto-generated charts and plots from query results
- 🔒 **Security Guardrails**: SQL injection protection and prompt injection detection
- 💾 **Chat History**: Persistent conversation storage and management

### Planned Enhancements
- 🔍 **Advanced Analytics**: Enhanced statistical analysis and insights
- 📊 **Custom Visualization Options**: More chart types and customization
- 🚀 **Performance Optimization**: Query caching and optimization

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

### 4. Query Your Data
- Navigate to the "💬 Talk to your data" tab
- Ask questions about your generated data in natural language
- Examples: "Show me the average salary by department", "Create a chart of employee ages"
- The AI will generate SQL queries, execute them safely, and create visualizations

### 5. Modify Data
- In the Data Generation tab, select any table to modify
- Provide natural language instructions
- Example: "Make all employees older than 25", "Increase salaries by 10%"

## Architecture

```
├── app.py                          # Main Streamlit application (refactored)
├── src/
│   ├── chat/                            # Natural language querying system
│   │   ├── chat_interface.py            # Main chat interface with AI and visualization
│   │   ├── chat_history.py              # Chat conversation persistence
│   │   └── __init__.py                  # Package exports
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
│   │   ├── chat_ui.py                   # Chat interface UI components
│   │   └── pages.py                     # Page components and navigation
│   └── utils/
│       ├── ddl_parser.py                # DDL parsing logic
│       ├── langfuse_observer.py         # Observability integration
│       └── session_utils.py             # Session utilities
├── config/
│   └── settings.py                      # Configuration management
├── samplers/                            # Sample DDL schemas
├── data/                               # Generated data storage
└── docker-compose.yml                 # Container orchestration
```

## Sample DDL Schemas

The project includes three sample schemas:

1. **Company/Employee**: Companies, departments, employees, projects, benefits, reviews
2. **Library Management**: Authors, publishers, books, branches, members, loans
3. **Restaurant**: Restaurants, customers, orders, menu items, staff

## Natural Language Querying

The "Talk to your data" feature provides a conversational AI interface for querying your generated data:

### Key Capabilities
- **Natural Language Processing**: Ask questions in plain English about your data
- **SQL Generation**: AI automatically converts your questions into safe SQL queries
- **Data Visualization**: Automatic chart generation (bar, line, scatter, histogram)
- **Security**: Built-in SQL injection and prompt injection protection
- **Context Awareness**: Maintains conversation history for follow-up questions

### Example Queries
- "Show me the average salary by department"
- "Which employees earn more than $75,000?"
- "Create a chart showing the age distribution"
- "How many orders were placed last month?"
- "What's the most popular menu item?"

### Security Features
- Only SELECT queries are allowed (no data modification)
- Pattern matching prevents dangerous SQL operations
- Prompt injection detection blocks malicious inputs
- Query validation before execution

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
- Chat conversation analysis

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

### Adding New Chat Features

1. Extend `ChatInterface` class in `src/chat/chat_interface.py`
2. Add function declarations for new AI capabilities
3. Update UI components in `src/ui/chat_ui.py`

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

4. **Chat/Query issues**:
   - Ensure you have generated data before using the chat feature
   - Check that your natural language queries are clear and specific
   - Verify database connectivity if queries fail

5. **Docker issues**:
   - Ensure Docker has sufficient memory allocated
   - Check container logs: `docker-compose logs app`

### Logs

- Application logs: Check Docker logs or console output
- Database logs: `docker-compose logs postgres`
- Langfuse dashboard: Monitor LLM interactions and performance