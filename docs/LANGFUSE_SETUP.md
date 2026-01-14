# Langfuse Setup Guide

## 1. Create a Langfuse Account

1. Go to [langfuse.com](https://langfuse.com) and sign up
2. Create a new project 
3. Copy your API keys from the project settings

## 2. Configure Environment Variables

Add these to your `.env` file:

```bash
# Langfuse Configuration (optional)
LANGFUSE_SECRET_KEY=sk-lf-xxx-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-xxx-your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 3. Install Dependencies

Langfuse is already included in `requirements.txt`, but if you need to add it manually:

```bash
pip install langfuse
```

## 4. What You'll See in Langfuse

### 🔍 **Traces & Spans**
- **Data Generation Sessions**: Complete flow from schema upload to data generation
- **LLM Calls**: Every Gemini API request with prompts and responses
- **Table Generation**: Individual table creation with timing and success/failure

### 📊 **Events**
- **User Actions**: Schema uploads, data generation requests, table modifications
- **System Events**: Database connections, errors, performance metrics
- **Error Tracking**: Detailed error logs with stack traces

### 📈 **Analytics**
- **Performance Metrics**: Generation times, token usage, success rates
- **User Behavior**: Most used schemas, common modification patterns
- **Cost Tracking**: Token consumption and API usage

### 🎯 **Benefits**
- **Debug Issues**: See exactly what prompts were sent and what responses were received
- **Optimize Performance**: Identify slow operations and bottlenecks
- **Track Usage**: Monitor API costs and usage patterns
- **Improve Prompts**: A/B test different prompts and measure success rates

## 5. Viewing Your Data

1. Login to your Langfuse dashboard
2. Select your project
3. View traces in real-time as you use the app
4. Analyze performance and usage patterns
5. Set up alerts for errors or unusual usage

## 6. Example Traces

After setup, you'll see traces like:
- `schema_data_generation` - Full data generation session
- `user_action` - User interactions (uploads, modifications)
- `table_generated` - Individual table creation events
- `performance_metrics` - Timing and resource usage

The observability is completely optional - if you don't provide Langfuse credentials, the app works normally without tracking.