# Doodle Recognition Backend

A FastAPI-based backend service for recognizing hand-drawn doodles using deep learning.

## Features

- RESTful API endpoints for doodle recognition
- User authentication and authorization
- Pre-trained deep learning model for doodle classification
- Input validation using Pydantic models
- Environment variable configuration

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd doodle-recogniser-3-main/Refactored\ Backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Running the Application

Start the development server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## Project Structure

```
.
├── app.py              # FastAPI application setup
├── config.py           # Configuration settings
├── models.py           # Database models
├── preprocessing.py    # Image preprocessing utilities
├── routes.py           # API route definitions
├── schemas.py          # Pydantic models
├── services.py         # Business logic
├── requirements.txt    # Project dependencies
└── .env       # Example environment variables
```

## Environment Variables

Copy `.env.example` to `.env` and update the values:

```
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
