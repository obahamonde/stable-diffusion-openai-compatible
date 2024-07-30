from dotenv import load_dotenv

load_dotenv()
from src import create_app

app = create_app()
