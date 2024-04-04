import logging.config
import json
import logging
import uvicorn
from app.app import app

'''def setup_logging():
    with open('logging_config.json', 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)

# Initialize logging
setup_logging()'''

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
