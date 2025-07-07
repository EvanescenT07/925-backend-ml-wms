import logging

# Initialize logging configuration
# This module sets up logging for the Warehouse Management System (WMS) 
class LoggingConfig:
    def __init__(self):
        self.log_config()

    # Configure logging
    def log_config(self):
        logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("wms.log"),
            logging.StreamHandler()
        ]
    )