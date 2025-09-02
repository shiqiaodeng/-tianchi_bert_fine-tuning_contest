import logging

class LogMng:
    def __init__(self, log_file = None):
        self.logger = logging.getLogger("BERT_MultiTask_Classifier")
        self.logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        if log_file:
            file_handler  = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler )
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s][%(name)s]: %(message)s')
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    def get_logger(self):
        return self.logger
    
logger = LogMng().get_logger()